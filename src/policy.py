"""ACT with saved LeRobot processors and explicit dataset feature ordering."""
import json
from pathlib import Path
import time

import numpy as np


def load_features(source):
    if isinstance(source, dict):
        return source.get('features', source)
    p = Path(source)
    if p.is_dir():
        p = p / 'meta/info.json'
    with p.open() as f:
        data = json.load(f)
    return data.get('features', data)


def validate_schema(features, observation_features, action_features):
    for key, hardware in (('observation.state', observation_features), ('action', action_features)):
        ft = features[key]
        names = ft['names']
        if not isinstance(names, list) or len(set(names)) != len(names) or list(ft['shape']) != [len(names)]:
            raise ValueError(f'Invalid feature names/shape for {key}')
        motors = {k for k, v in hardware.items() if v is float}
        if set(names) != motors:
            raise ValueError(f'{key} motors differ: dataset={names}, robot={sorted(motors)}')
    cameras = {k.removeprefix('observation.images.'): ft for k, ft in features.items()
               if k.startswith('observation.images.')}
    hardware_cameras = {k: v for k, v in observation_features.items() if isinstance(v, tuple)}
    if set(cameras) != set(hardware_cameras):
        raise ValueError(f'Camera mapping mismatch: dataset={list(cameras)}, robot={list(hardware_cameras)}')
    for key, ft in cameras.items():
        if tuple(ft['shape']) != tuple(hardware_cameras[key]):
            raise ValueError(f'Camera shape mismatch: {key}')


def validate_processors(pre, post, cfg):
    """LeRobot permits missing stats to act as identity; deployment must not."""
    from lerobot.processor import NormalizerProcessorStep, UnnormalizerProcessorStep
    for pipeline, cls, features in ((pre, NormalizerProcessorStep, cfg.input_features),
                                    (post, UnnormalizerProcessorStep, cfg.output_features)):
        steps = [s for s in pipeline.steps if isinstance(s, cls)]
        if len(steps) != 1:
            raise ValueError(f'Expected one saved {cls.__name__}')
        step = steps[0]
        for key, feature in features.items():
            saved = step.features.get(key)
            if saved is None or saved.type != feature.type or tuple(saved.shape) != tuple(feature.shape):
                raise ValueError(f'Processor/checkpoint feature mismatch: {key}')
            mode = cfg.normalization_mapping[feature.type.value]
            if step.norm_map.get(feature.type) != mode:
                raise ValueError(f'Processor/checkpoint normalization mismatch: {key}')
            required = {'IDENTITY': (), 'MEAN_STD': ('mean', 'std'), 'MIN_MAX': ('min', 'max'),
                        'QUANTILES': ('q01', 'q99'), 'QUANTILE10': ('q10', 'q90')}[mode.value]
            for stat in required:
                value = (step.stats or {}).get(key, {}).get(stat)
                if value is None or not np.isfinite(value).all():
                    raise ValueError(f'Missing or invalid saved normalization statistics: {key}.{stat}')
                if np.broadcast_shapes(np.shape(value), tuple(feature.shape)) != tuple(feature.shape):
                    raise ValueError(f'Normalization statistics shape mismatch: {key}.{stat}')


class Policy:
    def __init__(self, checkpoint, dataset_features=None, device='cpu', n_action_steps=30,
                 local_files_only=False, dataset=None):
        import torch
        from huggingface_hub import snapshot_download, hf_hub_download
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.datasets.utils import dataset_to_policy_features
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.policies.factory import make_pre_post_processors

        path = Path(checkpoint)
        if path.is_dir():
            if not (path / 'config.json').is_file():
                raise FileNotFoundError(f'{path}: point to the pretrained_model directory containing config.json')
        else:
            if str(checkpoint).startswith(('.', '/', 'checkpoints/')):
                raise FileNotFoundError(f'Local checkpoint missing: {checkpoint}')
            path = Path(snapshot_download(str(checkpoint), local_files_only=local_files_only))
        self.checkpoint = str(path)
        self.device = torch.device(device)
        if self.device.type not in ('cpu', 'cuda') or (self.device.type == 'cuda' and not torch.cuda.is_available()):
            raise ValueError(f'Unavailable inference device: {device}')
        if dataset_features is None:
            dataset_features = path / 'dataset_info.json'
            if not dataset_features.is_file():
                if not dataset:
                    raise FileNotFoundError('Supply dataset metadata or bundle dataset_info.json with the checkpoint')
                dataset_features = hf_hub_download(dataset, 'meta/info.json', repo_type='dataset',
                                                  local_files_only=local_files_only)
        self.features = load_features(dataset_features)
        cfg = PreTrainedConfig.from_pretrained(path)
        if cfg.type != 'act':
            raise ValueError('Checkpoint must contain an ACT policy')
        if type(n_action_steps) is not int or not 1 <= n_action_steps <= cfg.chunk_size:
            raise ValueError(f'n_action_steps must be between 1 and chunk_size={cfg.chunk_size}')
        expected = dataset_to_policy_features(self.features)
        for k, v in {**cfg.input_features, **cfg.output_features}.items():
            if k not in expected or expected[k].type != v.type or tuple(expected[k].shape) != tuple(v.shape):
                raise ValueError(f'Checkpoint and dataset feature mismatch: {k}')
        if set(cfg.input_features) != {k for k in expected if k.startswith('observation.')} or set(cfg.output_features) != {'action'}:
            raise ValueError('Unsupported ACT observation/output schema')
        cfg.device = str(self.device)
        cfg.n_action_steps = n_action_steps
        cfg.temporal_ensemble_coeff = None
        # Only skip ImageNet initialization; all backbone weights load from ACT.
        # This prevents torchvision reaching the network for LOCAL inference.
        cfg.pretrained_backbone_weights = None
        self.model = ACTPolicy.from_pretrained(path, config=cfg, strict=True).eval()
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            cfg, pretrained_path=str(path),
            preprocessor_overrides={'device_processor': {'device': str(self.device)},
                                   'normalizer_processor': {'device': str(self.device)}},
            postprocessor_overrides={'device_processor': {'device': 'cpu'},
                                    'unnormalizer_processor': {'device': str(self.device)}})
        validate_processors(self.preprocessor, self.postprocessor, cfg)
        self.selection_ms = []
        self.forward_ms = []
        self.model.model.register_forward_pre_hook(self._forward_start)
        self.model.model.register_forward_hook(self._forward_end)
        self.reset()

    def _sync(self):
        if self.device.type == 'cuda':
            import torch
            torch.cuda.synchronize(self.device)

    def _forward_start(self, module, args):
        self._sync()
        self._started = time.perf_counter()

    def _forward_end(self, module, args, result):
        self._sync()
        self.forward_ms.append(1000 * (time.perf_counter() - self._started))

    def validate(self, robot):
        validate_schema(self.features, robot.observation_features, robot.action_features)

    def mock_observation(self):
        obs = {name: 0. for name in self.features['observation.state']['names']}
        for key, ft in self.features.items():
            if key.startswith('observation.images.'):
                obs[key.removeprefix('observation.images.')] = np.zeros(ft['shape'], dtype=np.uint8)
        return obs

    def reset(self):
        self.model.reset()
        self.preprocessor.reset()
        self.postprocessor.reset()

    def get_action(self, observation):
        import torch
        from lerobot.datasets.utils import build_dataset_frame
        from lerobot.policies.utils import prepare_observation_for_inference, make_robot_action

        start = time.perf_counter()
        for name in self.features['observation.state']['names']:
            if name not in observation or not np.isfinite(observation[name]):
                raise ValueError(f'Missing or invalid motor observation: {name}')
        for key, ft in self.features.items():
            if key.startswith('observation.images.'):
                raw = observation.get(key.removeprefix('observation.images.'))
                if not isinstance(raw, np.ndarray) or raw.shape != tuple(ft['shape']) or raw.dtype != np.uint8:
                    raise ValueError(f'Missing or invalid RGB image: {key}')
        frame = build_dataset_frame(self.features, observation, prefix='observation')
        with torch.inference_mode():
            batch = prepare_observation_for_inference(frame, self.device)
            batch = self.preprocessor(batch)
            action = self.postprocessor(self.model.select_action(batch))
        if tuple(action.shape) != (1, len(self.features['action']['names'])) or not torch.isfinite(action).all():
            raise ValueError('ACT returned invalid action shape or non-finite values')
        result = make_robot_action(action, self.features)
        self.selection_ms.append(1000 * (time.perf_counter() - start))
        return result

    def warmup(self, observation):
        try:
            self.reset()
            return self.get_action(observation)
        finally:
            self.reset()

    def metrics(self):
        return {'forward_ms': float(np.mean(self.forward_ms)) if self.forward_ms else None,
                'selection_ms': float(np.mean(self.selection_ms)) if self.selection_ms else None}
