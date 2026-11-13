import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

from src.config import make_native
from src.robot import Robot, validate_action, PortLease
from src.policy import Policy, validate_schema


class Bus:
    def __init__(self):
        self.is_connected = True
        self.writes = []
        self.fail = False

    def sync_read(self, register):
        if self.fail:
            raise OSError('Serial failure')
        return {'shoulder_pan': 0.}

    def sync_write(self, register, value):
        self.writes.append((register, value))

    def disconnect(self, disable_torque=True):
        assert disable_torque is False
        self.is_connected = False


def fake_robot(cfg):
    left, right = Bus(), Bus()
    native = SimpleNamespace(left_arm=SimpleNamespace(bus=left, cameras={}),
                             right_arm=SimpleNamespace(bus=right, cameras={}),
                             observation_features={'left_shoulder_pan.pos': float, 'right_shoulder_pan.pos': float},
                             action_features={'left_shoulder_pan.pos': float, 'right_shoulder_pan.pos': float})
    native.send_action = lambda action: left.sync_write('action', action)
    return Robot(cfg, native=native), left, right


@pytest.mark.parametrize('action', [
    {'left_shoulder_pan.pos': float('nan'), 'right_shoulder_pan.pos': 0.},
    {'left_shoulder_pan.pos': float('inf'), 'right_shoulder_pan.pos': 0.},
    {'left_shoulder_pan.pos': 91., 'right_shoulder_pan.pos': 0.},
    {'left_shoulder_pan.pos': 4., 'right_shoulder_pan.pos': 0.},
    {'left_shoulder_pan.pos': [1.], 'right_shoulder_pan.pos': 0.},
    {'left_shoulder_pan.pos': 0.},
    {'wrong.pos': 0., 'right_shoulder_pan.pos': 0.},
])
def test_reject_invalid_before_either_arm_moves(robot_config, action):
    robot, left, right = fake_robot(robot_config)
    robot.resume()
    with pytest.raises(ValueError):
        robot.send_action(action)
    assert robot.stopped
    # Only measured-position holds may have been written, never the invalid action.
    assert all(register == 'Goal_Position' for register, _ in left.writes + right.writes)


def test_fault_in_one_arm_still_holds_other_and_disconnects(robot_config):
    robot, left, right = fake_robot(robot_config)
    left.fail = True
    robot.disconnect()
    assert right.writes == [('Goal_Position', {'shoulder_pan': 0.})]
    assert not left.is_connected and not right.is_connected


def test_dry_run_never_writes_motor_register(robot_config):
    robot, left, right = fake_robot(robot_config)
    robot.observation_only = True
    robot.resume()
    with pytest.raises(RuntimeError):
        robot.send_action(dict.fromkeys(robot.action_features, 0.))
    robot.disconnect()
    assert not left.writes and not right.writes


def test_previous_command_step_limit(robot_config):
    robot, left, right = fake_robot(robot_config)
    robot.resume()
    robot.send_action(dict.fromkeys(robot.action_features, 2.5))
    with pytest.raises(ValueError, match='Per-step'):
        robot.send_action(dict.fromkeys(robot.action_features, -2.5))
    assert sum(register == 'action' for register, _ in left.writes) == 1


def test_stale_camera_prevents_group_command(robot_config):
    robot, left, right = fake_robot(robot_config)
    def stale(**kwargs):
        raise TimeoutError('Stale camera')
    robot.native.left_arm.cameras['overhead'] = SimpleNamespace(read_latest=stale)
    robot.resume()
    with pytest.raises(TimeoutError, match='Stale camera'):
        robot.send_action(dict.fromkeys(robot.action_features, 0.))
    assert robot.stopped and all(register == 'Goal_Position' for register, _ in left.writes + right.writes)


def test_port_lease_exclusion(tmp_path):
    a = PortLease([str(tmp_path / 'port')])
    try:
        with pytest.raises(RuntimeError):
            PortLease([str(tmp_path / 'port')])
    finally:
        a.close()
    PortLease([str(tmp_path / 'port')]).close()


@pytest.fixture
def native_schema(robot_config, tmp_path):
    robot_config['robot']['calibration_dir'] = str(tmp_path / 'calibration')
    for camera in robot_config['cameras'].values():
        camera.update(width=32, height=32)
    native = make_native(robot_config)
    from lerobot.datasets.utils import hw_to_dataset_features
    features = {**hw_to_dataset_features(native.observation_features, 'observation'),
                **hw_to_dataset_features(native.action_features, 'action')}
    return native, features


def test_native_schema_and_explicit_mapping(native_schema):
    native, features = native_schema
    assert len(features['action']['names']) == 12
    assert set(native.cameras) == {'overhead', 'left_wrist', 'right_wrist'}
    assert 'observation.images.right_right_wrist' in features
    # Different dict iteration order remains valid; dataset order is authoritative.
    validate_schema(features, dict(reversed(list(native.observation_features.items()))), native.action_features)
    from lerobot.datasets.utils import build_dataset_frame
    obs = {name: float(i) for i, name in enumerate(features['observation.state']['names'])}
    for k, shape in native.observation_features.items():
        if isinstance(shape, tuple):
            obs[k] = np.zeros(shape, dtype=np.uint8)
    features['observation.state']['names'].reverse()
    frame = build_dataset_frame(features, obs, 'observation')
    assert frame['observation.state'][0] == 11
    from lerobot.policies.utils import make_robot_action
    import torch
    features['action']['names'].reverse()
    action = make_robot_action(torch.arange(12).reshape(1, 12), features)
    assert action[features['action']['names'][0]] == 0.
    features['action']['names'][0] = 'wrong.pos'
    with pytest.raises(ValueError, match='motors differ'):
        validate_schema(features, native.observation_features, native.action_features)


def test_camera_shape_validation(native_schema):
    native, features = native_schema
    features['observation.images.left_overhead']['shape'] = [640, 480, 3]
    with pytest.raises(ValueError, match='shape mismatch'):
        validate_schema(features, native.observation_features, native.action_features)


def test_existing_calibration_alias_is_reused(robot_config, tmp_path):
    directory = tmp_path / 'calibration'
    directory.mkdir()
    robot_config['robot']['calibration_dir'] = str(directory)
    robot_config['robot']['calibration_ids']['left'] = 'existing_left'
    content = json.dumps({'shoulder_pan': {'id': 1, 'drive_mode': 0, 'homing_offset': 0,
                                          'range_min': 0, 'range_max': 4095}})
    file = directory / 'existing_left.json'
    file.write_text(content)
    native = make_native(robot_config)
    assert native.left_arm.calibration_fpath == file
    assert native.left_arm.bus.calibration is native.left_arm.calibration
    assert native.left_arm.calibration['shoulder_pan'].range_max == 4095
    assert file.read_text() == content


@pytest.fixture
def checkpoint(tmp_path, native_schema):
    """Real, small ACT round-trip fixture; random weights, not a trained folding model."""
    import torch
    from lerobot.configs.types import FeatureType
    from lerobot.datasets.utils import dataset_to_policy_features
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors
    torch.set_num_threads(2)
    native, features = native_schema
    pf = dataset_to_policy_features(features)
    cfg = ACTConfig(device='cpu', chunk_size=4, n_action_steps=2, pretrained_backbone_weights=None,
                    dim_model=32, n_heads=4, dim_feedforward=64, n_encoder_layers=1, n_decoder_layers=1,
                    use_vae=False, input_features={k: v for k, v in pf.items() if k != 'action'},
                    output_features={'action': pf['action']})
    model = ACTPolicy(cfg).eval()
    stats = {k: {'mean': torch.zeros((3, 1, 1) if v.type == FeatureType.VISUAL else v.shape),
                 'std': torch.ones((3, 1, 1) if v.type == FeatureType.VISUAL else v.shape)} for k, v in pf.items()}
    pre, post = make_pre_post_processors(cfg, dataset_stats=stats)
    path = tmp_path / 'pretrained_model'
    model.save_pretrained(path)
    pre.save_pretrained(path)
    post.save_pretrained(path)
    (path / 'dataset_info.json').write_text(json.dumps({'features': features}))
    # Emulate a checkpoint trained on CUDA with ImageNet initialization enabled.
    data = json.loads((path / 'config.json').read_text())
    data.update(device='cuda', pretrained_backbone_weights='ResNet18_Weights.IMAGENET1K_V1')
    (path / 'config.json').write_text(json.dumps(data))
    for name in ('policy_preprocessor.json', 'policy_postprocessor.json'):
        p = path / name
        data = json.loads(p.read_text())
        for step in data['steps']:
            if 'device' in step.get('config', {}):
                step['config']['device'] = 'cuda'
        p.write_text(json.dumps(data))
    return path, native


def test_actual_act_processors_queue_and_offline_reload(checkpoint, monkeypatch):
    path, native = checkpoint
    monkeypatch.setenv('HF_HUB_OFFLINE', '1')
    policy = Policy(path, device='cpu', n_action_steps=2, local_files_only=True)
    policy.validate(native)
    obs = policy.mock_observation()
    for _ in range(3):
        action = policy.get_action(obs)
        assert set(action) == set(native.action_features) and all(np.isfinite(v) for v in action.values())
    assert len(policy.forward_ms) == 2  # Native queue, no forward on selection 2.
    assert len(policy.model._action_queue) == 1
    policy.reset()
    assert len(policy.model._action_queue) == 0
    policy.warmup(obs)
    assert not policy.model._action_queue
    obs.pop('left_overhead')
    with pytest.raises(ValueError, match='RGB image'):
        policy.get_action(obs)


def test_horizon_validation(checkpoint):
    path, _ = checkpoint
    with pytest.raises(ValueError, match='chunk_size'):
        Policy(path, n_action_steps=5)


def test_missing_stats_fail_instead_of_identity_normalization(checkpoint):
    path, _ = checkpoint
    p = path / 'policy_preprocessor.json'
    config = json.loads(p.read_text())
    for step in config['steps']:
        step.pop('state_file', None)
    p.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='normalization statistics'):
        Policy(path, n_action_steps=2)


def test_real_dry_run_cli(checkpoint, robot_config, task_config, tmp_path, monkeypatch):
    import yaml
    from src.deploy import main
    from lerobot.robots.so_follower import SOFollower
    path, _ = checkpoint
    def forbidden(*args, **kwargs):
        raise AssertionError('Dry-run attempted hardware access')
    monkeypatch.setattr(SOFollower, 'connect', forbidden)
    monkeypatch.setattr(SOFollower, 'send_action', forbidden)
    robot_config['robot']['calibration_dir'] = str(tmp_path / 'dry-calibration')
    for camera in robot_config['cameras'].values():
        camera.update(width=32, height=32)
    for stage in task_config['stages']:
        stage.update(checkpoint=str(path), metadata=str(path / 'dataset_info.json'), n_action_steps=2)
    r, t = tmp_path / 'robot.yaml', tmp_path / 'task.yaml'
    r.write_text(yaml.safe_dump(robot_config))
    t.write_text(yaml.safe_dump(task_config))
    assert main(['--robot-config', str(r), '--task-config', str(t), '--dry-run', '--benchmark-actions', '3']) == 0


@pytest.mark.skipif(not os.environ.get('CFOLD_CHECKPOINT'), reason='No trained compatible checkpoint supplied')
def test_trained_checkpoint_inference():
    policy = Policy(os.environ['CFOLD_CHECKPOINT'], os.environ.get('CFOLD_METADATA'),
                    n_action_steps=int(os.environ.get('CFOLD_HORIZON', '30')), local_files_only=True)
    action = policy.warmup(policy.mock_observation())
    assert set(action) == set(policy.features['action']['names'])
