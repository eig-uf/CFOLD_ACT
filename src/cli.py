"""Shared JSON configuration launcher for official LeRobot CLIs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from .config import load_robot, native_config


TASKS = {1: 'Fold the unfolded cloth in half into a rectangle.',
         2: 'Fold the half-folded cloth into a smaller square.'}


def env(name, default, cast=str):
    return cast(os.environ.get(name, default))


def boolean(value):
    if str(value).lower() not in ('true', 'false'):
        raise ValueError('Boolean environment values must be true or false')
    return str(value).lower() == 'true'


def build_config(mode, stage, robot_path):
    if mode == 'train':
        cfg = {
            'dataset': {'repo_id': env('DATASET_REPO_ID', f'ufeig/cfold-fold{stage}-v1')},
            'policy': {'type': 'act', 'device': env('DEVICE', 'cuda'), 'chunk_size': 100,
                       'n_action_steps': 30, 'repo_id': env('POLICY_REPO_ID', f'ufeig/cfold-act-fold{stage}-v1'),
                       'push_to_hub': env('PUSH_TO_HUB', 'false', boolean)},
            'output_dir': env('OUTPUT_DIR', f'outputs/fold{stage}'),
            'batch_size': env('BATCH_SIZE', '8', int), 'steps': env('STEPS', '50000', int),
            'save_freq': env('SAVE_FREQ', '5000', int), 'seed': env('SEED', '1000', int),
            'num_workers': env('NUM_WORKERS', '4', int), 'wandb': {'enable': False},
        }
    else:
        rcfg = load_robot(robot_path)
        cfg = {'robot': native_config(rcfg), 'teleop': native_config(rcfg, True),
               'display_data': env('DISPLAY_DATA', 'false', boolean)}
        if mode == 'record':
            cfg['dataset'] = {'repo_id': env('DATASET_REPO_ID', f'ufeig/cfold-fold{stage}-v1'),
                              'single_task': TASKS[stage], 'fps': rcfg['control']['fps'],
                              'episode_time_s': env('EPISODE_TIME_S', '18', float),
                              'reset_time_s': env('RESET_TIME_S', '15', float),
                              'num_episodes': env('NUM_EPISODES', '10', int),
                              'push_to_hub': env('PUSH_TO_HUB', 'false', boolean),
                              'video': env('VIDEO', 'true', boolean),
                              'vcodec': env('VCODEC', 'h264'),
                              'streaming_encoding': env('STREAMING_ENCODING', 'false', boolean),
                              'encoder_threads': env('ENCODER_THREADS', '2', int)}
            cfg['play_sounds'] = False
        else:
            cfg['fps'] = rcfg['control']['fps']
    if 'dataset' in cfg and os.environ.get('DATASET_ROOT'):
        cfg['dataset']['root'] = os.environ['DATASET_ROOT']
    return cfg


def bundle_checkpoints(output, repo_id, dataset_root=None, push=False, policy_repo=None):
    """Carry name ordering alongside the saved architecture, weights and processors."""
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    meta = LeRobotDatasetMetadata(repo_id, root=dataset_root)
    info = json.dumps(meta.info, indent=2)
    checkpoints = list(Path(output).glob('checkpoints/*/pretrained_model'))
    if not checkpoints:
        raise FileNotFoundError(f'No trained checkpoints under {output}')
    for checkpoint in checkpoints:
        (checkpoint / 'dataset_info.json').write_text(info)
    if push:
        from huggingface_hub import HfApi
        HfApi().upload_file(path_or_fileobj=info.encode(), path_in_repo='dataset_info.json', repo_id=policy_repo)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('record', 'teleoperate', 'train'))
    parser.add_argument('--stage', type=int, choices=(1, 2), default=1)
    parser.add_argument('--robot-config', default=os.environ.get('ROBOT_CONFIG', 'configs/robot.yaml'))
    parser.add_argument('--print-config', action='store_true', help='Print generated native CLI config without connecting')
    args, overrides = parser.parse_known_args(argv)
    cfg = build_config(args.mode, args.stage, args.robot_config)
    if args.print_config:
        print(json.dumps(cfg, indent=2))
        return
    # Native CLI overrides are supported for recording/teleop; hardware overrides
    # must be in shared YAML so safety validation sees the identical definition.
    if any(v.startswith(('--robot.', '--teleop.', '--config_path', '--dataset.fps', '--dataset.rename_map', '--fps')) for v in overrides):
        parser.error('Edit shared robot YAML to override hardware, FPS or feature naming')
    if args.mode == 'train' and overrides:
        parser.error('Use documented environment settings for training so metadata export matches the run')
    with tempfile.TemporaryDirectory(prefix='cfold-cli-') as temp:
        path = Path(temp) / 'config.json'
        path.write_text(json.dumps(cfg))
        if args.mode == 'train':
            import torch
            if cfg['policy']['device'].startswith('cuda') and not torch.cuda.is_available():
                raise RuntimeError('CUDA training requested but unavailable; install CUDA torch wheels or set DEVICE=cpu')
            completed = False
            try:
                subprocess.run([sys.executable, '-m', 'lerobot.scripts.lerobot_train', f'--config_path={path}'], check=True)
                completed = True
            finally:
                if completed or list(Path(cfg['output_dir']).glob('checkpoints/*/pretrained_model')):
                    bundle_checkpoints(cfg['output_dir'], cfg['dataset']['repo_id'], cfg['dataset'].get('root'),
                                       completed and cfg['policy']['push_to_hub'], cfg['policy']['repo_id'])
        else:
            subprocess.run([sys.executable, '-m', 'src.native_guard', args.mode,
                            args.robot_config, f'--config_path={path}', *overrides], check=True)


if __name__ == '__main__':
    main()
