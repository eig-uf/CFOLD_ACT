"""Shared configuration translation for LeRobot 0.4.4."""
from copy import deepcopy
import math
from pathlib import Path

import yaml


def positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def camera_identity(device):
    path = Path(f'/dev/video{device}' if isinstance(device, int) else device).resolve()
    sysfs = Path('/sys/class/video4linux') / path.name / 'device'
    # UVC capture and metadata nodes, and multiple interfaces on the same USB
    # device, must not be mistaken for distinct cameras.
    if sysfs.exists():
        return str(sysfs.resolve().parent)
    return str(path)


def load_robot(path):
    with open(path) as stream:
        cfg = yaml.safe_load(stream)
    r = cfg['robot']
    if r['type'] not in ('bi_so_follower', 'so101_follower'):
        raise ValueError('Unsupported robot type')
    bi = r['type'] == 'bi_so_follower'
    arms = [r['left_arm_config'], r['right_arm_config']] if bi else [r]
    ports = [str(Path(a['port']).resolve()) for a in arms]
    if len(set(ports)) != len(ports):
        raise ValueError('Follower ports must differ')
    for arm in arms:
        if arm.get('disable_torque_on_disconnect', True):
            raise ValueError('Set disable_torque_on_disconnect=false to preserve holding torque')
        positive(arm['max_relative_target'], 'max_relative_target')
        if not arm.get('use_degrees', True):
            raise ValueError('This configuration uses degrees; use_degrees must be true')
    teleop = cfg['teleop']
    expected_teleop = 'bi_so_leader' if bi else 'so101_leader'
    if teleop['type'] != expected_teleop:
        raise ValueError(f'Teleoperator type must be {expected_teleop}')
    leader_arms = [teleop['left_arm_config'], teleop['right_arm_config']] if bi else [teleop]
    if any(not arm.get('use_degrees', True) for arm in leader_arms):
        raise ValueError('Leader and follower position units must both use degrees')
    all_ports = ports + [str(Path(a['port']).resolve()) for a in leader_arms]
    if len(set(all_ports)) != len(all_ports):
        raise ValueError('Leader and follower ports must all differ')
    if not r.get('id') or not teleop.get('id'):
        raise ValueError('Robot and teleoperator IDs are required to reuse calibration')
    for group in (r, teleop):
        ids = group.get('calibration_ids')
        if ids is not None:
            if set(ids) != ({'left', 'right'} if bi else {'single'}):
                raise ValueError('calibration_ids must specify each arm')
            if any(not name or Path(name).name != name or name in ('.', '..') for name in ids.values()):
                raise ValueError('Calibration IDs must be simple filenames without .json')
    if not isinstance(cfg['control']['fps'], int):
        raise ValueError('control.fps must be an integer')
    positive(cfg['control']['fps'], 'control.fps')
    for key in ('max_step', 'camera_max_age_s', 'action_max_age_s'):
        positive(cfg['safety'][key], key)
    for name, limits in cfg['safety']['joint_limits'].items():
        if len(limits) != 2 or not all(math.isfinite(v) for v in limits) or limits[0] >= limits[1]:
            raise ValueError(f'Invalid joint limits: {name}')
    devices = []
    for name, c in cfg['cameras'].items():
        if bi and c['arm'] not in ('left', 'right'):
            raise ValueError(f'Invalid camera arm for {name}')
        for k in ('width', 'height', 'fps'):
            positive(c[k], f'{name}.{k}')
        if c.get('rotation', 0) not in (0, 90, 180, -90):
            raise ValueError(f'Unsupported camera rotation: {name}')
        if c.get('color_mode', 'rgb') != 'rgb':
            raise ValueError('All policy cameras must use RGB')
        devices.append(camera_identity(c['index_or_path']))
    if len(set(devices)) != len(devices):
        raise ValueError('Duplicate camera assignments (including symlink aliases)')
    if not devices:
        raise ValueError('At least one camera required')
    return cfg


def load_task(path):
    with open(path) as stream:
        cfg = yaml.safe_load(stream)
    positive(cfg['task']['control_fps'], 'task.control_fps')
    if [s['name'] for s in cfg['stages']] != ['fold_1', 'fold_2']:
        raise ValueError('Exactly fold_1 and fold_2 stages are required')
    for s in cfg['stages']:
        for key in ('duration_s', 'timeout_s', 'n_action_steps'):
            positive(s[key], key)
        if s['duration_s'] > s['timeout_s']:
            raise ValueError('duration_s must not exceed timeout_s')
        if not isinstance(s['n_action_steps'], int):
            raise ValueError('n_action_steps must be an integer')
        if s.get('max_actions') is not None:
            positive(s['max_actions'], 'max_actions')
            if not isinstance(s['max_actions'], int):
                raise ValueError('max_actions must be an integer')
        if not isinstance(s['verify'], bool):
            raise ValueError('verify must be boolean')
        if not s['checkpoint']:
            raise ValueError('checkpoint is required')
    return cfg


def camera_map(cfg):
    """Logical name -> native raw observation key. Never infer from camera order."""
    bi = cfg['robot']['type'] == 'bi_so_follower'
    return {name: f"{c['arm']}_{name}" if bi else name for name, c in cfg['cameras'].items()}


def camera_definitions(cfg):
    return {name: {'type': 'opencv', **{k: v for k, v in c.items() if k not in ('arm', 'model')}}
            for name, c in cfg['cameras'].items()}


def native_config(cfg, teleop=False):
    r = deepcopy(cfg['teleop' if teleop else 'robot'])
    r.pop('calibration_ids', None)
    if not teleop:
        cameras = camera_definitions(cfg)
        for name, c in cfg['cameras'].items():
            target = r[f"{c['arm']}_arm_config"] if r['type'] == 'bi_so_follower' else r
            target.setdefault('cameras', {})[name] = cameras[name]
    if r.get('calibration_dir') is None:
        r.pop('calibration_dir', None)
    return r


def configure_calibration(arm, cfg, teleop=False):
    """Select an existing calibration file; never create or overwrite one."""
    group = cfg['teleop' if teleop else 'robot']
    ids = group.get('calibration_ids')
    if not ids:
        return
    if group['type'].startswith('bi_'):
        side = 'left' if arm.id == f"{group['id']}_left" else 'right'
    else:
        side = 'single'
    arm.calibration_fpath = arm.calibration_dir / f'{ids[side]}.json'
    arm.calibration = {}
    if arm.calibration_fpath.is_file():
        arm._load_calibration()
    arm.bus.calibration = arm.calibration


def make_native(cfg, teleop=False):
    import draccus
    if teleop:
        from lerobot.teleoperators import TeleoperatorConfig, make_teleoperator_from_config
        from lerobot.teleoperators import bi_so_leader, so_leader  # register native types
        native = make_teleoperator_from_config(draccus.decode(TeleoperatorConfig, native_config(cfg, True)))
    else:
        from lerobot.robots import RobotConfig, make_robot_from_config
        from lerobot.robots import bi_so_follower, so_follower
        from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
        native = make_robot_from_config(draccus.decode(RobotConfig, native_config(cfg)))
    for arm in ([native.left_arm, native.right_arm] if hasattr(native, 'left_arm') else [native]):
        configure_calibration(arm, cfg, teleop)
    return native
