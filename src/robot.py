"""Thin native LeRobot adapter with group validation and best-effort torque-on hold."""
import fcntl
import hashlib
import logging
import math
from pathlib import Path
import time

import numpy as np

from .config import camera_map, load_robot, make_native

log = logging.getLogger(__name__)


def arms(native):
    return [native.left_arm, native.right_arm] if hasattr(native, 'left_arm') else [native]


class PortLease:
    """Prevent this project's record/teleop/deploy processes sharing serial ports."""
    def __init__(self, ports):
        self.files = []
        try:
            for port in sorted(ports):
                key = hashlib.sha256(str(Path(port).resolve()).encode()).hexdigest()[:20]
                f = open(f'/tmp/cfold-port-{key}.lock', 'a')
                self.files.append(f)
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            self.close()
            raise RuntimeError('A follower port is already in use by CFOLD')

    def close(self):
        for f in self.files:
            f.close()
        self.files.clear()


def validate_action(action, names, positions, safety):
    if set(action) != set(names) or len(action) != len(names):
        raise ValueError(f'Action names must match robot schema: {list(names)}')
    for name in names:
        value = action[name]
        if np.ndim(value) != 0 or not math.isfinite(float(value)):
            raise ValueError(f'Non-finite or non-scalar action: {name}')
        bare = name.removeprefix('left_').removeprefix('right_')
        lo, hi = safety['joint_limits'].get(name, safety['joint_limits'].get(bare, (None, None)))
        if lo is None:
            raise ValueError(f'Missing joint limits: {name}')
        if not lo <= float(value) <= hi:
            raise ValueError(f'Joint limit exceeded: {name}={value}, bounds={lo, hi}')
        if not math.isfinite(float(positions[name])) or abs(float(value) - positions[name]) > safety['max_step']:
            raise ValueError(f'Per-step limit exceeded: {name}; align leader/follower poses before teleoperation')


class Robot:
    def __init__(self, config, native=None):
        self.config = load_robot(config) if isinstance(config, (str, Path)) else config
        self.native = native if native is not None else make_native(self.config)
        self.observation_features = self.native.observation_features
        self.action_features = self.native.action_features
        self.camera_names = camera_map(self.config)
        self.stopped = True
        self.last_action = None
        self.lease = None
        self.observation_only = False
        # Check limit coverage before connecting.
        for name in self.action_features:
            bare = name.removeprefix('left_').removeprefix('right_')
            if name not in self.config['safety']['joint_limits'] and bare not in self.config['safety']['joint_limits']:
                raise ValueError(f'Missing joint limits: {name}')

    def connect(self, observation_only=False):
        self.observation_only = observation_only
        self.lease = PortLease([a.config.port for a in arms(self.native)])
        try:
            for a in arms(self.native):
                if not a.calibration:
                    raise RuntimeError(f'Existing calibration required: {a.calibration_fpath}')
                # Preflight bus calibration BEFORE native configure enables torque.
                a.bus.connect()
                try:
                    if not a.is_calibrated:
                        raise RuntimeError(f'Motor calibration mismatch: {a.calibration_fpath}; calibrate separately')
                finally:
                    a.bus.disconnect(disable_torque=False)
            if observation_only:
                # Native connect configures motors and toggles torque. A camera/
                # observation dry-run must only open transports, with no writes.
                for a in arms(self.native):
                    a.bus.connect()
                    for camera in a.cameras.values():
                        camera.connect()
            else:
                self.native.connect(calibrate=False)
        except BaseException:
            self.disconnect()
            raise

    def positions(self):
        result = {}
        group = arms(self.native)
        for i, a in enumerate(group):
            prefix = ('left_' if i == 0 else 'right_') if len(group) == 2 else ''
            result.update({f'{prefix}{k}.pos': float(v) for k, v in a.bus.sync_read('Present_Position').items()})
        return result

    def get_observation(self):
        start = time.perf_counter()
        obs = self.native.get_observation()
        self.check_cameras()
        if time.perf_counter() - start > self.config['safety']['camera_max_age_s']:
            raise TimeoutError('Observation acquisition exceeded camera freshness budget')
        for key, shape in self.observation_features.items():
            if key not in obs:
                raise ValueError(f'Missing observation {key}')
            if isinstance(shape, tuple):
                if not isinstance(obs[key], np.ndarray) or obs[key].shape != shape or obs[key].dtype != np.uint8:
                    raise ValueError(f'Invalid RGB frame: {key}, expected uint8 {shape}')
            elif not math.isfinite(float(obs[key])):
                raise ValueError(f'Non-finite motor observation: {key}')
        return obs

    def check_cameras(self):
        for a in arms(self.native):
            for camera in a.cameras.values():
                camera.read_latest(max_age_ms=int(1000 * self.config['safety']['camera_max_age_s']))

    def resume(self):
        self.last_action = None
        self.stopped = False

    def validate_action(self, action):
        validate_action(action, self.action_features, self.positions(), self.config['safety'])
        if self.last_action is not None:
            validate_action(action, self.action_features, self.last_action, self.config['safety'])

    def send_action(self, action, *, before_send=None):
        if self.stopped or self.observation_only:
            raise RuntimeError('Robot stopped; explicitly resume at stage start')
        try:
            self.check_cameras()
            self.validate_action(action)
            # Position safety reads may themselves block. Recheck the execution
            # guard at the last point before entering the native group command.
            if self.stopped:
                raise RuntimeError('Robot stopped during action validation')
            if before_send is not None:
                before_send()
            sent = self.native.send_action(action)
            self.last_action = dict(sent if isinstance(sent, dict) else action)
            return sent
        except BaseException:
            self.stop()
            raise

    def hold(self):
        if self.observation_only:
            return
        # Read and command each arm independently: a failed arm must not prevent
        # attempting to hold the other. Bypass cameras and policy output entirely.
        for a in arms(self.native):
            if a.bus.is_connected:
                try:
                    q = a.bus.sync_read('Present_Position')
                    if not all(math.isfinite(float(v)) for v in q.values()):
                        raise ValueError('Invalid measured hold position')
                    a.bus.sync_write('Goal_Position', q)
                except Exception:
                    log.exception('Hold failed for %s; use physical power cutoff/support arms', a)

    def stop(self):
        self.stopped = True
        self.last_action = None
        self.hold()

    def disconnect(self):
        self.stop()
        # Native group disconnect requires BOTH arms/cameras connected. Release
        # components individually so partial-connect failures are cleaned up too.
        for a in arms(self.native):
            for camera in a.cameras.values():
                try:
                    if camera.is_connected:
                        camera.disconnect()
                except Exception:
                    log.exception('Camera disconnect failed')
            try:
                if a.bus.is_connected:
                    a.bus.disconnect(disable_torque=False)
            except Exception:
                log.exception('Bus disconnect failed')
        if self.lease:
            self.lease.close()
            self.lease = None
