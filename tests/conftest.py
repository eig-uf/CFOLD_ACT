from copy import deepcopy
from pathlib import Path
import pytest

from src.config import load_robot, load_task


@pytest.fixture(autouse=True)
def isolated_camera_topology(monkeypatch):
    """Keep synthetic camera paths independent of the host's real USB topology."""
    def identity(device):
        return str(Path(f'/dev/video{device}' if isinstance(device, int) else device).resolve())

    # Containers expose host sysfs even without camera device passthrough.
    # Path/symlink duplicate validation remains active for the fake devices.
    monkeypatch.setattr('src.config.camera_identity', identity)


@pytest.fixture
def robot_config():
    return load_robot('configs/robot.yaml')


@pytest.fixture
def task_config():
    cfg = load_task('configs/task.yaml')
    for s in cfg['stages']:
        s.update(duration_s=.11, timeout_s=1, max_actions=2)
    return cfg


class Clock:
    def __init__(self):
        self.now = 0.

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class MockRobot:
    def __init__(self):
        self.actions = []
        self.stopped = True
        self.stops = 0
        self.reads = 0
        self.disconnected = False

    def get_observation(self):
        self.reads += 1
        return {'q.pos': float(self.reads)}

    def resume(self):
        self.stopped = False

    def send_action(self, action, *, before_send=None):
        assert not self.stopped
        if before_send:
            before_send()
        self.actions.append(action)

    def stop(self):
        self.stopped = True
        self.stops += 1

    def disconnect(self):
        self.disconnected = True
        self.stop()


class MockPolicy:
    def __init__(self, callback=None):
        self.resets = 0
        self.callback = callback
        self.observations = []
        self.queue = []

    def reset(self):
        self.resets += 1
        self.queue.clear()

    def get_action(self, observation):
        self.observations.append(deepcopy(observation))
        self.queue = ['stale']
        if self.callback:
            self.callback()
        return {'q.pos': observation['q.pos']}

    def metrics(self):
        return {'forward_ms': 0., 'selection_ms': 0.}
