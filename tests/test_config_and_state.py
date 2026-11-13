import pytest
import yaml

from src.config import load_robot, load_task, camera_map, native_config
from src.cli import build_config
from src.state_machine import State, StateMachine
from conftest import Clock


def write(tmp_path, cfg):
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_robot_config(robot_config):
    assert camera_map(robot_config) == {'overhead': 'left_overhead', 'left_wrist': 'left_left_wrist',
                                        'right_wrist': 'right_right_wrist'}
    cfg = native_config(robot_config)
    assert set(cfg['left_arm_config']['cameras']) == {'overhead', 'left_wrist'}
    assert 'model' not in cfg['left_arm_config']['cameras']['overhead']


@pytest.mark.parametrize('field,value', [('max_step', 0), ('camera_max_age_s', float('nan')),
                                         ('action_max_age_s', -1)])
def test_invalid_safety(tmp_path, robot_config, field, value):
    robot_config['safety'][field] = value
    with pytest.raises(ValueError):
        load_robot(write(tmp_path, robot_config))


def test_duplicates_and_rotation(tmp_path, robot_config):
    robot_config['cameras']['overhead']['index_or_path'] = '/dev/video2'
    with pytest.raises(ValueError, match='Duplicate'):
        load_robot(write(tmp_path, robot_config))
    robot_config['cameras']['overhead']['index_or_path'] = '/dev/video0'
    robot_config['cameras']['overhead']['rotation'] = 270
    with pytest.raises(ValueError, match='rotation'):
        load_robot(write(tmp_path, robot_config))


def test_task_validation(tmp_path, task_config):
    assert load_task('configs/task.yaml')['stages'][1]['name'] == 'fold_2'
    task_config['stages'][0]['duration_s'] = 10
    with pytest.raises(ValueError, match='duration_s'):
        load_task(write(tmp_path, task_config))


def test_single_arm_configuration(tmp_path, robot_config):
    r, t = robot_config['robot'], robot_config['teleop']
    robot_config['robot'] = {**r['left_arm_config'], 'type': 'so101_follower', 'id': 'single_follower'}
    robot_config['teleop'] = {**t['left_arm_config'], 'type': 'so101_leader', 'id': 'single_leader'}
    cfg = load_robot(write(tmp_path, robot_config))
    assert camera_map(cfg)['overhead'] == 'overhead'
    assert 'cameras' in native_config(cfg)


def test_official_cli_configs_decode(monkeypatch):
    import draccus
    from lerobot.scripts.lerobot_record import RecordConfig
    from lerobot.scripts.lerobot_teleoperate import TeleoperateConfig
    from lerobot.configs.train import TrainPipelineConfig
    monkeypatch.setenv('DEVICE', 'cpu')
    for mode, cls in [('record', RecordConfig), ('teleoperate', TeleoperateConfig), ('train', TrainPipelineConfig)]:
        cfg = build_config(mode, 1, 'configs/robot.yaml')
        decoded = draccus.decode(cls, cfg)
        assert decoded is not None


def test_fsm_success_and_reset():
    m = StateMachine()
    m.start()
    assert m.state == State.FOLD_1
    m.execution_done()
    assert not any(m.verified)
    m.verify(True)
    assert m.state == State.FOLD_2
    m.execution_done()
    m.verify(True)
    assert m.state == State.COMPLETE and all(m.verified)
    with pytest.raises(RuntimeError):
        m.start()
    m.reset()
    assert m.state == State.READY and not any(m.verified)


def test_fsm_failure_timeout_and_invalid_transition():
    clock = Clock()
    m = StateMachine(clock)
    with pytest.raises(RuntimeError):
        m.verify(True)
    m.start()
    clock.sleep(2)
    with pytest.raises(TimeoutError):
        m.check_timeout(1)
    assert m.state == State.ABORTED
    m.reset()
    m.start()
    m.execution_done()
    m.verify(False)
    assert m.state == State.ABORTED


@pytest.mark.parametrize('steps', [0, 1, 2, 3, 4])
def test_abort_every_active_state(steps):
    m = StateMachine()
    transitions = [m.start, m.execution_done, lambda: m.verify(True), m.execution_done]
    for transition in transitions[:steps]:
        transition()
    m.abort('stop')
    assert m.state == State.ABORTED
