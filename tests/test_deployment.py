import threading
import pytest

from src.deploy import execute_attempt, run_policy, OperatorStop
from src.state_machine import StateMachine, State
from conftest import Clock, MockRobot, MockPolicy


def attempt(cfg, answers, policies=None):
    clock, robot = Clock(), MockRobot()
    policies = policies or [MockPolicy(), MockPolicy()]
    answers = iter(answers)
    result = execute_attempt(robot, policies, cfg, lambda _: next(answers), threading.Event(),
                             clock=clock, sleep=clock.sleep)
    return result, robot, policies


def test_success_and_fresh_stage_observations(task_config):
    result, robot, policies = attempt(task_config, [True, True, True])
    assert result['success'] and result['state'] == 'COMPLETE'
    assert len(robot.actions) == 4
    assert policies[1].observations[0]['q.pos'] > policies[0].observations[-1]['q.pos']
    assert all(p.resets >= 3 and not p.queue for p in policies)
    assert robot.stopped


def test_failed_verification(task_config):
    result, robot, policies = attempt(task_config, [True, False])
    assert not result['success'] and result['state'] == 'ABORTED'
    assert not policies[1].observations and len(robot.actions) == 2


def test_disabled_verification_does_not_claim_success(task_config):
    for stage in task_config['stages']:
        stage['verify'] = False
    result, _, _ = attempt(task_config, [True])
    assert result['state'] == 'COMPLETE' and not result['success']


def test_stop_after_blocking_inference_discards_action(task_config):
    clock, robot, event = Clock(), MockRobot(), threading.Event()
    policy = MockPolicy(event.set)
    machine = StateMachine(clock)
    machine.start()
    with pytest.raises(OperatorStop):
        run_policy(robot, policy, task_config['stages'][0], 30, event, machine, clock=clock, sleep=clock.sleep)
    assert not robot.actions and robot.stopped and not policy.queue


def test_timeout_after_blocking_inference(task_config):
    clock, robot = Clock(), MockRobot()
    policy = MockPolicy(lambda: clock.sleep(2))
    machine = StateMachine(clock)
    machine.start()
    with pytest.raises(TimeoutError):
        run_policy(robot, policy, task_config['stages'][0], 30, threading.Event(), machine,
                   clock=clock, sleep=clock.sleep)
    assert machine.state == State.ABORTED and not robot.actions and robot.stopped


def test_duration_deadline_before_command(task_config):
    clock, robot = Clock(), MockRobot()
    policy = MockPolicy(lambda: clock.sleep(.2))
    machine = StateMachine(clock)
    machine.start()
    result = run_policy(robot, policy, task_config['stages'][0], 30, threading.Event(), machine,
                        clock=clock, sleep=clock.sleep)
    assert result['actions'] == 0 and not robot.actions and not policy.queue


def test_deadline_after_blocking_motor_validation(task_config):
    clock, robot = Clock(), MockRobot()
    original = robot.send_action
    def delayed(action, *, before_send=None):
        clock.sleep(.2)
        return original(action, before_send=before_send)
    robot.send_action = delayed
    m = StateMachine(clock)
    m.start()
    result = run_policy(robot, MockPolicy(), task_config['stages'][0], 30, threading.Event(), m,
                        clock=clock, sleep=clock.sleep)
    assert result['actions'] == 0 and robot.stopped


def test_abort_during_inference_discards_queue(task_config):
    clock, robot = Clock(), MockRobot()
    m = StateMachine(clock)
    m.start()
    policy = MockPolicy(lambda: m.abort('external stop'))
    with pytest.raises(OperatorStop):
        run_policy(robot, policy, task_config['stages'][0], 30, threading.Event(), m,
                   clock=clock, sleep=clock.sleep)
    assert not robot.actions and not policy.queue


def test_inference_failure_holds_and_aborts(task_config):
    def fail():
        raise ValueError('Bad output')
    result, robot, policies = attempt(task_config, [True], [MockPolicy(fail), MockPolicy()])
    assert result['state'] == 'ABORTED' and result['abort_reason'] == 'Bad output'
    assert robot.stopped and not robot.actions and not policies[0].queue


def test_hardware_failure_holds(task_config):
    clock, robot = Clock(), MockRobot()
    def fail():
        raise OSError('Camera disconnected')
    robot.get_observation = fail
    result = execute_attempt(robot, [MockPolicy(), MockPolicy()], task_config, lambda _: True,
                             threading.Event(), clock=clock, sleep=clock.sleep)
    assert result['abort_reason'] == 'Camera disconnected' and robot.stopped


def test_initial_stop_no_actions(task_config):
    clock, robot, stop = Clock(), MockRobot(), threading.Event()
    stop.set()
    result = execute_attempt(robot, [MockPolicy(), MockPolicy()], task_config, lambda _: True,
                             stop, clock=clock, sleep=clock.sleep)
    assert not robot.actions and not result['success']


def test_no_catchup_burst_and_reports_overruns(task_config):
    clock, robot = Clock(), MockRobot()
    stage = dict(task_config['stages'][0], duration_s=.6, max_actions=3)
    policy = MockPolicy(lambda: clock.sleep(.08))
    m = StateMachine(clock)
    m.start()
    result = run_policy(robot, policy, stage, 30, threading.Event(), m, clock=clock, sleep=clock.sleep)
    assert result['actions'] == 3 and result['overruns'] == 3 and result['frequency_hz'] < 30


def test_main_disconnects_on_warmup_failure(monkeypatch, tmp_path, robot_config, task_config):
    import yaml
    import src.deploy as deploy
    robot = MockRobot()
    robot.camera_names = {}
    robot.connect = lambda **kw: None
    policy = MockPolicy()
    policy.checkpoint = 'fake'
    policy.validate = lambda robot: None
    def fail(obs):
        raise RuntimeError('Warmup failure')
    policy.warmup = fail
    monkeypatch.setattr(deploy, 'Robot', lambda cfg: robot)
    monkeypatch.setattr(deploy, 'Policy', lambda *args: policy)
    r, t = tmp_path / 'robot.yaml', tmp_path / 'task.yaml'
    r.write_text(yaml.safe_dump(robot_config))
    t.write_text(yaml.safe_dump(task_config))
    with pytest.raises(RuntimeError, match='Warmup failure'):
        deploy.main(['--robot-config', str(r), '--task-config', str(t)])
    assert robot.disconnected and robot.stopped
