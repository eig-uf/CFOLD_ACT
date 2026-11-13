"""Sequential, synchronous ACT deployment. Ctrl+C stops and holds both followers."""
import argparse
import json
import logging
from pathlib import Path
import signal
import threading
import time

from .config import load_robot, load_task
from .policy import Policy
from .robot import Robot
from .state_machine import State, StateMachine

log = logging.getLogger(__name__)


class OperatorStop(RuntimeError):
    pass


class StageExecutionDone(RuntimeError):
    pass


def run_policy(robot, policy, stage, fps, stop, machine, *, clock=time.monotonic, sleep=None):
    """One authoritative command source; all exits discard ACT's native queue."""
    sleep = sleep or stop.wait
    policy.reset()
    start = clock()
    deadline = start
    count = overruns = 0
    max_action_age = getattr(robot, 'config', {}).get('safety', {}).get('action_max_age_s', 2.)

    def check():
        if stop.is_set() or machine.state == State.ABORTED:
            raise OperatorStop('Operator stop')
        machine.check_timeout(stage['timeout_s'])
        return clock() - start < stage['duration_s']

    try:
        robot.resume()
        while check():
            if stage.get('max_actions') is not None and count >= stage['max_actions']:
                break
            observation = robot.get_observation()
            captured = clock()
            if not check():
                break
            action = policy.get_action(observation)
            if not check():
                break
            if clock() - captured > max_action_age:
                raise TimeoutError('Inference action is stale; increase CPU performance or revise validated action_max_age_s')
            def before_send():
                if not check():
                    raise StageExecutionDone
                if clock() - captured > max_action_age:
                    raise TimeoutError('Action became stale during hardware validation')
            try:
                robot.send_action(action, before_send=before_send)
            except StageExecutionDone:
                break
            count += 1
            deadline += 1 / fps
            now = clock()
            if now > deadline:
                overruns += 1
                # Skip missed slots; never send a burst of catch-up commands.
                deadline += (int((now - deadline) * fps) + 1) / fps
            sleep(max(0., min(deadline, start + stage['duration_s'], start + stage['timeout_s']) - clock()))
        elapsed = clock() - start
        metrics = {'elapsed_s': elapsed, 'actions': count, 'frequency_hz': count / elapsed if elapsed else 0.,
                   'overruns': overruns, **policy.metrics()}
        log.info('%s: %s', stage['name'], metrics)
        return metrics
    finally:
        # Hold even if queue reset or inference fails.
        try:
            policy.reset()
        finally:
            robot.stop()


def execute_attempt(robot, policies, cfg, confirm, stop, *, clock=time.monotonic, sleep=None):
    machine = StateMachine(clock)
    started = clock()
    results = []
    try:
        if stop.is_set() or not confirm('Workspace ready? Start motion [y/N]: '):
            raise OperatorStop('Workspace not confirmed')
        if stop.is_set():
            raise OperatorStop('Operator stop')
        machine.start()
        for i, stage in enumerate(cfg['stages']):
            log.info('Active stage %s, checkpoint=%s', stage['name'], stage['checkpoint'])
            results.append(run_policy(robot, policies[i], stage, cfg['task']['control_fps'], stop, machine,
                                      clock=clock, sleep=sleep))
            machine.execution_done()
            passed = confirm(f'Is fold {i + 1} successful? [y/N]: ') if stage['verify'] else None
            if stop.is_set():
                raise OperatorStop('Operator stop during verification')
            machine.verify(passed)
            log.info('Fold %d verification: %s', i + 1, passed)
            if machine.state == State.ABORTED:
                break
    except (Exception, KeyboardInterrupt) as exc:
        if machine.state not in (State.ABORTED, State.COMPLETE):
            machine.abort(str(exc) or 'Operator stop')
        log.exception('Attempt aborted: %s', machine.abort_reason)
    finally:
        try:
            for policy in policies:
                policy.reset()
        finally:
            robot.stop()
    return {'fold1_success': machine.verified[0], 'fold2_success': machine.verified[1],
            'total_duration_s': clock() - started,
            'success': machine.state == State.COMPLETE and all(machine.verified),
            'state': machine.state.name, 'abort_reason': machine.abort_reason, 'stages': results}


def confirm(prompt):
    try:
        return input(prompt).strip().lower() == 'y'
    except EOFError:
        return False


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--robot-config', default='configs/robot.yaml')
    parser.add_argument('--task-config', default='configs/task.yaml')
    parser.add_argument('--device')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--connect-hardware', action='store_true', help='Use real observations in dry-run; never sends actions')
    parser.add_argument('--benchmark-actions', type=int, default=31, help='Dry-run selections (31 covers a second H=30 chunk)')
    args = parser.parse_args(argv)
    if args.benchmark_actions < 1:
        parser.error('--benchmark-actions must be positive')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    rcfg, task = load_robot(args.robot_config), load_task(args.task_config)
    if task['task']['control_fps'] != rcfg['control']['fps']:
        raise ValueError('Task and hardware control FPS must match')
    device = args.device or task['task']['device']
    policies = [Policy(s['checkpoint'], s.get('metadata'), device, s['n_action_steps'],
                       task['task'].get('local_files_only', False), s.get('dataset')) for s in task['stages']]
    robot = Robot(rcfg)
    for policy in policies:
        policy.validate(robot)
        log.info('Loaded %s on %s; cameras=%s', policy.checkpoint, device, robot.camera_names)
    stop = threading.Event()

    def shutdown(signum, frame):
        stop.set()
        raise KeyboardInterrupt

    previous = {sig: signal.signal(sig, shutdown) for sig in (signal.SIGINT, signal.SIGTERM)}
    connected = False
    try:
        if not args.dry_run or args.connect_hardware:
            robot.connect(observation_only=args.dry_run)
            connected = True
        for policy in policies:
            obs = robot.get_observation() if connected else policy.mock_observation()
            action = policy.warmup(obs)
            if args.dry_run:
                for _ in range(args.benchmark_actions):
                    if stop.is_set():
                        raise OperatorStop('Operator stop')
                    action = policy.get_action(robot.get_observation() if connected else obs)
                policy.reset()
            log.info('Non-motion check: %d named targets; latency=%s', len(action), policy.metrics())
        if args.dry_run:
            log.info('Dry-run complete: no actions sent')
            return 0
        attempt = 0
        while not stop.is_set():
            attempt += 1
            result = {'attempt': attempt, **execute_attempt(robot, policies, task, confirm, stop)}
            log.info('Outcome: %s', result)
            if path := task['task'].get('evaluation_log'):
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                with open(path, 'a') as f:
                    f.write(json.dumps(result) + '\n')
            if stop.is_set() or not confirm('Prepare another attempt? [y/N]: '):
                return 0 if result['success'] else 1
        return 1
    except KeyboardInterrupt:
        log.warning('Operator stop')
        return 130
    finally:
        stop.set()
        try:
            for policy in policies:
                policy.reset()
        finally:
            robot.disconnect()
            for sig, handler in previous.items():
                signal.signal(sig, handler)


if __name__ == '__main__':
    raise SystemExit(main())
