"""Safety guards around native recording/teleoperation, without replacing their loops."""
import logging
import math
import signal
import sys

from .config import configure_calibration, load_robot
from .robot import PortLease, Robot


def main():
    mode, path, *cli_args = sys.argv[1:]
    cfg = load_robot(path)
    from lerobot.robots.so_follower import SOFollower
    from lerobot.teleoperators.so_leader import SOLeader
    from lerobot.robots.bi_so_follower import BiSOFollower
    connected = []
    adapters = []

    def no_calibration(self):
        raise RuntimeError(f'Automatic calibration disabled. Reuse/verify calibration separately: {self.calibration_fpath}')

    for cls in (SOFollower, SOLeader):
        original = cls.connect

        def guarded_connect(self, calibrate=True, _original=original):
            connected.append(self)
            configure_calibration(self, cfg, teleop=isinstance(self, SOLeader))
            if not self.calibration:
                no_calibration(self)
            self.bus.connect()
            try:
                if not self.is_calibrated:
                    no_calibration(self)
            finally:
                self.bus.disconnect(disable_torque=False)
            return _original(self, calibrate=False)

        cls.connect = guarded_connect
        cls.calibrate = no_calibration

    original_disconnect = SOFollower.disconnect

    def holding_disconnect(self):
        if self.bus.is_connected:
            try:
                q = self.bus.sync_read('Present_Position')
                if not all(math.isfinite(float(v)) for v in q.values()):
                    raise ValueError('Non-finite hold position')
                self.bus.sync_write('Goal_Position', q)
            except Exception:
                logging.exception('Follower hold failed; physical intervention may be required')
        return original_disconnect(self)

    SOFollower.disconnect = holding_disconnect

    cls = BiSOFollower if cfg['robot']['type'] == 'bi_so_follower' else SOFollower
    original_send = cls.send_action

    def guarded_send(native, action):
        adapter = next((a for a in adapters if a.native is native), None)
        if adapter is None:
            adapter = Robot(cfg, native=native)
            # Bind original method to avoid recursing through the safety guard.
            adapter.native_send = lambda a: original_send(native, a)
            adapters.append(adapter)
        try:
            adapter.check_cameras()
            adapter.validate_action(action)
            sent = adapter.native_send(action)
            adapter.last_action = dict(sent)
            return sent
        except BaseException:
            adapter.stop()
            raise

    cls.send_action = guarded_send
    r, t = cfg['robot'], cfg['teleop']
    ports = [a['port'] for group in (r, t) for a in
             ([group['left_arm_config'], group['right_arm_config']] if 'left_arm_config' in group else [group])]
    if len(set(ports)) != len(ports):
        raise ValueError('Leader and follower ports must be distinct')
    lease = PortLease(ports)
    def terminate(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, terminate)
    try:
        sys.argv = [f'lerobot-{mode}', *cli_args]
        if mode == 'record':
            from lerobot.scripts.lerobot_record import main as native_main
        else:
            from lerobot.scripts.lerobot_teleoperate import main as native_main
        native_main()
    finally:
        for adapter in adapters:
            adapter.stop()
        # Clean partial connections even if the native CLI failed before entering
        # its own cleanup scope. Never drop follower torque on error.
        for a in connected:
            for camera in getattr(a, 'cameras', {}).values():
                try:
                    if camera.is_connected:
                        camera.disconnect()
                except Exception:
                    logging.exception('Camera cleanup failed')
            if a.bus.is_connected:
                if isinstance(a, SOFollower):
                    try:
                        q = a.bus.sync_read('Present_Position')
                        if not all(math.isfinite(float(v)) for v in q.values()):
                            raise ValueError('Non-finite hold position')
                        a.bus.sync_write('Goal_Position', q)
                    except Exception:
                        logging.exception('Follower hold failed')
                try:
                    a.bus.disconnect(disable_torque=False)
                except Exception:
                    logging.exception('Bus cleanup failed')
        lease.close()


if __name__ == '__main__':
    main()
