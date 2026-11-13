#!/usr/bin/env python
"""Simultaneous LeRobot camera-only test; no motor interfaces are instantiated."""
import argparse
import logging
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import camera_definitions, load_robot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--robot-config', default='configs/robot.yaml')
    parser.add_argument('--duration-s', type=float, default=15)
    parser.add_argument('--no-display', action='store_true')
    args = parser.parse_args()
    if args.duration_s <= 0:
        parser.error('duration must be positive')
    import cv2
    import draccus
    from lerobot.cameras import CameraConfig
    from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

    logging.basicConfig(level=logging.INFO)
    cfg = load_robot(args.robot_config)
    definitions = camera_definitions(cfg)
    cameras = {k: OpenCVCamera(draccus.decode(CameraConfig, c)) for k, c in definitions.items()}
    stats = {k: {'frames': 0, 'gaps': 0, 'last': None, 'latencies': [], 'read_ms': [], 'capture_ms': []} for k in cameras}
    for name, camera in cameras.items():
        original = camera._read_from_hardware
        def measured_capture(_read=original, _samples=stats[name]['capture_ms']):
            t = time.perf_counter()
            frame = _read()
            _samples.append(1000 * (time.perf_counter() - t))
            return frame
        # Instrument the pinned native capture method; capture remains LeRobot's.
        camera._read_from_hardware = measured_capture
    window = None
    labels = {}
    failed = False
    try:
        if not args.no_display:
            # LeRobot ships headless OpenCV. Use stdlib Tk + its existing Pillow
            # dependency for a minimal preview without conflicting cv2 installs.
            import tkinter as tk
            from PIL import Image, ImageTk
            window = tk.Tk()
            window.title('CFOLD cameras — close to exit')
            for i, name in enumerate(cameras):
                label = tk.Label(window, text=name, compound='top')
                label.grid(row=0, column=i)
                labels[name] = label
        for name, camera in cameras.items():
            camera.connect()
            cap = camera.videocapture
            values = {'width': cap.get(cv2.CAP_PROP_FRAME_WIDTH), 'height': cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
                      'fps': cap.get(cv2.CAP_PROP_FPS), 'fourcc': int(cap.get(cv2.CAP_PROP_FOURCC))}
            logging.info('%s (%s): actual capture %s', name, cfg['cameras'][name].get('model'), values)
            # LeRobot treats width/height as OUTPUT size; 90-degree rotations swap capture dimensions.
            expected = definitions[name]
            width, height = expected['width'], expected['height']
            if expected.get('rotation', 0) in (90, -90):
                width, height = height, width
            if values['width'] != width or values['height'] != height or abs(values['fps'] - expected['fps']) > 1:
                raise RuntimeError(f'{name}: capture settings do not match request')
        start = time.perf_counter()
        for s in stats.values():
            s['capture_ms'].clear()
        while time.perf_counter() - start < args.duration_s:
            for name, camera in cameras.items():
                t = time.perf_counter()
                frame = camera.read_latest(max_age_ms=int(cfg['safety']['camera_max_age_s'] * 1000))
                s = stats[name]
                s['read_ms'].append((time.perf_counter() - t) * 1000)
                with camera.frame_lock:
                    stamp = camera.latest_timestamp
                if stamp != s['last']:
                    if s['last'] is not None:
                        s['gaps'] += max(0, round((stamp - s['last']) * definitions[name]['fps']) - 1)
                    s['last'] = stamp
                    s['frames'] += 1
                    s['latencies'].append((time.perf_counter() - stamp) * 1000)
                    if window:
                        img = ImageTk.PhotoImage(Image.fromarray(frame).resize((400, 300)))
                        labels[name].configure(image=img)
                        labels[name].image = img
            if window:
                try:
                    window.update()
                except tk.TclError:
                    break
            time.sleep(.002)
        elapsed = time.perf_counter() - start
        for name, s in stats.items():
            hz = s['frames'] / elapsed
            mean = lambda x: sum(x) / len(x) if x else 0
            logging.info('%s: delivery=%.2f Hz, capture=%.2f Hz, estimated skipped frames=%d, '
                         'buffer age=%.2f ms, buffer read=%.2f ms, hardware capture wait=%.2f ms',
                         name, hz, len(s['capture_ms']) / elapsed, s['gaps'], mean(s['latencies']),
                         mean(s['read_ms']), mean(s['capture_ms']))
            if hz < .9 * definitions[name]['fps'] or s['gaps']:
                logging.error('%s: insufficient delivery or dropped frames', name)
                failed = True
        return int(failed)
    finally:
        for camera in cameras.values():
            if camera.is_connected:
                camera.disconnect()
        if window:
            try:
                window.destroy()
            except Exception:
                pass


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
