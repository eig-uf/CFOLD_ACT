#!/usr/bin/env bash
set -euo pipefail
command -v v4l2-ctl >/dev/null || { echo 'Install v4l-utils (Arch: sudo pacman -S v4l-utils)' >&2; exit 1; }
v4l2-ctl --list-devices
for directory in /dev/v4l/by-id /dev/v4l/by-path; do
  if [[ -d "$directory" ]]; then ls -l "$directory"; fi
done
shopt -s nullglob
for device in /dev/video*; do
  echo "Device: $device"
  v4l2-ctl --device "$device" --all --list-formats-ext || true
done
echo 'Multiple nodes may belong to one UVC camera. Assign identical wrists by USB topology.'
