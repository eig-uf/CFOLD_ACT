#!/usr/bin/env bash
# Supply resources at submission: sbatch --account=... --partition=... --qos=... --gres=gpu:1 ...
set -euo pipefail
cd -- "${SLURM_SUBMIT_DIR:?Submit from the CFOLD_ACT project directory}"
export PYTHON="${PYTHON:-$PWD/.venv/bin/python}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
stage="${1:?Usage: sbatch [cluster resources] scripts/train_slurm.sh 1 or 2}"
case "$stage" in
  1|2) exec bash "scripts/train_fold${stage}.sh" ;;
  *) echo 'Stage must be 1 or 2' >&2; exit 2 ;;
esac
