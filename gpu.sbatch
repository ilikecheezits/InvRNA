#!/bin/bash
#SBATCH --job-name=rnagfn-gpu
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --gres=gpu:v100-32:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# GPU experiments: budget, sweep, amortized.
#
#   sbatch -J rnagfn-budget gpu.sbatch budget --seconds 600
#   sbatch -J rnagfn-sweep --array=0-17 gpu.sbatch sweep --iterations 3000
#   sbatch -J rnagfn-amort --time=12:00:00 gpu.sbatch amortized --iterations 20000
#
# In an array job --task-id is appended automatically (size it from
# `python run.py sweep --list`). Submit from a shell with the env active.
#
# cpus-per-task is 8 rather than 32 because the GPU does the work here, but the
# folding pool still uses every core it is given -- raise it if the reward
# turns out to be the bottleneck for your target lengths.

set -euo pipefail
umask 027
cd "${RNAGFN_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
mkdir -p logs results figures

PY="${RNAGFN_PY:-python}"

# Fail fast rather than quietly burning a GPU allocation on the CPU path.
$PY - <<'PYCHK'
import sys
try:
    import torch
except ImportError:
    sys.exit("[!!] torch not installed in this environment")
if not torch.cuda.is_available():
    sys.exit("[!!] CUDA unavailable on this node -- refusing to run on CPU")
print(f"[*] CUDA ok: {torch.cuda.get_device_name(0)}")
PYCHK

ARGS=("$@")
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  ARGS+=(--task-id "$SLURM_ARRAY_TASK_ID")
fi

echo "[*] $(hostname)  cpus=${SLURM_CPUS_PER_TASK:-?}  run.py ${ARGS[*]}"
$PY run.py "${ARGS[@]}"
