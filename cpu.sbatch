#!/bin/bash
#SBATCH --job-name=rnagfn-cpu
#SBATCH --partition=RM-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# CPU experiments: hitrate, designability, eterna, figures, summary.
# Folding is the whole cost and it parallelises across cores, so ask for many.
#
#   sbatch cpu.sbatch summary
#   sbatch -J rnagfn-hit --array=0-7 cpu.sbatch hitrate --n-samples 20000
#   sbatch -J rnagfn-eterna --array=0-9 cpu.sbatch eterna --seconds 120
#
# In an array job --shard/--n-shards are appended automatically.
# Submit from a shell with the conda env already active; sbatch inherits it.

set -euo pipefail
umask 027
cd "${RNAGFN_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
mkdir -p logs results figures

# Use the rnagfn env's interpreter directly so jobs do not depend on which
# conda env (if any) the submitting shell had active.
ENVPY="${RNAGFN_ENV:-$HOME/.conda/envs/rnagfn}/bin/python"
if [[ -n "${RNAGFN_PY:-}" ]]; then PY="$RNAGFN_PY"
elif [[ -x "$ENVPY" ]]; then PY="$ENVPY"
else PY=python; fi
$PY -c "import rnagfn, RNA" 2>/dev/null || {
  echo "[!!] $PY cannot import rnagfn + ViennaRNA from $PWD -- is the conda env active?" >&2
  exit 1
}

ARGS=("$@")
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  ARGS+=(--shard "$SLURM_ARRAY_TASK_ID"
         --n-shards "${RNAGFN_NSHARDS:-${SLURM_ARRAY_TASK_COUNT:-1}}")
fi

echo "[*] $(hostname)  cpus=${SLURM_CPUS_PER_TASK:-?}  run.py ${ARGS[*]}"
$PY run.py "${ARGS[@]}"
