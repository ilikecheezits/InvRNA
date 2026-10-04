#!/bin/bash
# Submit experiments to Slurm on PSC Bridges-2.
#
#   ./submit.sh all          everything, with dependencies
#   ./submit.sh hitrate      one experiment
#   ./submit.sh sweep
#
# Set your allocation first:  export RNAGFN_ACCOUNT=xxxxxxx
set -euo pipefail
cd "$(dirname "$0")"

ACCOUNT="${RNAGFN_ACCOUNT:?set RNAGFN_ACCOUNT to your Bridges-2 allocation}"
ENV_NAME="${RNAGFN_ENV:-rnagfn}"
mkdir -p logs results figures

# BLAS stays single-threaded on purpose: the folding process pool already owns
# the cores, and oversubscription there is a real and easily missed slowdown.
PREAMBLE="module load anaconda3
source \$(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME
cd $PWD
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1"

submit() {           # submit <name> <sbatch-flags> <command>
  local name=$1 flags=$2 command=$3
  sbatch --parsable -A "$ACCOUNT" -J "rnagfn-$name" \
         -o "logs/${name}_%A_%a.out" -e "logs/${name}_%A_%a.err" \
         $flags --wrap="$PREAMBLE
$command"
}

CPU="-p RM-shared -N 1 --cpus-per-task=32 -t 04:00:00"
GPU="-p GPU-shared -N 1 --gpus=v100-32:1 --cpus-per-task=16 -t 06:00:00"

case "${1:-all}" in
  hitrate)
    submit hitrate "$CPU --array=0-7" \
      'python run.py hitrate --shard $SLURM_ARRAY_TASK_ID --n-shards 8 --n-samples 20000'
    ;;
  designability)
    submit designability "$CPU -t 06:00:00" \
      'python run.py designability --n-steps 1500 --n-walkers 512'
    ;;
  budget)
    submit budget "$GPU" 'python run.py budget --seconds 600'
    ;;
  sweep)
    # Size the array from:  python run.py sweep --list
    submit sweep "$GPU --array=0-17 -t 03:00:00" \
      'python run.py sweep --task-id $SLURM_ARRAY_TASK_ID --iterations 3000 --eval-samples 20000'
    ;;
  amortized)
    submit amortized "$GPU -t 12:00:00" \
      'python run.py amortized --iterations 20000 --batch-size 256 --checkpoint results/amortized.pt'
    ;;
  eterna)
    submit eterna "$CPU --cpus-per-task=16 -t 08:00:00 --array=0-9" \
      'python run.py eterna --shard $SLURM_ARRAY_TASK_ID --n-shards 10 --seconds 120 --max-puzzle-length 150'
    ;;
  all)
    # Only hitrate is a real dependency: the others rank targets from it.
    HIT=$(submit hitrate "$CPU --array=0-7" \
      'python run.py hitrate --shard $SLURM_ARRAY_TASK_ID --n-shards 8 --n-samples 20000')
    echo "hitrate        $HIT"
    echo "designability  $(submit designability "$CPU -t 06:00:00 -d afterok:$HIT" 'python run.py designability --n-steps 1500 --n-walkers 512')"
    echo "budget         $(submit budget "$GPU -d afterok:$HIT" 'python run.py budget --seconds 600')"
    echo "sweep          $(submit sweep "$GPU --array=0-17 -t 03:00:00 -d afterok:$HIT" 'python run.py sweep --task-id $SLURM_ARRAY_TASK_ID --iterations 3000 --eval-samples 20000')"
    echo "amortized      $(submit amortized "$GPU -t 12:00:00" 'python run.py amortized --iterations 20000 --batch-size 256 --checkpoint results/amortized.pt')"
    echo "eterna         $(submit eterna "$CPU --cpus-per-task=16 -t 08:00:00 --array=0-9" 'python run.py eterna --shard $SLURM_ARRAY_TASK_ID --n-shards 10 --seconds 120 --max-puzzle-length 150')"
    echo
    echo "when they finish:  python run.py figures && python run.py summary"
    ;;
  *)
    echo "usage: $0 {all|hitrate|designability|budget|sweep|amortized|eterna}" >&2
    exit 1
    ;;
esac
