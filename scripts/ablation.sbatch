#!/bin/bash
# Feature ablation of the trained STraTS Chronicle models (src/ablation.py), inference on the test split only.
# Settings in configs/ablation.yaml. Submit from the repo root:
#   sbatch scripts/ablation.sbatch                     # every model in one job
#   sbatch --array=0-3 scripts/ablation.sbatch         # the models split over 4 jobs (GPUs), same output_dir
#   CONFIG=configs/other.yaml sbatch scripts/ablation.sbatch   # path relative to the repo root
# Output: output_dir of the config (ablation_results.csv, ...), strats_ablation-<job id>.out (one per array task)
#SBATCH --job-name=strats_ablation
#SBATCH --partition=gpu_nodes
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=strats_ablation-%j.out

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
[[ -f src/main.py ]] || { echo "Submit from the repo root (sbatch scripts/ablation.sbatch)" >&2; exit 1; }
config=${CONFIG:-configs/ablation.yaml}
[[ "$config" = /* ]] || config="$SLURM_SUBMIT_DIR/$config"
source ~/venvs/strats/bin/activate
export USE_TF=0   # transformers must not import TensorFlow
cd src

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    # the array must be 0..N-1, so that every model is in exactly one task
    if (( SLURM_ARRAY_TASK_MIN != 0 || SLURM_ARRAY_TASK_MAX != SLURM_ARRAY_TASK_COUNT - 1 )); then
        echo "Submit the array as --array=0-<N-1> (here: $SLURM_ARRAY_TASK_MIN-$SLURM_ARRAY_TASK_MAX, $SLURM_ARRAY_TASK_COUNT tasks)" >&2
        exit 1
    fi
    python ablation.py --config "$config" --task "$SLURM_ARRAY_TASK_ID" --num_tasks "$SLURM_ARRAY_TASK_COUNT"
else
    python ablation.py --config "$config"
fi
