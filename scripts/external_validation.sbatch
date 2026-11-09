#!/bin/bash
# External validation of a trained STraTS Chronicle model (inference only), one GPU.
# Settings in configs/external_validation.yaml. Submit from the repo root:
#   sbatch scripts/external_validation.sbatch
#   CONFIG=configs/external_validation_site_b.yaml sbatch scripts/external_validation.sbatch   # path relative to the repo root
#SBATCH --job-name=strats_ext
#SBATCH --partition=gpu_nodes
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=strats_ext-%j.out

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
[[ -f src/main.py ]] || { echo "Submit from the repo root (sbatch scripts/external_validation.sbatch)" >&2; exit 1; }
config=${CONFIG:-configs/external_validation.yaml}
[[ "$config" = /* ]] || config="$SLURM_SUBMIT_DIR/$config"
source ~/venvs/strats/bin/activate
export USE_TF=0   # transformers must not import TensorFlow
cd src

python external_validation.py --config "$config"
