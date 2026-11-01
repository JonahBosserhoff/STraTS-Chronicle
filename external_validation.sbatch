#!/bin/bash
# External validation of a trained STraTS Chronicle model (inference only), one GPU.
# Paths are set in external_validation.yaml (or another config via CONFIG, relative to src/).
# Submit from the repo root:
#   sbatch external_validation.sbatch
#   CONFIG=../external_validation_site_b.yaml sbatch external_validation.sbatch
#SBATCH --job-name=strats_ext
#SBATCH --partition=gpu_nodes
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=strats_ext-%j.out

source ~/venvs/strats/bin/activate
export USE_TF=0   # transformers must not import TensorFlow
cd "$SLURM_SUBMIT_DIR/src"

python external_validation.py --config ${CONFIG:-../external_validation.yaml}
