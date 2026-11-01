#!/bin/bash
# STraTS on a Chronicle cohort (OS), one run on one GPU. See "Benchmarking on Chronicle data" in README.md.
# Requires data/processed/chronicle.pkl (preprocess_chronicle.sbatch). Submit from the repo root:
#   sbatch train_chronicle.sbatch
#SBATCH --job-name=strats_os
#SBATCH --partition=gpu_nodes
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=strats_os-%j.out

source ~/venvs/strats/bin/activate
export USE_TF=0   # transformers must not import TensorFlow
cd "$SLURM_SUBMIT_DIR/src"

# hyperparameters of the STraTS (ss-) runs in run_main.sh (no self-supervised pretraining)
python main.py --dataset chronicle --train_frac 1.0 --model_type strats --hid_dim 64 --num_layers 2 \
    --num_heads 16 --dropout 0.2 --attention_dropout 0.2 --lr 5e-4 --max_epochs 50 --patience 10 --run 1o5
