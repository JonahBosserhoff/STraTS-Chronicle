#!/bin/bash
# STraTS on a Chronicle cohort, one run on one GPU. See "Benchmarking on Chronicle data" in README.md.
# Requires data/processed/chronicle.pkl (preprocess_chronicle.sbatch). Submit from the repo root:
#   sbatch train_chronicle.sbatch                                         # OS
#   TARGET=label_min_lab_hemoglobin_60_days sbatch train_chronicle.sbatch # any other target
# TARGET: OS, TTNT or any label_* column of the cohort (preprocess_chronicle.py prints the list).
# Several runs / targets in parallel: submit_chronicle.sh (train_chronicle_array.sbatch).
#SBATCH --job-name=strats_chronicle
#SBATCH --partition=gpu_nodes
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=strats_chronicle-%j.out

set -eo pipefail
source ~/venvs/strats/bin/activate
export USE_TF=0   # transformers must not import TensorFlow
cd "$SLURM_SUBMIT_DIR/src"

# hyperparameters of the STraTS (ss-) runs in run_main.sh (no self-supervised pretraining).
# --eval_during_training eval_train: test is not evaluated at every validation, only once with the
# best checkpoint at the end. Same checkpoint and same final test result, ~20% less time per epoch.
python main.py --dataset chronicle --target "${TARGET:-OS}" --train_frac 1.0 --model_type strats \
    --hid_dim 64 --num_layers 2 --num_heads 16 --dropout 0.2 --attention_dropout 0.2 --lr 5e-4 \
    --max_epochs 50 --patience 10 --run "${RUN:-1o5}" --eval_during_training eval_train
