#!/bin/bash
# STraTS on a Chronicle cohort (OS), 5 runs with different seeds in parallel, one GPU each.
# Same as train_chronicle.sbatch, but as a job array: task K runs with --run Ko5 (seed 2023+K) and
# writes to its own output dir. Requires data/processed/chronicle.pkl (src/preprocess_chronicle.py);
# do not preprocess inside the array, all tasks would write the same file. Submit from the repo root:
#   sbatch train_chronicle_array.sbatch
#SBATCH --job-name=strats_os
#SBATCH --partition=gpu_nodes
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=1-5
#SBATCH --output=strats_os-%A_%a.out

source ~/venvs/strats/bin/activate
export USE_TF=0   # transformers must not import TensorFlow
cd "$SLURM_SUBMIT_DIR/src"

# hyperparameters of the STraTS (ss-) runs in run_main.sh (no self-supervised pretraining)
python main.py --dataset chronicle --train_frac 1.0 --model_type strats --hid_dim 64 --num_layers 2 \
    --num_heads 16 --dropout 0.2 --attention_dropout 0.2 --lr 5e-4 --run ${SLURM_ARRAY_TASK_ID}o5
