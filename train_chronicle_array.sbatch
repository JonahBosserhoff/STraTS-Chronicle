#!/bin/bash
# STraTS on a Chronicle cohort: several runs in parallel, one per GPU, as a SLURM job array.
# Every array task is one independent run of train_chronicle.sbatch (same hyperparameters), for
# one target and one seed (--run <r>o<NUM_RUNS>). Submit with submit_chronicle.sh, which sets
# the array size, or by hand from the repo root:
#   TARGETS="OS TTNT" NUM_RUNS=5 sbatch --array=0-9%4 train_chronicle_array.sbatch
# Task i trains target TARGETS[i / NUM_RUNS] with run (i % NUM_RUNS)+1, so the array must be
# 0-(#targets * NUM_RUNS - 1); %4 = at most 4 tasks (GPUs) at a time.
#SBATCH --job-name=strats_chronicle
#SBATCH --partition=gpu_nodes
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-4
#SBATCH --output=strats_chronicle-%A_%a.out

set -eo pipefail
read -r -a targets <<< "${TARGETS:-OS}"
num_runs=${NUM_RUNS:-5}
i=${SLURM_ARRAY_TASK_ID:?submit as an array job (sbatch --array=...)}
num_tasks=$(( ${#targets[@]} * num_runs ))
if (( i >= num_tasks )); then
    echo "Array task $i out of range: ${#targets[@]} targets x $num_runs runs = $num_tasks tasks (0-$((num_tasks-1)))" >&2
    exit 1
fi
target=${targets[$(( i / num_runs ))]}
run=$(( i % num_runs + 1 ))o${num_runs}
echo "Array task $i: target $target, run $run"

source ~/venvs/strats/bin/activate
export USE_TF=0   # transformers must not import TensorFlow
cd "$SLURM_SUBMIT_DIR/src"

# as train_chronicle.sbatch
python main.py --dataset chronicle --target "$target" --train_frac 1.0 --model_type strats \
    --hid_dim 64 --num_layers 2 --num_heads 16 --dropout 0.2 --attention_dropout 0.2 --lr 5e-4 \
    --max_epochs 50 --patience 10 --run "$run" --eval_during_training eval_train
