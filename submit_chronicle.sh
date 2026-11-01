#!/bin/bash
# Submit STraTS Chronicle runs in parallel (train_chronicle_array.sbatch): every target x NUM_RUNS
# seeds, each on its own GPU, at most MAX_PARALLEL at a time. From the repo root:
#   ./submit_chronicle.sh                                   # OS, 5 runs
#   ./submit_chronicle.sh OS TTNT label_min_lab_hemoglobin_60_days
#   NUM_RUNS=1 MAX_PARALLEL=8 ./submit_chronicle.sh OS TTNT ...
# Targets: OS, TTNT or any label_* column of the cohort (preprocess_chronicle.py prints the list).
set -euo pipefail
targets=("$@")
(( ${#targets[@]} )) || targets=(OS)
export TARGETS="${targets[*]}"
export NUM_RUNS=${NUM_RUNS:-5}
max_parallel=${MAX_PARALLEL:-4}
num_tasks=$(( ${#targets[@]} * NUM_RUNS ))
echo "Submitting $num_tasks runs (targets: $TARGETS; $NUM_RUNS runs each), at most $max_parallel at a time"
sbatch --array=0-$(( num_tasks - 1 ))%"$max_parallel" train_chronicle_array.sbatch
