#!/bin/bash
# STraTS on the Chronicle cohort: every target x run of configs/train_chronicle.yaml, one GPU per run, as a SLURM job
# array. Requires data/processed/chronicle.pkl (scripts/preprocess_chronicle.sbatch). All settings are in the config.
#
#   bash scripts/train_chronicle.sbatch          # submits the array (with bash, not sbatch: it sets the array size)
#   bash scripts/train_chronicle.sbatch --list   # only shows the runs of the config, submits nothing
#   CONFIG=configs/other.yaml bash scripts/train_chronicle.sbatch   # another config (path relative to the repo root)
#
# The submitted runs use a copy of the config (outputs/submitted_configs/), so the config can be edited while they
# wait. sbatch scripts/train_chronicle.sbatch (from the repo root) works only for a config with a single run.
# Output: outputs/chronicle/<run folder>/ and strats_chronicle-<array job id>_<task>.out (first line: target and run).
#SBATCH --job-name=strats_chronicle
#SBATCH --partition=gpu_nodes
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=strats_chronicle-%j.out

set -eo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    # started with bash: submit every run of the config as one array job
    repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
    cd "$repo"
    config=${CONFIG:-configs/train_chronicle.yaml}
    [[ "$config" = /* ]] || config="$repo/$config"
    [[ -f "$config" ]] || { echo "Config not found: $config" >&2; exit 1; }
    source ~/venvs/strats/bin/activate
    if [[ "${1:-}" == "--list" ]]; then
        cd src && exec python train_chronicle.py --config "$config" --list
    fi
    [[ -z "${1:-}" ]] || { echo "Unknown argument: $1 (only --list)" >&2; exit 1; }
    counts=$(cd src && python train_chronicle.py --config "$config" --count)
    read -r num_tasks max_parallel <<< "$counts"
    mkdir -p outputs/submitted_configs
    snapshot="$repo/outputs/submitted_configs/train_chronicle-$(date +%Y%m%d-%H%M%S)-$$.yaml"
    cp "$config" "$snapshot"
    echo "Submitting $num_tasks runs of $config (at most $max_parallel at a time), config copy: $snapshot"
    CONFIG="$snapshot" sbatch --array=0-$(( num_tasks - 1 ))%"$max_parallel" \
        --output=strats_chronicle-%A_%a.out "$repo/scripts/train_chronicle.sbatch"
    exit 0
fi

# in a SLURM job: one run (the array task, or the only run of the config)
cd "$SLURM_SUBMIT_DIR"
[[ -f src/main.py ]] || { echo "Submit from the repo root" >&2; exit 1; }
config=${CONFIG:-configs/train_chronicle.yaml}
[[ "$config" = /* ]] || config="$SLURM_SUBMIT_DIR/$config"
source ~/venvs/strats/bin/activate
export USE_TF=0   # transformers must not import TensorFlow
cd src
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    python train_chronicle.py --config "$config" --task "$SLURM_ARRAY_TASK_ID" \
        --array_min "$SLURM_ARRAY_TASK_MIN" --array_max "$SLURM_ARRAY_TASK_MAX"
else
    python train_chronicle.py --config "$config"
fi
