#!/bin/bash
# Chronicle cohort -> data/processed/chronicle.pkl (src/preprocess_chronicle.py), CPU only.
# Run once before training (one pickle for all targets). From the repo root:
#   CHRONICLE_DATA_DIR=<chronicle-preprocess output dir> GLOBAL_FORMAT=<schema csv> sbatch scripts/preprocess_chronicle.sbatch
#SBATCH --job-name=strats_prep
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=strats_prep-%j.out

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
[[ -f src/main.py ]] || { echo "Submit from the repo root (sbatch scripts/preprocess_chronicle.sbatch)" >&2; exit 1; }
source ~/venvs/strats/bin/activate
cd src

python preprocess_chronicle.py --data_dir "${CHRONICLE_DATA_DIR:?set CHRONICLE_DATA_DIR}" \
    --global_format "${GLOBAL_FORMAT:?set GLOBAL_FORMAT}"
