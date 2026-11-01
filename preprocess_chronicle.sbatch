#!/bin/bash
# Chronicle cohort -> data/processed/chronicle.pkl (src/preprocess_chronicle.py), CPU only.
# Run once before train_chronicle(_array).sbatch. Submit from the repo root:
#   CHRONICLE_DATA_DIR=<chronicle-preprocess output dir> GLOBAL_FORMAT=<schema csv> sbatch preprocess_chronicle.sbatch
#SBATCH --job-name=strats_prep
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=strats_prep-%j.out

source ~/venvs/strats/bin/activate
cd "$SLURM_SUBMIT_DIR/src"

python preprocess_chronicle.py --data_dir ${CHRONICLE_DATA_DIR:?set CHRONICLE_DATA_DIR} \
    --global_format ${GLOBAL_FORMAT:?set GLOBAL_FORMAT}
