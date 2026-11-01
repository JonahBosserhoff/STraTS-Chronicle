### Run
```

Ausgangspunkt: Du hast nur den Chronicle-Ordner (aus chronicle-preprocess, mit Scalern usw.) und die Global-Format-CSV.

git clone -b benchmarking-chronicle https://github.com/JonahBosserhoff/STraTS-Chronicle.git
cd STraTS-Chronicle

python3.10 -m venv ~/venvs/strats
source ~/venvs/strats/bin/activate
pip install --upgrade pip
pip install "numpy<2" pandas pytz tqdm matplotlib scikit-learn lifelines pyyaml torch transformers==4.35.2

CHRONICLE_DATA_DIR=/pfad/zum/chronicle_ordner GLOBAL_FORMAT=/pfad/zum/global_format.csv sbatch preprocess_chronicle.sbatch

sbatch train_chronicle.sbatch          # ein Lauf (oder train_chronicle_array.sbatch: 5 Seeds parallel)

Hyperparameter (Epochen, Patience, Lernrate, ...) stehen als Argumente in train_chronicle.sbatch, Defaults in src/main.py (parse_args).

Gelesen werden aus dem Chronicle-Ordner nur die drei *_{train,val,test}.csv. Die Scaler sind darin schon angewendet und werden nicht gebraucht.

Die Ergebnisse landen in outputs/chronicle/strats,…|run:1o5/.

Externe Validierung (nur Inference): Pfade in external_validation.yaml eintragen, dann

sbatch external_validation.sbatch
