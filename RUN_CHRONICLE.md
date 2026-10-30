### Run
```

Ausgangspunkt: Du hast nur den Chronicle-Ordner (aus chronicle-preprocess, mit Scalern usw.) und die Global-Format-CSV.

git clone -b benchmarking-chronicle https://github.com/JonahBosserhoff/STraTS-Chronicle.git
cd STraTS-Chronicle

python3.10 -m venv ~/venvs/strats
source ~/venvs/strats/bin/activate
pip install --upgrade pip
pip install "numpy<2" pandas pytz tqdm matplotlib scikit-learn lifelines torch transformers==4.35.2

cd src
python preprocess_chronicle.py --data_dir /pfad/zum/chronicle_ordner --global_format /pfad/zum/global_format.csv
cd ..

sbatch train_chronicle.sbatch

Gelesen werden aus dem Chronicle-Ordner nur die drei *_{train,val,test}.csv. Die Scaler sind darin schon angewendet und werden nicht gebraucht.

Die Ergebnisse landen in outputs/chronicle/strats,…|run:1o5/.