# STraTS auf Chronicle-Daten – Kurzanleitung

## Target (Label) wählen

Pro Lauf wird **genau ein** Chronicle-Target gelernt (ein Output-Head). Welches, bestimmt die
Umgebungsvariable `TARGET` (bzw. `--target` in `src/main.py`). Default ist `OS`.

```bash
sbatch train_chronicle.sbatch                                            # OS
TARGET=TTNT sbatch train_chronicle.sbatch                                # TTNT
TARGET=label_min_lab_hemoglobin_60_days sbatch train_chronicle.sbatch    # beliebiges label_*
TARGET=label_supportive_care_transfusion_7_days RUN=2o5 sbatch train_chronicle.sbatch  # anderer Seed
```

- **Welche Targets es gibt:** Das Preprocessing schreibt sie am Ende in `strats_prep-<jobid>.out`
  (`Targets (choose one with main.py --target): [...]`). Möglich sind alle Survival-Targets
  (`<T>_TIME`/`<T>_EVENT`-Paar, z. B. `OS`, `TTNT`) und jede `label_*`-Spalte der Kohorte. Ein
  falscher Name bricht sofort ab und listet die verfügbaren Targets.
- **Ein Preprocessing für alle Targets.** Nach dem Wechsel des Targets muss nichts neu
  vorverarbeitet werden. Nur ein `chronicle.pkl` aus der Zeit vor `--target` muss einmal neu
  erzeugt werden (sonst Fehlermeldung „old OS-only format“).
- **Der Typ wird automatisch erkannt** (wie in Chronicle), daraus folgen Loss und Metrik:

  | Target | erkannt an | Loss | Metrik (Checkpoint-Auswahl auf val) |
  |---|---|---|---|
  | Survival | Spaltenpaar `<T>_TIME` / `<T>_EVENT` | Cox (Breslow) | C-Index |
  | Binär | `label_*` enthält nur 0/1 | BCE mit Logits | ROC AUC |
  | Regression | jedes andere `label_*` | MSE | Pearson r |

- **Mehrere Targets / Seeds parallel** (SLURM-Job-Array, je eine GPU pro Lauf):
  ```bash
  ./submit_chronicle.sh                                            # OS, 5 Seeds (1o5 … 5o5)
  ./submit_chronicle.sh OS TTNT label_min_lab_hemoglobin_60_days   # 3 Targets x 5 Seeds = 15 Läufe
  NUM_RUNS=1 MAX_PARALLEL=8 ./submit_chronicle.sh OS TTNT          # 1 Seed pro Target, bis 8 GPUs gleichzeitig
  ```
  `NUM_RUNS` = Seeds pro Target (Default 5), `MAX_PARALLEL` = höchstens so viele Läufe
  gleichzeitig (Default 4). Log pro Lauf: `strats_chronicle-<arrayjobid>_<taskid>.out`; die
  erste Zeile nennt Target und Run.
- **Ergebnisse** liegen in
  `outputs/chronicle/strats,…|target:<TARGET>|train_frac:1.0|run:<r>o<n>/`: `log.txt`,
  `checkpoint_best.bin`, `inference.pkl`, `predictions_test.csv` und bei Regression/Binär
  zusätzlich `predictions_test_all_days.csv`. Das Endergebnis steht in `log.txt` unter
  `Test res of the best checkpoint` (bei Regression/Binär zusätzlich `…, all days with a target`).
- **Hyperparameter** (Epochen, Patience, Lernrate, …) stehen als Argumente in
  `train_chronicle.sbatch` **und** `train_chronicle_array.sbatch`. Beide sollten gleich bleiben.
  Die Defaults stehen in `src/main.py` (`parse_args`).
- **Externe Validierung:** Das Target kommt automatisch aus dem trainierten Lauf (`model_dir` in
  `external_validation.yaml`). Die externe Kohorte braucht dessen Spalten. Bei einem
  Regressions-Target muss `scaler_dir` auch `label_scalers.pkl` enthalten.

## Run

Ausgangspunkt: Du hast nur den Chronicle-Ordner (aus chronicle-preprocess, mit Scalern usw.) und die Global-Format-CSV.

```bash
git clone -b benchmarking-chronicle https://github.com/JonahBosserhoff/STraTS-Chronicle.git
cd STraTS-Chronicle

python3.10 -m venv ~/venvs/strats
source ~/venvs/strats/bin/activate
pip install --upgrade pip
pip install "numpy<2" pandas pytz tqdm matplotlib scikit-learn lifelines pyyaml torch transformers==4.35.2

# einmal, für alle Targets; gibt am Ende die verfügbaren Targets aus
CHRONICLE_DATA_DIR=/pfad/zum/chronicle_ordner GLOBAL_FORMAT=/pfad/zum/global_format.csv sbatch preprocess_chronicle.sbatch

# Training: ein Lauf …
TARGET=OS sbatch train_chronicle.sbatch
# … oder mehrere parallel
./submit_chronicle.sh OS TTNT

# Externe Validierung (nur Inference): Pfade in external_validation.yaml eintragen, dann
sbatch external_validation.sbatch
```

Gelesen werden aus dem Chronicle-Ordner nur die drei `*_{train,val,test}.csv`. Die Feature-Scaler
sind darin schon angewendet und werden für das Training nicht gebraucht. Für die externe
Validierung braucht man sie (`scaler_dir`).

**Schneller als vorher:** Der Test-Split wird nur noch einmal am Ende mit dem besten Checkpoint
ausgewertet statt in jeder Epoche (`--eval_during_training eval_train` in beiden sbatch-Skripten).
Checkpoint und Testergebnis bleiben gleich, pro Epoche spart das ca. 20 % Zeit. Außerdem rechnet
`submit_chronicle.sh` mehrere Seeds/Targets gleichzeitig auf mehreren GPUs.

## Änderungen gegenüber dem Original-Repo

Original: [sindhura97/STraTS](https://github.com/sindhura97/STraTS), letzter Commit `e936cda`
(„Create LICENSE“, 2024-08-03). Den vollständigen Diff zeigt `git diff e936cda`.

### Unverändert

`src/dataset.py`, `src/dataset_pretrain.py`, `src/evaluator.py`, `src/evaluator_pretrain.py`,
`src/utils.py`, `src/modeling_gru.py`, `src/modeling_grud.py`, `src/modeling_interpnet.py`,
`src/modeling_sand.py`, `src/modeling_tcn.py`, `src/preprocess_physionet_2012.py`,
`src/preprocess_mimic_iii_large.py`, `LICENSE`. Die Original-Datensätze (PhysioNet-2012,
MIMIC-III) und -Modelle laufen wie im Original. Alle Chronicle-Änderungen greifen nur mit
`--dataset chronicle`. Einzige Ausnahmen sind der CPU-Fallback in `main.py` und die
vektorisierte Variable-Dropout-Schleife in `modeling_strats.py` (gleiches Ergebnis, siehe unten).

### Geänderte Original-Dateien

**`src/main.py`**
- Neue Argumente:
  - `--target` (Default `OS`): das zu lernende Chronicle-Target.
  - `--num_eval_draws` (Default 10): zufällige Schnitt-Tage pro Patient für die Evaluation.
  - `--eval_during_training` (Default `eval_train,test` = Originalverhalten): welche Splits
    neben val bei jeder Validierung ausgewertet werden. Ein anderer Wert ist nur mit
    `--dataset chronicle` erlaubt.
- Ausgabeordner: Mit `--dataset chronicle` steht zusätzlich `|target:<TARGET>` vor `|train_frac:`.
- Gerät: `cuda` falls verfügbar, sonst CPU. Das Original nutzt immer `cuda` und bricht ohne GPU ab.
- Mit `--dataset chronicle`:
  - Lädt `ChronicleDataset` und `SurvivalEvaluator`.
  - Erlaubt nur `--model_type strats`/`istrats`, kein `--pretrain` und kein `--load_ckpt_path`.
  - Speichert `inference.pkl` (Argumente des Laufs und Normalisierungsstatistik) für die
    externe Validierung.
- Bei Schritt -1 und jeder Validierung werden die Splits aus `--eval_during_training`
  ausgewertet statt fest `eval_train` und `test`.
- Checkpoint-Auswahl mit `--dataset chronicle`: Validierungsmetrik des Targets (`c_index`,
  `correlation` oder `auc`) statt `auprc+auroc`. Patience und Early Stopping sind wie im Original.
- Am Ende (nur Chronicle): Der beste Checkpoint wird geladen, auf test ausgewertet und als
  `predictions_test.csv` gespeichert. Bei Regression/Binär wird zusätzlich auf allen Tagen mit
  Target ausgewertet (`predictions_test_all_days.csv`). Wird test nicht während des Trainings
  ausgewertet, verweist die Zeile `Final test res` darauf.

**`src/models.py`**
- Neue Funktion `cox_loss`: negative Cox-Partial-Log-Likelihood (Breslow, ohne
  Tie-Korrektur), Risk-Set = Batch, wie `chronicle.losses.cox_loss`.
- `TimeSeriesModel`: Mit `--dataset chronicle` liefert der Output-Head (weiterhin ein
  `nn.Linear(…, 1)`) einen Logit bzw. Wert. Der Loss hängt vom Target-Typ ab: `cox_loss`
  (Survival), `F.mse_loss` (Regression), `F.binary_cross_entropy_with_logits` ohne
  `pos_weight` (Binär). Ohne Labels gibt das Modell den Rohwert zurück (Risk-Score, Wert oder
  Logit). Der Original-Pfad (BCE mit `pos_class_weight`) ist unverändert.

**`src/modeling_strats.py`**
- Variable-Dropout im Training vektorisiert:
  `obs_mask*(1-var_mask.gather(1, varis.long()))` statt einer Schleife über alle `V` Variablen.
  Das maskiert exakt dieselben Beobachtungen, ist aber bei Chronicles ~800 Variablen viel
  schneller.

**`README.md`**
- Der Originaltext ist unverändert. Neu ist der Abschnitt „Benchmarking on Chronicle data“.

**`run_main.sh`**
- Gelöscht (Commit `86c95ac`). Das war das Original-Skript für die PhysioNet-/MIMIC-Experimente.
  Der Original-Teil der README verweist noch darauf.

### Neue Dateien

| Datei | Inhalt |
|---|---|
| `src/preprocess_chronicle.py` | Chronicle-CSVs (`*_{train,val,test}.csv`) → `data/processed/chronicle.pkl` = `[data, oc, train_ids, val_ids, test_ids]`. `data`: ein Triplet (`ts_id, day, variable, value`) pro Beobachtung, nur Features aus dem Schema. `oc`: pro Patient und Tag alle Targets, aus der ersten Zeile des Tages. |
| `src/dataset_chronicle.py` | Dataset: Historie bis zu einem Schnitt-Tag, Label des gewählten Targets an diesem Tag, Normalisierung, Batches |
| `src/evaluator_survival.py` | Metrik je Target-Typ (C-Index / Pearson r / ROC AUC), Mittel ± Std über die Ziehungen, Speichern der Vorhersagen |
| `src/external_validation.py`, `external_validation.yaml`, `external_validation.sbatch` | Inference eines trainierten Modells auf einer externen Kohorte |
| `preprocess_chronicle.sbatch` | SLURM: Preprocessing (CPU) |
| `train_chronicle.sbatch` | SLURM: ein Lauf (`TARGET`, `RUN`), eine GPU |
| `train_chronicle_array.sbatch`, `submit_chronicle.sh` | SLURM-Job-Array: mehrere Targets × Seeds parallel |
| `RUN_CHRONICLE.md` | diese Anleitung |
| `.gitignore` | `data/`, `outputs/`, `.DS_Store`, SLURM-Logs werden nicht versioniert |

Zusätzlich sind 15 Dateien `src/__pycache__/*.pyc` mit eingecheckt (Commit `9e32359`).
Das sind Python-Cache-Dateien ohne Wirkung auf die Ergebnisse.

### Methodische Unterschiede zur Original-STraTS-Pipeline

| | Original (PhysioNet / MIMIC-III) | Chronicle |
|---|---|---|
| Aufgabe | Mortalität, binär, ein festes Beobachtungsfenster (MIMIC-III: erste 24 h, PhysioNet-2012: 48 h) | ein frei wählbares Chronicle-Target: Survival, Regression oder Binär |
| Zeitachse | Minuten seit Aufnahme | Tage seit Erstdiagnose (`time_since_initial_diagnosis`), skaliert auf [-1, 1] mit dem maximalen Tag im Train-Split |
| Vorhersagezeitpunkt | Ende des Fensters, eine Sequenz pro Patient | Schnitt an einem beobachteten Tag mit Target: alle Beobachtungen bis einschließlich dieses Tages, Label an diesem Tag (wie Chronicles `<CLS>`-Token). Training: neuer zufälliger Schnitt-Tag bei jedem Ziehen des Patienten. Evaluation: 10 feste zufällige Schnitt-Tage pro Patient (Seed `--seed`). |
| Mehr als `--max_obs` (880) Beobachtungen | zufällige 880 (einmal beim Laden gezogen) | die letzten 880 bis zum Schnitt-Tag |
| Loss / Metrik | BCE mit `pos_weight` / AUROC, AUPRC, minRP | siehe Tabelle oben; Metrik als Mittel ± Std über die 10 Ziehungen (wie Chronicles `c_index_samples`). Bei Regression/Binär zusätzlich über alle Tage mit Target (wie Chronicle). |
| Checkpoint-Auswahl | `auprc+auroc` auf val | Metrik des Targets auf val |
| Splits | eigener Split; `--train_frac`/`--run` wählen eine Teilmenge | die Patienten-Splits von `chronicle-preprocess`. `--train_frac` muss 1.0 sein, `--run` ändert nur den Trainings-Seed. |
| Statische Variablen (`demo`) | MIMIC-III: Age, Gender; PhysioNet-2012: Age, Gender, Height, ICUType | `age_at_diagnosis`, `gender_female`; fehlende Werte mit dem Train-Mittel gefüllt, dann z-standardisiert |
| Normalisierung | z-Score pro Variable (Train), Std 0 → 1 | zusätzlich zu Chronicles Scaling z-Score pro Variable (Train-Patienten mit Zeitreihendaten, unabhängig vom Target); Std 0 oder undefiniert (Variable nur einmal gesehen) → 1 |
| Patienten | alle | Patienten ohne Zeitreihen-Beobachtung und, je Target, ohne einen Tag mit Target werden entfernt (Anzahl im Log) |
| Pretraining | optional (self-supervised) | nicht unterstützt (STraTS-(ss-)-Setting) |

Hyperparameter in den sbatch-Skripten wie die STraTS-(ss-)-Läufe im Original-`run_main.sh`:
`--hid_dim 64 --num_layers 2 --num_heads 16 --dropout 0.2 --attention_dropout 0.2 --lr 5e-4`,
`--max_epochs 50 --patience 10`, Batch-Größe 16 (Default von `main.py`, nicht im Skript gesetzt).

Weitere Unterschiede zu Chronicle selbst (Multi-Task, Batch-Größe 128, bis 30 000
Beobachtungen, …) stehen in der README unter „Differences to Chronicle“.
