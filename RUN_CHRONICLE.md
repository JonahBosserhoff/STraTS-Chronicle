# STraTS auf Chronicle-Daten – Kurzanleitung

## Aufbau

Alles wird aus dem Repo-Root gestartet. Einstellungen stehen in `configs/`, die SLURM-Skripte in `scripts/`:

| Schritt | Skript | Config |
|---|---|---|
| Preprocessing (einmal) | `scripts/preprocess_chronicle.sbatch` | – (`CHRONICLE_DATA_DIR`, `GLOBAL_FORMAT` im Terminal) |
| Training | `scripts/train_chronicle.sbatch` (mit **bash** starten, nicht sbatch) | `configs/train_chronicle.yaml` |
| Externe Validierung | `scripts/external_validation.sbatch` | `configs/external_validation.yaml` |
| Feature-Ablation | `scripts/ablation.sbatch` | `configs/ablation.yaml` |

Eine andere Config gibt man mit `CONFIG=<pfad relativ zum Repo-Root>` vor dem Befehl an, z. B.
`CONFIG=configs/train_os_only.yaml bash scripts/train_chronicle.sbatch`.

## Target (Label) wählen

Pro Lauf wird **genau ein** Chronicle-Target gelernt (ein Output-Head). Welche Targets trainiert
werden, steht in `configs/train_chronicle.yaml` unter `targets` (`all` = alle Targets der Kohorte,
oder eine Liste). `scripts/train_chronicle.sbatch` startet für jedes Target und jeden Run einen
eigenen Lauf von `src/main.py --target <T>` auf einer eigenen GPU (SLURM-Job-Array).

```yaml
targets: all                                    # alle Targets der Kohorte
targets: [OS, TTNT, label_min_lab_hemoglobin_60_days]
num_runs: 5                                     # 5 Seeds pro Target (run:1o5 … 5o5); 1 = nur ein Lauf (run:1o1)
runs: null                                      # oder z. B. [2] / [2, 5]: nur diese Runs (fehlgeschlagene wiederholen)
max_parallel: 4                                 # höchstens so viele GPUs gleichzeitig
```

```bash
bash scripts/train_chronicle.sbatch --list      # zeigt nur die Läufe der Config, schickt nichts ab
bash scripts/train_chronicle.sbatch             # schickt alle Läufe als Job-Array ab
```

- **Die 5 Runs sind keine 5-fache Kreuzvalidierung:** Alle Runs nutzen dieselben Train/Val/Test-
  Patienten aus chronicle-preprocess und unterscheiden sich nur im Trainings-Seed.
- **Mit `bash`, nicht `sbatch` starten:** Das Skript liest die Config, berechnet daraus die
  Array-Größe (#Targets × #Runs) und schickt sich selbst per `sbatch --array=…` ab. Die Läufe
  verwenden eine Kopie der Config (`outputs/submitted_configs/`), man kann die Config also
  weiterbearbeiten, während die Jobs warten. `sbatch scripts/train_chronicle.sbatch` geht nur bei
  genau einem Lauf (ein Target, ein Run).
- **Welche Targets es gibt:** Das Preprocessing schreibt sie am Ende in `strats_prep-<jobid>.out`
  (`Targets (choose one with main.py --target): [...]`) und nach
  `data/processed/chronicle_targets.txt` (daraus liest `targets: all`). Möglich sind alle Survival-Targets
  (`<T>_TIME`/`<T>_EVENT`-Paar, z. B. `OS`, `TTNT`) und jede `label_*`-Spalte der Kohorte. Ein
  falscher Name bricht den Lauf sofort ab und listet die verfügbaren Targets.
- **Ein Preprocessing für alle Targets.** Nach dem Wechsel des Targets muss nichts neu
  vorverarbeitet werden. Nur ein `chronicle.pkl` aus der Zeit vor `--target` muss einmal neu
  erzeugt werden (sonst Fehlermeldung „old OS-only format“).
- **Der Typ wird automatisch erkannt** (wie in Chronicle), daraus folgen Loss und Metrik:

  | Target | erkannt an | Loss | Metrik (Checkpoint-Auswahl auf val) |
  |---|---|---|---|
  | Survival | Spaltenpaar `<T>_TIME` / `<T>_EVENT` | Cox (Breslow) | C-Index |
  | Binär | `label_*` enthält nur 0/1 | BCE mit Logits | ROC AUC |
  | Regression | jedes andere `label_*` | MSE | Pearson r |

- **Log pro Lauf:** `strats_chronicle-<arrayjobid>_<taskid>.out`; die erste Zeile nennt Target
  und Run, die zweite den genauen `main.py`-Aufruf.
- **Ergebnisse** liegen in
  `outputs/chronicle/strats,…|target:<TARGET>|train_frac:1.0|run:<r>o<n>/`: `log.txt`,
  `checkpoint_best.bin`, `inference.pkl`, `predictions_test.csv` (die 10 gezogenen Tage pro
  Patient) und `predictions_test_all_days.csv` (alle Tage mit Target). Das Endergebnis steht in
  `log.txt` unter `Test res of the best checkpoint` (bei Regression/Binär zusätzlich
  `…, all days with a target`).
- **Overall-Evaluation wie Chronicle** (C-Index für OS/TTNT, AUROC für die 6 Labs, Transfusion
  und febrile Neutropenie, gleiche Schwellen, 50 Ziehungen mit Seed 42 wie Chronicles
  `overall_evaluation.py`):
  ```bash
  cd src && python overall_evaluation.py --cohort <kohorten_ordner>   # liest ../outputs/chronicle
  ```
  `<kohorten_ordner>/<ordnername>.csv` ist die rohe Kohorten-CSV (Geschlecht für AST/ALT/Kreatinin).
  Ergebnis: eine Zeile pro Lauf in `outputs/chronicle/overall_evaluation.csv`. Die Werte weichen
  leicht vom Log ab (dort 10 Ziehungen mit Seed 2023). Läufe von vor dieser Änderung haben für
  Survival keine `predictions_test_all_days.csv`; dann werden die gezogenen Tage aus
  `predictions_test.csv` verwendet (mit Warnung, nicht ganz vergleichbar).
- **Vorhersagedateien wie bei Chronicle:** Spalten `patient_id, draw, day, prediction` plus
  `time, event` (Survival) bzw. `ground_truth` (Regression/Binär). Bei Regression stehen
  `prediction` und `ground_truth` in **Originaleinheiten** (z. B. g/dL). Zurückgerechnet wird mit
  Chronicles `label_scalers.pkl`, die das Preprocessing nach
  `data/processed/chronicle_label_scalers.pkl` kopiert. Die Metriken im Log werden wie bei
  Chronicle auf den skalierten Werten berechnet. Fehlt die Scaler-Datei, steht eine `WARNING` in
  `log.txt` und die Werte bleiben skaliert. Binär: `prediction` = Logit
  (Wahrscheinlichkeit = `1/(1+exp(-prediction))`). Survival: `prediction` = Risk-Score,
  `time` = verbleibende Tage ab `day`.
- **Hyperparameter** (Epochen, Patience, Lernrate, …) stehen in `configs/train_chronicle.yaml`
  unter `hyperparameters` und werden als `--<name> <wert>` an `src/main.py` übergeben (jedes
  Argument von `main.py` geht, `null` = Default aus `parse_args`). `dataset`, `target`, `run` und
  `train_frac` (1.0) setzt das Skript selbst.
- **Externe Validierung:** Das Target kommt automatisch aus dem trainierten Lauf (`model_dir` in
  `configs/external_validation.yaml`). Die externe Kohorte braucht dessen Spalten. Bei einem
  Regressions-Target muss `scaler_dir` auch `label_scalers.pkl` enthalten.

## Feature-Ablation

Wie stark hängt ein trainiertes Modell an seinen Features? `scripts/ablation.sbatch` lädt jedes
trainierte Modell und rechnet Inference auf dem Test-Split, wobei ein Prozentsatz aller Features
maskiert ist (Default 10 %, 20 %, …, 90 %). Kein Training. Einstellungen in `configs/ablation.yaml`:

```yaml
results_dir: ../outputs/chronicle      # Trainingsläufe
targets: all                           # alle Labels der Kohorte, oder eine Liste
runs: null                             # alle gefundenen Runs, oder z. B. ["1o5"]
percentages: [10, 20, 30, 40, 50, 60, 70, 80, 90]
num_mask_draws: 1                      # >1: mehrere unabhängige Feature-Sets pro Prozentsatz
include_static: true                   # age_at_diagnosis / gender_female mit maskieren
cohort: null                           # Kohorten-Ordner: zusätzlich Metriken wie overall_evaluation.py
output_dir: ../outputs/chronicle_ablation/
```

```bash
sbatch scripts/ablation.sbatch                  # alle Modelle in einem Job
sbatch --array=0-3 scripts/ablation.sbatch      # auf 4 GPUs verteilt (gleicher output_dir)
```

- **Welche Modelle:** Für jedes Label jeder fertige Lauf in `results_dir` (mit
  `checkpoint_best.bin`, `inference.pkl` und „Test res of the best checkpoint“ im Log). Ein Label
  ohne Modell wird übersprungen, ebenso ein unfertiger Lauf. Beides steht am Ende im Log.
- **Welche Features:** Der Pool sind alle Features, die das Modell kennt: das Variablen-Vokabular
  (jedes Zeitreihen-Feature des Train-Splits, gleich für alle Labels) plus, mit `include_static`, die
  beiden statischen Features. Davon werden round(p % × Anzahl) zufällig gezogen.
- **Jeder Prozentsatz wird neu gezogen**, unabhängig von den anderen (die 20 % enthalten nicht die
  10 %). Die Ziehung hängt nur von `seed`, Prozentsatz und Mask-Draw ab. Alle Modelle (alle Labels,
  alle Runs) bekommen bei gleichem Prozentsatz also dieselben maskierten Features und sind direkt
  vergleichbar. Welche das sind, steht in `masked_features.csv`.
- **Maskieren heißt:** Ein Zeitreihen-Feature wird aus den Testdaten entfernt, als wäre es nie
  gemessen worden (die letzten 880 Beobachtungen kommen dann aus den übrigen Features). Ein
  statisches Feature wird auf den Train-Mittelwert gesetzt, wie ein fehlender Wert.
- **Gleiche Testdaten wie am Trainingsende:** Patienten, Schnitt-Tage (Seed und `num_eval_draws` des
  Laufs), Normalisierung und Metrik sind identisch, nur die Eingabe ändert sich. Mit `0` in
  `percentages` kommt exakt „Test res of the best checkpoint“ aus dem Trainingslog heraus. Wurden
  die Daten seit dem Training neu vorverarbeitet (andere Normalisierung), bricht die Ablation für
  dieses Modell mit Fehlermeldung ab.
- **Leere Historie:** Sind alle Features bis zum Schnitt-Tag maskiert, bleibt keine Beobachtung
  übrig. STraTS kann darüber nicht attenden. Diese Schnitte werden nur aus den statischen Features
  vorhergesagt (Zeitreihen-Embedding = 0). Ihr Anteil steht in der Spalte `empty_frac`.
- **Ergebnis** in `output_dir`: `ablation_results.csv` (eine Zeile pro Modell, Prozentsatz und
  Mask-Draw: `metric`, `value`, `std` wie im Trainingslog, bei Regression/Binär auch
  `value_all_days`; außerdem `obs_masked_frac` = Anteil entfernter Test-Beobachtungen und
  `empty_frac`). Mit `cohort` zusätzlich `ablation_overall_evaluation.csv` (C-Index / AUROC wie
  `overall_evaluation.py`, mit Chronicles Schwellen). Pro Modell ein Ordner mit `results.csv`,
  `masked_features.csv`, `log.txt` und, mit `save_predictions`, den Vorhersagen pro Prozentsatz
  (`mask<p>_draw<d>/predictions_test*.csv`).
- Die CSVs in `output_dir` werden aus allen Modell-Ordnern in `output_dir` zusammengeführt. Für
  andere Einstellungen deshalb einen neuen `output_dir` nehmen.

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
CHRONICLE_DATA_DIR=/pfad/zum/chronicle_ordner GLOBAL_FORMAT=/pfad/zum/global_format.csv sbatch scripts/preprocess_chronicle.sbatch

# Training: Targets, Runs, Hyperparameter in configs/train_chronicle.yaml, dann (mit bash!)
bash scripts/train_chronicle.sbatch

# Externe Validierung (nur Inference): Pfade in configs/external_validation.yaml eintragen, dann
sbatch scripts/external_validation.sbatch

# Feature-Ablation (nur Inference): configs/ablation.yaml, dann
sbatch scripts/ablation.sbatch
```

Gelesen werden aus dem Chronicle-Ordner die drei `*_{train,val,test}.csv` (Splits, schon
skaliert) und `scalers/label_scalers.pkl` (nur, um Regressions-Vorhersagen in Originaleinheiten
zu speichern). Für die externe Validierung braucht man zusätzlich die Feature-Scaler (`scaler_dir`).

**Schneller als vorher:** Der Test-Split wird nur noch einmal am Ende mit dem besten Checkpoint
ausgewertet statt in jeder Epoche (`eval_during_training: eval_train` in
`configs/train_chronicle.yaml`). Checkpoint und Testergebnis bleiben gleich, pro Epoche spart das
ca. 20 % Zeit. Außerdem rechnet `scripts/train_chronicle.sbatch` mehrere Seeds/Targets gleichzeitig
auf mehreren GPUs.

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
  - Lädt bei Regression Chronicles Label-Scaler des Targets
    (`data/processed/chronicle_label_scalers.pkl`), damit die Vorhersagen in Originaleinheiten
    gespeichert werden.
  - Speichert `inference.pkl` (Argumente des Laufs, Normalisierungsstatistik, Label-Scaler) für
    die externe Validierung.
- Bei Schritt -1 und jeder Validierung werden die Splits aus `--eval_during_training`
  ausgewertet statt fest `eval_train` und `test`.
- Checkpoint-Auswahl mit `--dataset chronicle`: Validierungsmetrik des Targets (`c_index`,
  `correlation` oder `auc`) statt `auprc+auroc`. Patience und Early Stopping sind wie im Original.
- Am Ende (nur Chronicle): Der beste Checkpoint wird geladen, auf test ausgewertet und als
  `predictions_test.csv` gespeichert. Zusätzlich werden die Vorhersagen für alle Tage mit
  Target gespeichert (`predictions_test_all_days.csv`). Bei Regression/Binär steht deren Metrik
  auch im Log, bei Survival nicht (ein C-Index über alle Tage aller Patienten ist nicht der von
  Chronicle). Daraus rechnet `overall_evaluation.py`. Wird test nicht während des Trainings
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
| `src/preprocess_chronicle.py` | Chronicle-CSVs (`*_{train,val,test}.csv`) → `data/processed/chronicle.pkl` = `[data, oc, train_ids, val_ids, test_ids]`. `data`: ein Triplet (`ts_id, day, variable, value`) pro Beobachtung, nur Features aus dem Schema. `oc`: pro Patient und Tag alle Targets, aus der ersten Zeile des Tages. Kopiert außerdem `scalers/label_scalers.pkl` nach `data/processed/chronicle_label_scalers.pkl` und schreibt die Target-Namen nach `data/processed/chronicle_targets.txt`. |
| `src/dataset_chronicle.py` | Dataset: Historie bis zu einem Schnitt-Tag, Label des gewählten Targets an diesem Tag, Normalisierung, Batches |
| `src/evaluator_survival.py` | Metrik je Target-Typ (C-Index / Pearson r / ROC AUC), Mittel ± Std über die Ziehungen (auf skalierten Werten), Speichern der Vorhersagen (Regression in Originaleinheiten, wie Chronicle) |
| `src/overall_evaluation.py` | C-Index / AUROC aller Läufe in `outputs/chronicle` wie Chronicles `overall_evaluation.py` (gleiche Schwellen, 50 Ziehungen, Seed 42) → `overall_evaluation.csv` |
| `src/external_validation.py`, `configs/external_validation.yaml`, `scripts/external_validation.sbatch` | Inference eines trainierten Modells auf einer externen Kohorte |
| `src/ablation.py`, `configs/ablation.yaml`, `scripts/ablation.sbatch` | Feature-Ablation: Inference auf dem Test-Split mit einem Prozentsatz der Features maskiert |
| `src/train_chronicle.py`, `configs/train_chronicle.yaml`, `scripts/train_chronicle.sbatch` | Training: Config → ein `main.py`-Lauf pro Target × Run, als SLURM-Job-Array (eine GPU pro Lauf) |
| `scripts/preprocess_chronicle.sbatch` | SLURM: Preprocessing (CPU) |
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

Hyperparameter in `configs/train_chronicle.yaml` wie die STraTS-(ss-)-Läufe im Original-`run_main.sh`:
`hid_dim 64, num_layers 2, num_heads 16, dropout 0.2, attention_dropout 0.2, lr 5e-4`,
`max_epochs 50, patience 10`, Batch-Größe 16 (= Default von `main.py`).

Weitere Unterschiede zu Chronicle selbst (Multi-Task, Batch-Größe 128, bis 30 000
Beobachtungen, …) stehen in der README unter „Differences to Chronicle“.
