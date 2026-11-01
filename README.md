# STraTS: Self-Supervised Transformer for Sparse and Irregularly Sampled Multivariate Clinical Time-Series
This repo contains an official re-implementation of STraTS in pytorch. <br>
Paper: https://arxiv.org/pdf/2107.14293.pdf

## Important changes
- We included implementations for the following models: [STraTS](https://arxiv.org/pdf/2107.14293.pdf), [GRU-D](https://arxiv.org/pdf/1606.01865.pdf), [InterpNet](https://openreview.net/pdf?id=r1efr3C9Ym), GRU, [TCN](https://arxiv.org/pdf/1803.01271.pdf), [SaND](https://dl.acm.org/doi/pdf/10.5555/3504035.3504536)
- For STraTS, we removed LayerNorm and replaced ReLU activations in the FFN with GELU, as this improved the performance.
- We used mostly similar hyperparameters for both the datasets, and used the same hidden dimension of 64 for all models.
- Taking inspiration from GRU-D, which uses the same dropout mask at each input time-step, resulting in masking out some variables, we also drop a fraction of variables from the input while training STraTS.
- These changes, along with some possible differences arising from Pytorch's inbuilt modules, make some of the results deviate from the original numbers published in the paper.

## Conda env setup
```
conda create -n strats python=3.10.9
source activate strats
pip install pytz pandas tqdm matplotlib 
pip install torch==2.0.0+cu117 torchvision==0.15.1+cu117 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu117
pip install transformers==4.35.2
pip install scikit-learn==1.2.2
```

## Dataset preprocessing
Download PhysioNet2012 dataset from https://physionet.org/content/challenge-2012/1.0.0/. <br>
Download MIMIC-III from https://physionet.org/content/mimiciii/1.4/, <br>
Update "RAW_DATA_PATH" variable in the preprocessing scripts and run them.
```
python preprocess_physionet_2012.py
python preprocess_mimic_iii_large.py
```

## Training and evaluation
The shell script run_main.sh contains the commands for training and evaluating each of the supported models.
```
chmod +x run_main.sh
./run_main.sh
```

## Results from run_main.sh
![image](https://github.com/sindhura97/STraTS/assets/42525474/25514af6-47a3-4a0b-8861-235217181abd|width=100)

## Benchmarking on Chronicle data
Branch `benchmarking-chronicle` trains STraTS on a Chronicle cohort to predict **one Chronicle target
per run** (single head), so that both models can be compared on the same patients, inputs,
prediction time points and metric: overall survival (`OS`, the default), `TTNT`, or any `label_*`
target of the cohort (lab values in 60 days, transfusion, febrile neutropenia, ...). Everything else
uses the STraTS defaults.

### Run
From the repo root (SLURM):
```
CHRONICLE_DATA_DIR=<chronicle-preprocess output dir> GLOBAL_FORMAT=<schema csv> sbatch preprocess_chronicle.sbatch

sbatch train_chronicle.sbatch                                          # one run, OS
TARGET=label_min_lab_hemoglobin_60_days sbatch train_chronicle.sbatch  # one run, another target

./submit_chronicle.sh OS TTNT label_min_lab_hemoglobin_60_days         # in parallel: 5 runs per target
NUM_RUNS=1 MAX_PARALLEL=8 ./submit_chronicle.sh OS TTNT                # 1 run per target, up to 8 GPUs
```
`preprocess_chronicle.py` prints the targets of the cohort. One preprocessing serves all targets
(it has to be re-run once for pickles written before `--target` existed).
Output in `outputs/chronicle/strats,...|target:<target>|train_frac:1.0|run:<r>o<n>/`: `log.txt`,
`checkpoint_best.bin`, `inference.pkl`, `predictions_test.csv` and, for regression / binary targets,
`predictions_test_all_days.csv`.

### Faster training
- **Parallel runs.** `submit_chronicle.sh` submits `train_chronicle_array.sbatch` as a SLURM job
  array: every (target, run) is an independent job on its own GPU, at most `MAX_PARALLEL` at a time,
  so 5 seeds take as long as one. Task `i` trains target `i / NUM_RUNS` with `--run (i % NUM_RUNS + 1)o<NUM_RUNS>`;
  runs differ only in the training seed, as `--run` in `run_main.sh`.
- **No test evaluation every epoch.** Both sbatch scripts pass `--eval_during_training eval_train`:
  during training only val (which selects the checkpoint) and eval_train are evaluated, and test once
  with the best checkpoint at the end. Evaluation draws no random numbers, so the training, the
  checkpoint and the test result are the same as with test evaluated at every validation (checked
  bit for bit on CPU; on a GPU, STraTS runs are not bitwise reproducible anyway, with or without it);
  it saves the test evaluation of every epoch (about a fifth of the time per epoch on the Essen
  cohort). The default (`eval_train,test`) keeps the original behaviour.
- A single run is not made faster otherwise: its time goes into STraTS' attention (16 heads over up
  to 880 observations per sample), and changing the batch size or the model would change the benchmark.

### What changed
| File | Change |
|---|---|
| `src/preprocess_chronicle.py` | new: Chronicle split CSVs -> `data/processed/chronicle.pkl`, with every target per patient and day; copies Chronicle's label scalers |
| `src/dataset_chronicle.py` | new: dataset with random cuts of each patient's history, labels of the chosen target (`--target`) |
| `src/evaluator_survival.py` | new: C-index / Pearson r / AUC evaluation, saves test predictions (regression in the original units, as Chronicle) |
| `src/models.py` | with `--dataset chronicle` the output head predicts the chosen target: Cox loss (survival), MSE (regression), BCE (binary) |
| `src/main.py` | selects the above for `--dataset chronicle` (`--target`), selects the checkpoint by the validation metric, evaluates and saves the test predictions of the best checkpoint, `--eval_during_training`, falls back to CPU without CUDA |
| `src/modeling_strats.py` | variable dropout vectorized (same result as the loop over all variables, which is slow with Chronicle's ~800 variables) |
| `src/external_validation.py`, `external_validation.yaml` | new: inference of a trained model on an external cohort |
| `preprocess_chronicle.sbatch`, `train_chronicle.sbatch`, `train_chronicle_array.sbatch`, `submit_chronicle.sh`, `external_validation.sbatch` | SLURM scripts |
| `.gitignore` | `data/` and `outputs/` are not tracked |

The original datasets and models (PhysioNet, MIMIC-III, GRU-D, ...) are unchanged.

### How it works
**Data and splits.** `preprocess_chronicle.py` reads the output of `chronicle-preprocess`
(`<prefix>_{train,val,test}.csv`), i.e. the patient split and the feature (and label) scaling of the
Chronicle run. Train / val / test are therefore the same patients. As in Chronicle, only features
listed in the schema are kept. Each observation becomes an STraTS triplet (day since initial
diagnosis, feature, value). The targets are stored once per patient and day, from the day's first row,
as Chronicle's `<CLS>` token of a day copies the labels of the day's first observation.

**Targets.** Found as in Chronicle: a `<T>_TIME` / `<T>_EVENT` pair is a survival target (`OS`,
`TTNT`); a `label_*` column holding only 0 and 1 is binary (how `chronicle-preprocess` decides which
labels to leave unscaled); any other `label_*` column is regression, in the scaled units of
`chronicle-preprocess` (Chronicle trains and scores on these too). One target per run, one output head.

**Variables.** STraTS builds its vocabulary from the variables in the training set, so Chronicle's
features need no mapping. Values are z-scored per variable with training statistics, as for the other
datasets (on top of Chronicle's scaling); a variable seen only once in training gets std 1. These
statistics come from all training patients with time-series data, whatever the target.
`age_at_diagnosis` and `gender_female` are the static variables (the `demo` vector, like Age and
Gender for MIMIC-III; missing values are filled with the training mean). All other features, including
those at day 0, are time-series triplets. Patients without any time-series observation are dropped and
the number is logged.

**Prediction time points.** STraTS attends in both directions, so a sequence must never contain
observations after the time point it predicts for. Each sample is a patient's history cut at one of
their observed days: all observations up to and including that day, and the target at that day
(survival: remaining time `<T>_TIME` and event at that day). This is the time point of a `<CLS>` token
in Chronicle. Only days with a target are cut days, as Chronicle scores only `<CLS>` tokens with a
target (NaN, e.g. no measurement in the label's window, or TTNT before the first therapy); patients
without such a day are dropped for that target and the number is logged. If more than `--max_obs`
(880) observations remain, the most recent ones are kept.
- Training: a new random cut day each time a patient is drawn, i.e. once per epoch.
- Evaluation (val, test, eval_train): `--num_eval_draws` (10) random cut days per patient, drawn once
  with `--seed`, so all evaluations and runs score the same cuts.
- Regression / binary targets: the best checkpoint is also scored on every cut day of every test
  patient, which is how Chronicle scores these targets (every `<CLS>` token with a target).

**Loss and metric**, as in Chronicle (`chronicle.losses`, `chronicle.metrics`):
| Target | Loss | Metric |
|---|---|---|
| survival (`OS`, `TTNT`) | negative Cox partial log-likelihood (Breslow), risk set = the batch | Harrell's C-index (lifelines) |
| regression (`label_*`) | MSE | Pearson r |
| binary (`label_*`, 0/1) | BCE with logits | ROC AUC |

Metrics are reported as mean ± std over the 10 draws, which is how Chronicle reports the C-index
(`c_index_samples: 10`); a draw whose metric is undefined (one class only) is left out with a warning.
The checkpoint with the best validation metric is kept (patience 10).

**Predictions.** `predictions_test.csv` holds the best checkpoint's test predictions, one row per
patient and draw: `patient_id, draw, day, prediction` plus `time, event` (survival; `time` = remaining
time in days) or `ground_truth` (regression / binary). `day` is the cut day. `prediction` is the risk
score (survival), the predicted value (regression) or the logit (binary).
`predictions_test_all_days.csv` (regression / binary) has the same columns with one row per patient
and day (`draw` 0). Columns as in Chronicle's `predictions_test/<target>.parquet`, plus `draw`.
As in Chronicle, regression `prediction` and `ground_truth` are mapped back to the original units
(e.g. g/dL) with Chronicle's label scaler of the target, which `preprocess_chronicle.py` copies from
`<chronicle dir>/scalers/label_scalers.pkl` to `data/processed/chronicle_label_scalers.pkl`; the
metrics are computed on the scaled values, as in Chronicle. Without that file (or for a label that
`chronicle-preprocess` did not scale) the values are saved as they are, and `log.txt` says so.

### External validation
Scores a trained model on an external cohort, without training:
```
sbatch external_validation.sbatch      # or: cd src && python external_validation.py --config ../external_validation.yaml
```
`external_validation.yaml` sets the external cohort (raw, unscaled CSV in Chronicle's long format, like
the input of `chronicle-preprocess`, with the target's columns), the Chronicle scalers of the training
cohort (`scaler_dir`, or `null` if the cohort is already scaled with them), the training run
(`model_dir`) and `output_dir`. The target is the one of the training run (runs from before `--target`
predict OS). The cohort goes through the same steps as the training data: Chronicle scalers (for a
regression target also its label scaler), then the STraTS normalization of the training run from
`inference.pkl` (variable vocabulary, per-variable mean/std, static mean/std), which `main.py` saves
next to `checkpoint_best.bin`. Variables the model was not trained on are dropped; a missing static
variable is filled with the training mean. The metric is computed over `num_eval_draws` random cut days
per patient, as for the test split, and for regression / binary targets also over every day with a
target. Output: `log.txt`, `predictions_external.csv` (columns as `predictions_test.csv`) and, for
regression / binary targets, `predictions_external_all_days.csv`. Regression predictions are saved in
the original units, with the label scaler from `scaler_dir` (or, with `scaler_dir: null`, the one
saved in `inference.pkl`; runs from before this have none and stay in the scaled units).

### Differences to Chronicle
- One target per model (single head); Chronicle trains all targets jointly (multi-task).
- Training batch size 16 (STraTS default) is also the Cox risk set; Chronicle uses 128.
- At most 880 observations per sample; Chronicle evaluates up to 30,000.
- One prediction per forward pass (one cut); Chronicle predicts every day in one pass. In training,
  regression / binary targets therefore see one random day per patient and epoch, Chronicle every day.
- No self-supervised pretraining (STraTS (ss-) setting); `--pretrain` and finetuning are not supported
  with `--dataset chronicle`, nor are models other than `strats` / `istrats`.
- `--train_frac` must be 1.0 (the full Chronicle splits); `--run` only changes the training seed.
- Patient ids are written as in the Chronicle CSVs; Chronicle converts them to integers.

## Citation
If you found this work useful, please cite our paper:
```
@article{tipirneni2022self,
  title={Self-supervised transformer for sparse and irregularly sampled multivariate clinical time-series},
  author={Tipirneni, Sindhu and Reddy, Chandan K},
  journal={ACM Transactions on Knowledge Discovery from Data (TKDD)},
  volume={16},
  number={6},
  pages={1--17},
  year={2022},
  publisher={ACM New York, NY}
}
```


