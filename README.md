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
Branch `benchmarking-chronicle` trains STraTS on a Chronicle cohort
to predict **overall survival (OS)**, so that both models can be compared on the same patients, inputs,
prediction time points and metric. Only OS is supported; everything else uses the STraTS defaults.

### Run
```
pip install lifelines
chmod +x run_chronicle.sh
CHRONICLE_DATA_DIR=<chronicle-preprocess output dir> GLOBAL_FORMAT=<schema csv> ./run_chronicle.sh
```
Output in `outputs/chronicle/strats,.../`: `log.txt`, `checkpoint_best.bin` and `predictions_test.csv`.

### What changed
| File | Change |
|---|---|
| `src/preprocess_chronicle.py` | new: Chronicle split CSVs -> `data/processed/chronicle.pkl` |
| `src/dataset_chronicle.py` | new: dataset with random cuts of each patient's history, OS labels |
| `src/evaluator_survival.py` | new: C-index evaluation, saves test predictions |
| `src/models.py` | Cox loss; with `--dataset chronicle` the output head is an OS risk score |
| `src/main.py` | selects the above for `--dataset chronicle`, selects the checkpoint by validation C-index, saves the test predictions of the best checkpoint, falls back to CPU without CUDA |
| `src/modeling_strats.py` | variable dropout vectorized (same result as the loop over all variables, which is slow with Chronicle's ~800 variables) |
| `run_chronicle.sh`, `.gitignore` | run script; `data/` and `outputs/` are not tracked |

The original datasets and models (PhysioNet, MIMIC-III, GRU-D, ...) are unchanged.

### How it works
**Data and splits.** `preprocess_chronicle.py` reads the output of `chronicle-preprocess`
(`<prefix>_{train,val,test}.csv`), i.e. the patient split and the feature scaling of the Chronicle run.
Train / val / test are therefore the same patients. As in Chronicle, only features listed in the schema
are kept. Each observation becomes an STraTS triplet (day since initial diagnosis, feature, value).

**Variables.** STraTS builds its vocabulary from the variables in the training set, so Chronicle's
features need no mapping. Values are z-scored per variable with training statistics, as for the other
datasets (on top of Chronicle's scaling); a variable seen only once in training gets std 1.
`age_at_diagnosis` and `gender_female` are the static variables (the `demo` vector, like Age and
Gender for MIMIC-III; missing values are filled with the training mean). All other features, including
those at day 0, are time-series triplets. Patients without any time-series observation are dropped and
the number is logged.

**Prediction time points.** STraTS attends in both directions, so a sequence must never contain
observations after the time point it predicts for. Each sample is a patient's history cut at one of
their observed days: all observations up to and including that day, and the OS label at that day
(remaining time `day + OS_TIME - cut day`, event). This is the time point of a `<CLS>` token in
Chronicle. If more than `--max_obs` (880) observations remain, the most recent ones are kept.
- Training: a new random cut day each time a patient is drawn, i.e. once per epoch.
- Evaluation (val, test, eval_train): `--num_eval_draws` (10) random cut days per patient, drawn once
  with `--seed`, so all evaluations and runs score the same cuts.

**Loss and metric.** The head outputs a risk score (higher = shorter survival), trained with the
negative Cox partial log-likelihood (Breslow), as `chronicle.losses.cox_loss`; the risk set is the
batch. The metric is Harrell's C-index (lifelines), as `chronicle.metrics.compute_c_index`, reported
as mean ± std over the 10 draws, which is how Chronicle reports it (`c_index_samples: 10`). The
checkpoint with the best validation C-index is kept (patience 10).

**Predictions.** `predictions_test.csv` holds the best checkpoint's test predictions, one row per
patient and draw: `patient_id, draw, day, prediction, time, event` (`day` = cut day, `time` =
remaining OS time in days). Columns as in Chronicle's `predictions_test/OS.parquet`, plus `draw`.

### Differences to Chronicle
- Training batch size 16 (STraTS default) is also the Cox risk set; Chronicle uses 128.
- At most 880 observations per sample; Chronicle evaluates up to 30,000.
- One prediction per forward pass (one cut); Chronicle predicts every day in one pass.
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


