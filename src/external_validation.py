"""External validation of a STraTS model trained on a Chronicle cohort: inference only.

The external cohort is a raw (unscaled) CSV in Chronicle's long format, like the input of
chronicle-preprocess. It is mapped into the training run's input space in two steps:
1. the Chronicle scalers of the training cohort (as chronicle-preprocess --pretrained-scaler-dir),
2. the STraTS normalization of the training run (inference.pkl, saved by main.py next to
   checkpoint_best.bin): variable vocabulary, per-variable mean/std, static mean/std, max day.
Variables the model was not trained on are dropped. The target is the one of the training run
(OS for runs from before --target); a regression target is scaled with the Chronicle label
scaler of the training cohort, as chronicle-preprocess does. The best checkpoint is then scored
as the test split at the end of training: metric mean ± std over num_eval_draws random cut days
per patient, and for a regression / binary target also over every day with a target.

    cd src && python external_validation.py --config ../external_validation.yaml

Output in output_dir: log.txt and predictions_external.csv (columns as predictions_test.csv),
for a regression / binary target also predictions_external_all_days.csv. Regression predictions
and labels are saved in the original units, as Chronicle does: mapped back with the label scaler
from scaler_dir, or with scaler_dir null with the one in inference.pkl (none for older runs).
"""
import argparse
import os
import pickle

import joblib
import pandas as pd
import torch
import yaml

from dataset_chronicle import ChronicleDataset
from evaluator_survival import SurvivalEvaluator
from modeling_strats import Strats
from preprocess_chronicle import to_strats
from utils import Logger


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='../external_validation.yaml')
cfg = yaml.safe_load(open(parser.parse_args().config))

# hyperparameters and training statistics of the training run
saved = pickle.load(open(os.path.join(cfg['model_dir'], 'inference.pkl'), 'rb'))
saved['args'].setdefault('target', 'OS') # runs from before --target predict OS
saved['args'].setdefault('target_type', 'survival')
args = argparse.Namespace(**saved['args'])
args.num_eval_draws, args.seed, args.eval_batch_size = cfg['num_eval_draws'], cfg['seed'], cfg['eval_batch_size']
os.makedirs(cfg['output_dir'], exist_ok=True)
args.logger = Logger(cfg['output_dir'], 'log.txt')
args.logger.write('\n'+str(cfg))
args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# cohort, scaled with the Chronicle scalers of the training cohort
df = pd.read_csv(cfg['cohort'])
args.logger.write('Read '+cfg['cohort']+': '+str(len(df))+' rows, '+str(df.patient_id.nunique())+' patients')
# regression: Chronicle label scaler for saving the predictions in the original units, as Chronicle.
# The one of the training run (saved by main.py; None for runs from before) unless scaler_dir is set.
label_scaler = saved.get('label_scaler')
if cfg['scaler_dir'] is not None:
    scalers = joblib.load(os.path.join(cfg['scaler_dir'], 'feature_scalers.pkl'))
    df['feature_value'] = df['feature_value'].astype(float)
    for feat, idx in df.groupby('feature_name').groups.items():
        if feat in scalers:
            df.loc[idx, 'feature_value'] = scalers[feat].transform(df.loc[idx, ['feature_value']].to_numpy()).ravel()
    args.logger.write('Scaled with '+str(len(scalers))+' Chronicle scalers from '+cfg['scaler_dir'])
    if args.target_type=='regression':
        # as chronicle-preprocess: a label is scaled if the training cohort has a scaler for it
        label_scalers = joblib.load(os.path.join(cfg['scaler_dir'], 'label_scalers.pkl'))
        # the same scaler maps the saved predictions back to the original units
        label_scaler = label_scalers.get(args.target)
        if label_scaler is not None and args.target in df.columns:
            values = df[args.target].astype(float)
            usable = values.index[values.notna()]
            df[args.target] = values
            df.loc[usable, args.target] = label_scaler.transform(
                values.loc[usable].to_numpy().reshape(-1, 1)).ravel()
            args.logger.write('Scaled target '+args.target+' with its Chronicle label scaler')
        else:
            args.logger.write('WARNING: target '+args.target+' not scaled (no label scaler in '
                              +cfg['scaler_dir']+', or not in the cohort)')
if args.target_type=='regression':
    args.logger.write('Predictions are saved in the original units (Chronicle label scaler of '+args.target+')'
                      if label_scaler is not None else 'WARNING: no Chronicle label scaler for '+args.target
                      +': predictions are saved in the units of the model output')
dataset = ChronicleDataset(args, external=(*to_strats(df), saved['norm']))

# best checkpoint of the training run
model = Strats(args)
model.load_state_dict(torch.load(os.path.join(cfg['model_dir'], 'checkpoint_best.bin'), map_location=args.device))
model.to(args.device)

evaluator = SurvivalEvaluator(args, label_scaler)
result = evaluator.evaluate(model, dataset, 'external', train_step=None,
                            save_path=os.path.join(cfg['output_dir'], 'predictions_external.csv'))
args.logger.write('External res: '+str(result))
if args.target_type!='survival':
    result = evaluator.evaluate(model, dataset, 'external', train_step=None, all_days=True,
                                save_path=os.path.join(cfg['output_dir'], 'predictions_external_all_days.csv'))
    args.logger.write('External res, all days with a target: '+str(result))
