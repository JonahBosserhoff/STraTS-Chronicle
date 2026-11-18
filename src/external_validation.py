"""External validation of a STraTS model trained on a Chronicle cohort: inference only.

Only the test split of the external cohort is used: the one <prefix>_test.csv in data_dir, the
output of chronicle-preprocess for the external cohort, already scaled with the Chronicle scalers
of the training cohort (chronicle-preprocess --pretrained-scaler-dir). It is mapped into the
training run's input space with the STraTS normalization of the training run (inference.pkl, saved
by main.py next to checkpoint_best.bin): variable vocabulary, per-variable mean/std, static
mean/std, max day. Variables the model was not trained on are dropped. The target is the one of
the training run (OS for runs from before --target). The best checkpoint is then scored
as the test split at the end of training: metric mean ± std over num_eval_draws random cut days
per patient, and for a regression / binary target also over every day with a target (the
predictions of every day are saved for every target).

    cd src && python external_validation.py --config ../configs/external_validation.yaml

Output in output_dir: log.txt, predictions_external.csv (columns as predictions_test.csv) and
predictions_external_all_days.csv (every day with a target). Regression predictions
and labels are saved in the original units, as Chronicle does: mapped back with the label scaler
of the training run in inference.pkl (none for older runs).
"""
import argparse
import os
import pickle
from pathlib import Path

import pandas as pd
import torch
import yaml

from dataset_chronicle import ChronicleDataset
from evaluator_survival import SurvivalEvaluator
from modeling_strats import Strats
from preprocess_chronicle import to_strats
from utils import Logger


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='../configs/external_validation.yaml')
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

# test split of the external cohort: the one *_test.csv in data_dir (chronicle-preprocess output),
# already scaled with the Chronicle scalers of the training cohort (--pretrained-scaler-dir)
paths = sorted(Path(cfg['data_dir']).glob('*_test.csv'))
assert len(paths)==1, 'Expected one *_test.csv in '+cfg['data_dir']+', found '+str(paths)
df = pd.read_csv(paths[0])
args.logger.write('Read '+str(paths[0])+': '+str(len(df))+' rows, '+str(df.patient_id.nunique())+' patients')
# regression: the Chronicle label scaler of the training run (saved by main.py; None for runs from
# before) maps the saved predictions and labels back to the original units, as Chronicle
label_scaler = saved.get('label_scaler')
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
# every day with a target, for every target (as main.py); the metric over all days for regression / binary only
result = evaluator.evaluate(model, dataset, 'external', train_step=None, all_days=True,
                            save_path=os.path.join(cfg['output_dir'], 'predictions_external_all_days.csv'))
if args.target_type!='survival':
    args.logger.write('External res, all days with a target: '+str(result))
