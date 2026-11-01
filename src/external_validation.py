"""External validation of a STraTS model trained on a Chronicle cohort: inference only.

The external cohort is a raw (unscaled) CSV in Chronicle's long format, like the input of
chronicle-preprocess. It is mapped into the training run's input space in two steps:
1. the Chronicle scalers of the training cohort (as chronicle-preprocess --pretrained-scaler-dir),
2. the STraTS normalization of the training run (inference.pkl, saved by main.py next to
   checkpoint_best.bin): variable vocabulary, per-variable mean/std, static mean/std, max day.
Variables the model was not trained on are dropped. The best checkpoint is then scored as the
test split during training: C-index mean ± std over num_eval_draws random cut days per patient.

    cd src && python external_validation.py --config ../external_validation.yaml

Output in output_dir: log.txt and predictions_external.csv (columns as predictions_test.csv).
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
args = argparse.Namespace(**saved['args'])
args.num_eval_draws, args.seed, args.eval_batch_size = cfg['num_eval_draws'], cfg['seed'], cfg['eval_batch_size']
os.makedirs(cfg['output_dir'], exist_ok=True)
args.logger = Logger(cfg['output_dir'], 'log.txt')
args.logger.write('\n'+str(cfg))
args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# cohort, scaled with the Chronicle scalers of the training cohort
df = pd.read_csv(cfg['cohort'])
args.logger.write('Read '+cfg['cohort']+': '+str(len(df))+' rows, '+str(df.patient_id.nunique())+' patients')
if cfg['scaler_dir'] is not None:
    scalers = joblib.load(os.path.join(cfg['scaler_dir'], 'feature_scalers.pkl'))
    df['feature_value'] = df['feature_value'].astype(float)
    for feat, idx in df.groupby('feature_name').groups.items():
        if feat in scalers:
            df.loc[idx, 'feature_value'] = scalers[feat].transform(df.loc[idx, ['feature_value']].to_numpy()).ravel()
    args.logger.write('Scaled with '+str(len(scalers))+' Chronicle scalers from '+cfg['scaler_dir'])
dataset = ChronicleDataset(args, external=(*to_strats(df), saved['norm']))

# best checkpoint of the training run
model = Strats(args)
model.load_state_dict(torch.load(os.path.join(cfg['model_dir'], 'checkpoint_best.bin'), map_location=args.device))
model.to(args.device)

result = SurvivalEvaluator(args).evaluate(model, dataset, 'external', train_step=None,
                                          save_path=os.path.join(cfg['output_dir'], 'predictions_external.csv'))
args.logger.write('External res: '+str(result))
