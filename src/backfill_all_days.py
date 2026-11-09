"""One-off: writes predictions_test_all_days.csv for survival runs (OS, TTNT, ...) trained before main.py saved it for
every target. Regression / binary runs already have it and are left alone, as is every run that already has it.

    cd src && python backfill_all_days.py --results_dir ../outputs/chronicle

For every finished survival run in results_dir: the test split is built as at the end of training (same patients,
cut days and normalization, checked against inference.pkl), the best checkpoint predicts every test day with a
target, exactly as main.py does now. Inference only; the run folder gets the new CSV and a few lines in log.txt.

Check before writing: the new predictions of the days in predictions_test.csv (the drawn days the training scored)
must be those of predictions_test.csv (up to float noise); otherwise nothing is written for that run.
"""
import argparse
import os
import pickle
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ablation import same_norm, unusable
from dataset_chronicle import ChronicleDataset, STATIC_VARIS
from evaluator_survival import SurvivalEvaluator
from modeling_strats import Strats
from overall_evaluation import ALL_DAYS_NAME, DRAWS_NAME, find_runs
from utils import Logger

TOLERANCE = 1e-4 # max abs difference to predictions_test.csv (risk scores; float noise of other batch sizes)


def backfill(run_dir, cache, device):
    with open(run_dir/'inference.pkl', 'rb') as f:
        saved = pickle.load(f)
    args = argparse.Namespace(**saved['args'])
    saved_type, saved_V, saved_D = args.target_type, args.V, args.D
    args.logger = Logger(str(run_dir), 'log.txt') # appends to the run's log
    args.logger.write('\nbackfill_all_days.py: predictions of every test day with a target ('+ALL_DAYS_NAME+')')

    key = (args.dataset, args.target, args.train_frac, args.seed, args.num_eval_draws, args.max_obs)
    if cache.get('key')!=key:
        cache.clear()
        cache['dataset'], cache['key'] = ChronicleDataset(args), key
    dataset = cache['dataset']
    assert dataset.args.target_type==saved_type=='survival'
    assert same_norm(dataset.norm, saved['norm']), \
        'The normalization of data/processed/chronicle.pkl differs from the one of this run (inference.pkl): ' \
        'the data were preprocessed again since the training'
    assert (len(dataset.norm['variables']), len(STATIC_VARIS))==(saved_V, saved_D)
    args.target_type, args.V, args.D, args.device = saved_type, saved_V, saved_D, device

    model = Strats(args)
    model.load_state_dict(torch.load(run_dir/'checkpoint_best.bin', map_location=device))
    model.to(device)
    tmp = run_dir/(ALL_DAYS_NAME+'.tmp')
    SurvivalEvaluator(args).evaluate(model, dataset, 'test', train_step=None, all_days=True, save_path=str(tmp))

    # check against the predictions the training saved for the drawn days
    new = pd.read_csv(tmp)
    old = pd.read_csv(run_dir/DRAWS_NAME).drop_duplicates(['patient_id', 'day'])
    both = old.merge(new, on=['patient_id', 'day'], how='left', suffixes=('_old', '_new'))
    assert both['prediction_new'].notna().all(), 'Drawn days of '+DRAWS_NAME+' missing in the new predictions'
    for col in ['time', 'event']:
        assert np.array_equal(both[col+'_old'].to_numpy(), both[col+'_new'].to_numpy()), col+' differs from '+DRAWS_NAME
    diff = float(np.abs(both['prediction_old']-both['prediction_new']).max())
    if not diff<=TOLERANCE:
        tmp.unlink()
        raise AssertionError('Predictions differ from '+DRAWS_NAME+' by up to '+str(diff)+': nothing written')
    os.replace(tmp, run_dir/ALL_DAYS_NAME)
    args.logger.write('Saved '+str(run_dir/ALL_DAYS_NAME)+': '+str(len(new))+' rows; the '+str(len(both))
                      +' drawn days of '+DRAWS_NAME+' match (max abs difference '+str(diff)+')')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default='../outputs/chronicle')
    results_dir = Path(parser.parse_args().results_dir)
    if not results_dir.is_dir():
        raise SystemExit(str(results_dir)+' does not exist')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device, flush=True)

    done, present, other, skipped, failed = [], [], [], [], []
    cache = {}
    for target, run, run_dir in find_runs(results_dir):
        name = run_dir.name
        reason = unusable(run_dir)
        if reason is not None:
            print('Skipping', name+':', reason, flush=True)
            skipped.append(name)
            continue
        with open(run_dir/'inference.pkl', 'rb') as f:
            target_type = pickle.load(f)['args'].get('target_type', 'survival')
        if target_type!='survival':
            other.append(name) # regression / binary: main.py saved every day
            continue
        if (run_dir/ALL_DAYS_NAME).is_file():
            present.append(name)
            continue
        print('\n'+name, flush=True)
        try:
            backfill(run_dir, cache, device)
            done.append(name)
        except Exception:
            print('ERROR:', name, '\n'+traceback.format_exc(), flush=True)
            failed.append(name)

    print('\nWritten ('+str(len(done))+'):', done)
    print('Survival runs that already had '+ALL_DAYS_NAME+' ('+str(len(present))+'):', present)
    print('Regression / binary runs, not affected ('+str(len(other))+')')
    print('Skipped, not finished or files missing ('+str(len(skipped))+'):', skipped)
    print('Failed ('+str(len(failed))+'):', failed)
    if failed:
        sys.exit(1)


if __name__=='__main__':
    main()
