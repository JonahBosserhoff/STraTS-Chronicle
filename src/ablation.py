"""Feature ablation of trained STraTS Chronicle models: inference on the test split with a share of all features
masked. Settings in configs/ablation.yaml:

    cd src && python ablation.py --config ../configs/ablation.yaml       (or: sbatch scripts/ablation.sbatch)

Models. For every target (label) of the config ("all" = every target of the cohort) every finished training run
in results_dir (run folders of main.py --dataset chronicle: ...|target:<T>|train_frac:...|run:<r>, with
checkpoint_best.bin, inference.pkl and "Test res of the best checkpoint" in log.txt) is ablated. A target without
such a run is skipped, as is an unfinished run; both are logged and listed at the end.

Test data. The test split is built as at the end of training (main.py): same patients, same cut days (the run's
seed and num_eval_draws), same normalization (checked against inference.pkl), same metric (mean ± std over the
draws of cut days and, for regression / binary targets, over every day with a target; the predictions of every
day are saved for every target). Only the input changes, so
0 % reproduces "Test res of the best checkpoint" of the run.

Masking. For every percentage p of the config and every mask draw, round(p % x #features) features are drawn at
random from all features the model knows: its variable vocabulary (every time-series feature of the training set)
and, with include_static, the static features age_at_diagnosis and gender_female.
- Every percentage is drawn anew, independently of the other percentages (the 20 % features do not contain the
  10 % ones). A draw depends only on seed, p and the mask draw (and the feature vocabulary), so every model (every
  target and run) gets the same masked features at the same p and mask draw. num_mask_draws > 1 draws several
  independent feature sets per p.
- A masked time-series feature is removed from the test data: none of its observations is seen, as if it had
  never been recorded (the max_obs most recent observations are taken from the remaining ones).
- A masked static feature is set to the training mean, as a missing static feature is.
- A cut whose history has no observation left (every feature observed up to the cut day masked) is predicted from
  the static features alone, with the time-series embedding set to 0: STraTS cannot attend over an empty
  sequence. Their share is reported (empty_frac).

Output in output_dir:
    ablation_results.csv             one row per model, p and mask draw: metric, std, share of masked observations,
                                     share of cuts without observations, ...
    ablation_overall_evaluation.csv  with cohort: the metrics of overall_evaluation.py (C-index for OS / TTNT,
                                     AUROC with Chronicle's thresholds for the labs, ...) of the same predictions
    log.txt                          (log_task<i>.txt for SLURM array task i)
    <run folder>/                    per model: results.csv, overall_evaluation.csv, masked_features.csv
                                     (p, mask_draw, feature), log.txt and, with save_predictions, one folder
                                     mask<p>_draw<d>/ with predictions_test.csv and predictions_test_all_days.csv
                                     as in the training run folder
The two CSVs in output_dir are merged from the per-model files of every model in output_dir: use a new output_dir
for a new ablation setting.

As a SLURM array (sbatch --array=0-3 scripts/ablation.sbatch) the models are split into consecutive blocks, one per
array task; every task writes its own models and merges the CSVs at its end (the last one has all models).
"""
import argparse
import os
import pickle
import sys
import traceback
from fractions import Fraction
from math import floor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from dataset_chronicle import ChronicleDataset, STATIC_VARIS
from evaluator_survival import SurvivalEvaluator
from modeling_strats import Strats
from overall_evaluation import (ALL_DAYS_NAME, DRAWS_NAME, SCALED_UNITS_WARNING, find_runs, label_tasks,
                                load_sex, predictions_path, saved_in_scaled_units)
from preprocess_chronicle import cohort_targets
from utils import Logger

FINISHED = 'Test res of the best checkpoint' # written by main.py after the final test evaluation
RESULTS_NAME, OVERALL_NAME, MASKED_NAME = 'results.csv', 'overall_evaluation.csv', 'masked_features.csv'
KEYS = ['results_dir', 'targets', 'runs', 'percentages', 'num_mask_draws', 'include_static', 'seed',
        'eval_batch_size', 'save_predictions', 'cohort', 'output_dir']


def is_int(x):
    return isinstance(x, int) and not isinstance(x, bool)


def load_config(path):
    """The config, checked: every key present, no unknown key, every value of the right type."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    def fail(message):
        raise SystemExit('Error in '+path+': '+message)
    if not isinstance(cfg, dict):
        fail('not a YAML mapping')
    missing, unknown = [k for k in KEYS if k not in cfg], [k for k in cfg if k not in KEYS]
    if missing or unknown:
        fail('missing keys '+str(missing)+', unknown keys '+str(unknown)+' (expected '+str(KEYS)+')')
    for k in ['results_dir', 'output_dir']:
        if not (isinstance(cfg[k], str) and cfg[k]):
            fail(k+' must be a path')
    if cfg['cohort'] is not None and not (isinstance(cfg['cohort'], str) and cfg['cohort']):
        fail('cohort must be null or a path')
    targets = cfg['targets']
    if targets!='all' and not (isinstance(targets, list) and len(targets)>0
                               and all(isinstance(t, str) and t for t in targets) and len(set(targets))==len(targets)):
        fail('targets must be "all" or a non-empty list of distinct target names')
    runs = cfg['runs']
    if runs is not None and not (isinstance(runs, list) and len(runs)>0
                                 and all(isinstance(r, str) and r for r in runs) and len(set(runs))==len(runs)):
        fail('runs must be null or a non-empty list of distinct run tags such as "1o5"')
    p = cfg['percentages']
    if not (isinstance(p, list) and len(p)>0 and all(isinstance(x, (int, float)) and not isinstance(x, bool)
                                                     and 0<=x<=100 for x in p)):
        fail('percentages must be a non-empty list of numbers from 0 to 100')
    if len({Fraction(str(x)) for x in p})<len(p):
        fail('percentages contains a percentage more than once')
    for k in ['num_mask_draws', 'eval_batch_size']:
        if not (is_int(cfg[k]) and cfg[k]>=1):
            fail(k+' must be an integer >= 1')
    if not (is_int(cfg['seed']) and cfg['seed']>=0):
        fail('seed must be an integer >= 0')
    for k in ['include_static', 'save_predictions']:
        if not isinstance(cfg[k], bool):
            fail(k+' must be true or false')
    if cfg['cohort'] is not None and not cfg['save_predictions']:
        fail('cohort (overall evaluation) needs save_predictions: true')
    return cfg


def num_masked(percentage, num_features):
    """round(percentage % of num_features), halves rounded up, computed exactly."""
    return floor(Fraction(str(percentage))*num_features/100+Fraction(1, 2))


def draw_masked(pool, percentage, mask_draw, seed):
    """The masked features of one percentage and mask draw: drawn anew for every percentage, the same for every
    model with the same feature pool."""
    pool = sorted(pool, key=str)
    p = Fraction(str(percentage))
    rng = np.random.default_rng([seed, p.numerator, p.denominator, mask_draw])
    chosen = rng.choice(len(pool), size=num_masked(percentage, len(pool)), replace=False)
    return [pool[i] for i in sorted(chosen)]


class MaskedDataset:
    """The test split of a ChronicleDataset with some features masked; everything else is the dataset's own (same
    patients, cut days, draws of cut days, labels). Only get_batch reads the observations."""
    def __init__(self, dataset, masked):
        masked = set(masked)
        var_to_ind = {v:i for i,v in enumerate(dataset.norm['variables'])}
        masked_ind = np.array(sorted(var_to_ind[v] for v in masked if v in var_to_ind), dtype=np.int64)
        masked_static = [j for j,v in enumerate(STATIC_VARIS) if v in masked]
        assert len(masked_ind)+len(masked_static)==len(masked), 'Masked features the model does not know'
        self.__dict__.update(dataset.__dict__)
        self.days, self.times, self.values, self.varis = (list(x) for x in
                                                          [dataset.days, dataset.times, dataset.values, dataset.varis])
        num_obs, num_kept = 0, 0
        for i in dataset.splits['test']:
            keep = ~np.isin(dataset.varis[i], masked_ind)
            for name in ['days', 'times', 'values', 'varis']:
                getattr(self, name)[i] = getattr(dataset, name)[i][keep]
            num_obs, num_kept = num_obs+len(keep), num_kept+int(keep.sum())
        self.obs_masked_frac = 1-num_kept/num_obs if num_obs>0 else np.nan
        # a masked static feature = the training mean = 0 after normalization, as a missing one
        self.demo = dataset.demo.copy()
        self.demo[:, masked_static] = 0.0
        self.num_masked_static = len(masked_static)

    all_cuts = ChronicleDataset.all_cuts
    get_batch = ChronicleDataset.get_batch

    def empty_frac(self, ind, cut_days):
        """Share of the cuts (patient index, cut day) without any observation up to the cut day."""
        empty = [np.searchsorted(self.days[i], cut, side='right')==0 for i, cut in zip(ind, cut_days)]
        return float(np.mean(empty)) if empty else np.nan


class AblationModel(torch.nn.Module):
    """The model for samples whose history may have no observation left: those are predicted from the static
    features alone (time-series embedding 0), the others by the model as they are (padding is masked out, so a
    sample's prediction does not depend on the other samples of its batch)."""
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.static_cache = {} # static features (bytes) -> prediction

    def forward(self, values, times, varis, obs_mask, demo):
        has_obs = obs_mask.sum(dim=1)>0
        if bool(has_obs.all()):
            return self.model(values=values, times=times, varis=varis, obs_mask=obs_mask, demo=demo)
        out = torch.empty(len(demo), device=demo.device)
        if bool(has_obs.any()):
            out[has_obs] = self.model(values=values[has_obs], times=times[has_obs], varis=varis[has_obs],
                                      obs_mask=obs_mask[has_obs], demo=demo[has_obs])
        out[~has_obs] = self.static_only(demo[~has_obs])
        return out

    def static_only(self, demo):
        """Predictions from the static features alone. One sample at a time and cached: the same static features
        give exactly the same prediction (in batches of different sizes the last bits can differ, which would
        break the ties a C-index / AUC must see, e.g. when every feature is masked)."""
        out = []
        for row in demo:
            key = row.detach().cpu().numpy().tobytes()
            if key not in self.static_cache:
                self.static_cache[key] = self.static_only_batch(row[None])[0]
            out.append(self.static_cache[key])
        return torch.stack(out)

    def static_only_batch(self, demo):
        """The end of Strats.forward with the time-series embedding ts_emb = 0."""
        m = self.model
        demo_emb = m.demo_emb(demo) if m.args.model_type=='strats' else demo
        ts_emb = torch.zeros((len(demo), m.args.hid_dim), device=demo.device, dtype=demo_emb.dtype)
        logits = m.binary_head(torch.cat((ts_emb, demo_emb), dim=-1))[:,0]
        return m.binary_cls_final(logits, None)


def same_norm(a, b):
    """Whether two normalizations (ChronicleDataset.norm) are the same: same variables in the same order (= the
    embedding indices of the model), same statistics."""
    def close(x, y):
        return x.index.equals(y.index) and np.allclose(x.to_numpy(dtype=float), y.to_numpy(dtype=float),
                                                       rtol=1e-9, atol=0, equal_nan=True)
    return (list(a['variables'])==list(b['variables'])
            and a['means_stds'].columns.equals(b['means_stds'].columns) and close(a['means_stds'], b['means_stds'])
            and close(a['demo_mean'], b['demo_mean']) and close(a['demo_std'], b['demo_std'])
            and np.isclose(float(a['max_day']), float(b['max_day']), rtol=1e-9, atol=0))


def unusable(run_dir):
    """Why a run folder cannot be ablated (missing files, training not finished), or None."""
    missing = [f for f in ['checkpoint_best.bin', 'inference.pkl', 'log.txt'] if not (run_dir/f).is_file()]
    if missing:
        return 'missing '+', '.join(missing)
    if FINISHED not in (run_dir/'log.txt').read_text(errors='replace'):
        return 'training not finished (no "'+FINISHED+'" in log.txt)'
    return None


def write_csv(rows, path):
    """Atomically, so that a merge of another array task never reads half a file."""
    tmp = str(path)+'.tmp'+str(os.getpid())
    pd.DataFrame(rows).to_csv(tmp, index=False)
    os.replace(tmp, path)


def merge(output_dir, name, out_name):
    """Concatenate the per-model CSVs <output_dir>/*/<name> into <output_dir>/<out_name>."""
    parts = [pd.read_csv(p) for p in sorted(output_dir.glob('*/'+name))]
    parts = [p for p in parts if len(p)>0]
    if parts:
        write_csv(pd.concat(parts, ignore_index=True), output_dir/out_name)
    return sum(len(p) for p in parts)


class Ablation:
    def __init__(self, cfg, logger, device, overall_tasks):
        self.cfg, self.logger, self.device, self.overall_tasks = cfg, logger, device, overall_tasks
        self.dataset, self.dataset_key = None, None

    def get_dataset(self, args):
        """The test data of a run: built as in main.py, reused for the next run of the same target."""
        key = (args.dataset, args.target, args.train_frac, args.seed, args.num_eval_draws, args.max_obs)
        if key!=self.dataset_key:
            self.dataset, self.dataset_key = None, None # free the previous one first
            self.dataset = ChronicleDataset(args)
            self.dataset_key = key
        return self.dataset

    def run(self, target, run, run_dir):
        cfg, device = self.cfg, self.device
        with open(run_dir/'inference.pkl', 'rb') as f:
            saved = pickle.load(f)
        assert saved['args'].get('target', 'OS')==target, \
            'inference.pkl is for target '+str(saved['args'].get('target'))+', the folder name says '+target
        args = argparse.Namespace(**saved['args'])
        saved_type, saved_V, saved_D = args.target_type, args.V, args.D
        args.logger = self.logger
        dataset = self.get_dataset(args)
        # the test data must be the one the model was trained for
        assert dataset.args.target_type==saved_type, 'Target type '+dataset.args.target_type+' in the data, '+saved_type+' in the run'
        assert same_norm(dataset.norm, saved['norm']), \
            'The normalization of data/processed/chronicle.pkl differs from the one of this run (inference.pkl): ' \
            'the data were preprocessed again since the training'
        assert (len(dataset.norm['variables']), len(STATIC_VARIS))==(saved_V, saved_D)
        args.target_type, args.V, args.D = saved_type, saved_V, saved_D
        args.eval_batch_size, args.device = cfg['eval_batch_size'], device

        out_dir = Path(cfg['output_dir'])/run_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        for name in [RESULTS_NAME, OVERALL_NAME, MASKED_NAME]: # results of an earlier ablation of this run
            if (out_dir/name).exists():
                (out_dir/name).unlink()
        args.logger = Logger(str(out_dir), 'log.txt')
        args.logger.write('\nAblation of '+str(run_dir)+'\n'+str(cfg))

        model = Strats(args)
        model.load_state_dict(torch.load(run_dir/'checkpoint_best.bin', map_location=device))
        model = AblationModel(model).to(device)
        evaluator = SurvivalEvaluator(args, saved.get('label_scaler'))
        # every day with a target is scored and saved for every target, as main.py does
        if not (run_dir/ALL_DAYS_NAME).is_file():
            args.logger.write('WARNING: the training run has no '+ALL_DAYS_NAME+' (survival run trained before main.py '
                              'saved it): overall_evaluation.py of the training run uses its drawn days, the ablation '
                              'every day')
        scaled_units = saved_in_scaled_units(run_dir)
        pool = list(dataset.norm['variables'])+(STATIC_VARIS if cfg['include_static'] else [])
        assert len(set(pool))==len(pool), 'Feature names occur more than once'
        overall = self.overall_tasks.get(target)
        if self.overall_tasks and overall is None:
            args.logger.write('No overall evaluation for '+target+' (not one of the labels of overall_evaluation.py)')

        test_ind = np.asarray(dataset.splits['test'])
        rows, overall_rows, masked_rows = [], [], []
        for p in cfg['percentages']:
            for mask_draw in range(1, cfg['num_mask_draws']+1):
                masked = draw_masked(pool, p, mask_draw, cfg['seed'])
                masked_rows += [{'percentage':p, 'mask_draw':mask_draw, 'feature':f} for f in masked]
                write_csv(masked_rows, out_dir/MASKED_NAME)
                data = MaskedDataset(dataset, masked)
                args.logger.write('\n'+target+', run '+run+': '+str(p)+' % masked (mask draw '+str(mask_draw)+'): '
                                  +str(len(masked))+' of '+str(len(pool))+' features ('+str(data.num_masked_static)
                                  +' static), '+str(data.obs_masked_frac)+' of the test observations')
                step_dir = out_dir/('mask'+format(p, 'g')+'_draw'+str(mask_draw))
                if cfg['save_predictions']:
                    step_dir.mkdir(exist_ok=True)
                    for name in [DRAWS_NAME, ALL_DAYS_NAME]: # of an earlier ablation of this run
                        if (step_dir/name).exists():
                            (step_dir/name).unlink()
                    with open(step_dir/'log.txt', 'w') as f: # overall_evaluation.py reads the units from here
                        f.write(str(p)+' % of the features masked (mask draw '+str(mask_draw)+'), model '+str(run_dir)
                                +'\n'+('WARNING: '+SCALED_UNITS_WARNING+' (as in the training run)\n' if scaled_units else ''))
                save = (lambda name:str(step_dir/name)) if cfg['save_predictions'] else (lambda name:None)
                res = evaluator.evaluate(model, data, 'test', train_step=None, save_path=save(DRAWS_NAME))
                row = {'target':target, 'target_type':args.target_type, 'run':run, 'run_dir':run_dir.name,
                       'percentage':p, 'mask_draw':mask_draw, 'num_features':len(pool),
                       'num_masked':len(masked), 'num_masked_static':data.num_masked_static,
                       'obs_masked_frac':data.obs_masked_frac, 'metric':evaluator.metric,
                       'value':res[evaluator.metric], 'std':res[evaluator.metric+'_std'],
                       'empty_frac':np.mean([data.empty_frac(test_ind, cuts) for cuts in dataset.eval_cuts['test']])}
                res_all = evaluator.evaluate(model, data, 'test', train_step=None, all_days=True,
                                             save_path=save(ALL_DAYS_NAME))
                # the metric over all days for regression / binary only, as main.py logs it (a C-index over every
                # day of every patient is not Chronicle's C-index)
                row.update({'value_all_days':res_all[evaluator.metric] if args.target_type!='survival' else np.nan,
                            'empty_frac_all_days':data.empty_frac(*dataset.all_cuts('test'))})
                args.logger.write('Test res: '+str(row))
                rows.append(row)
                write_csv(rows, out_dir/RESULTS_NAME)
                if overall is not None:
                    label, metric, definition, score = overall
                    path = predictions_path(step_dir)
                    assert path is not None
                    overall_rows.append({'label':label, 'target':target, 'metric':metric, 'definition':definition,
                                         'run':run, 'run_dir':run_dir.name, 'percentage':p, 'mask_draw':mask_draw,
                                         'predictions':path.name, **score(path)})
                    args.logger.write('Overall evaluation: '+str(overall_rows[-1]))
                    write_csv(overall_rows, out_dir/OVERALL_NAME)
        args.logger.write('\nDone: '+str(out_dir))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='../configs/ablation.yaml')
    # SLURM array: task <task> of <num_tasks> ablates one block of the models
    parser.add_argument('--task', type=int, default=None)
    parser.add_argument('--num_tasks', type=int, default=None)
    cli = parser.parse_args()
    if (cli.task is None)!=(cli.num_tasks is None) or (cli.task is not None and not 0<=cli.task<cli.num_tasks):
        raise SystemExit('--task and --num_tasks: both or none, 0 <= task < num_tasks')
    cfg = load_config(cli.config)

    output_dir = Path(cfg['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = Logger(str(output_dir), 'log.txt' if cli.task is None else 'log_task'+str(cli.task)+'.txt')
    logger.write('\nFeature ablation, config '+cli.config+(' (array task '+str(cli.task)+' of '+str(cli.num_tasks)+')'
                                                           if cli.task is not None else '')+'\n'+str(cfg))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # models: every run of every target in results_dir
    results_dir = Path(cfg['results_dir'])
    if not results_dir.is_dir():
        raise SystemExit('results_dir '+str(results_dir)+' does not exist')
    targets = cohort_targets()
    if cfg['targets']!='all':
        unknown = [t for t in cfg['targets'] if t not in targets]
        if unknown:
            raise SystemExit('Targets not in the cohort: '+str(unknown)+'. Targets: '+str(targets))
        targets = cfg['targets']
    runs = find_runs(results_dir)
    models, skipped_targets = [], []
    for target in targets:
        target_runs = [(t, run, run_dir) for t, run, run_dir in runs
                       if t==target and (cfg['runs'] is None or run in cfg['runs'])]
        if not target_runs:
            skipped_targets.append(target)
        models += target_runs
    logger.write('Targets without a model in '+str(results_dir)+' (skipped): '+str(skipped_targets))
    if cli.task is not None:
        blocks = np.array_split(np.arange(len(models)), cli.num_tasks)
        models = [models[i] for i in blocks[cli.task]]
    logger.write('Models: '+str([run_dir.name for _, _, run_dir in models]))

    overall_tasks = {}
    if cfg['cohort'] is not None:
        overall_tasks = {target:(label, metric, definition, score)
                         for label, target, metric, definition, score in label_tasks(load_sex(cfg['cohort']))}

    ablation = Ablation(cfg, logger, device, overall_tasks)
    done, skipped_runs, failed = [], [], []
    for target, run, run_dir in models:
        reason = unusable(run_dir)
        if reason is not None:
            logger.write('WARNING: skipping '+run_dir.name+': '+reason)
            skipped_runs.append(run_dir.name)
            continue
        try:
            ablation.run(target, run, run_dir)
            done.append(run_dir.name)
        except Exception:
            logger.write('ERROR: ablation of '+run_dir.name+' failed:\n'+traceback.format_exc())
            failed.append(run_dir.name)

    num_rows = merge(output_dir, RESULTS_NAME, 'ablation_results.csv')
    if cfg['cohort'] is not None:
        merge(output_dir, OVERALL_NAME, 'ablation_overall_evaluation.csv')
    logger.write('\nAblated: '+str(len(done))+' models '+str(done)
                 +'\nTargets without a model (skipped): '+str(skipped_targets)
                 +'\nRuns not usable (skipped): '+str(skipped_runs)
                 +'\nFailed: '+str(failed)
                 +'\n'+str(output_dir/'ablation_results.csv')+': '+str(num_rows)+' rows (every model in '+str(output_dir)+')')
    if failed:
        sys.exit(1)


if __name__=='__main__':
    main()
