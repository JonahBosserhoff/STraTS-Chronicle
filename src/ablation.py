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
day are saved for every target). Only the input changes and, as in Chronicle, the days and patients without any
observation left (see Masking), so 0 % reproduces "Test res of the best checkpoint" of the run.

Masked features, one of two sources:
- chronicle_ablation: a path. The features Chronicle removed in its own feature ablation (chronicle-evaluate ...
  feature_ablation_percent=<p> feature_ablation_fold=<f>, one <results_dir>/dropped_features.csv per step) are masked
  here too, so STraTS and Chronicle lose exactly the same features. Every dropped_features.csv below the path (any
  depth, e.g. <path>/percent_10/fold_5/dropped_features.csv) is one step: percentage and fold from its columns
  ablation_percent and fold, or, for a file without rows (nothing removed), from the folder names percent_<p> and
  fold_<f>. The fold becomes the mask draw (mask<p>_draw<fold>/). percentages: null = every step found, a list = only
  those percentages (0 = nothing masked, needs no file). num_mask_draws, include_static and seed are not used. A
  removed feature the model does not know (Chronicle's vocabulary also has features seen only outside the training
  split, which STraTS drops) is not in the model's input anyway; it is listed (in_model false in
  masked_features.csv, num_masked_unknown in the results).
- chronicle_ablation: null. Drawn here: for every percentage p of the config and every mask draw, round(p % x
  #features) features are drawn at random from all features the model knows: its variable vocabulary (every
  time-series feature of the training set) and, with include_static, the static features age_at_diagnosis and
  gender_female. Every percentage is drawn anew, independently of the other percentages (the 20 % features do not
  contain the 10 % ones). A draw depends only on seed, p and the mask draw (and the feature vocabulary), so every
  model (every target and run) gets the same masked features at the same p and mask draw. num_mask_draws > 1 draws
  several independent feature sets per p.

Masking.
- A masked time-series feature is removed from the test data: none of its observations is seen, as if it had
  never been recorded (the max_obs most recent observations are taken from the remaining ones).
- A masked static feature is set to the training mean, as a missing static feature is.
- As in Chronicle, a day without any observation left is not scored: Chronicle predicts a day through the <CLS>
  token it puts after that day's observations, so a day whose observations are all masked has no prediction there.
  A day counts as observed if a time-series observation or a static feature that is not masked is left on it (in
  Chronicle age_at_diagnosis and gender_female are observations like any other). Their share of the test days with
  a target is reported (days_dropped_frac). A test patient without such a day left is not scored at all
  (num_patients_dropped; Chronicle leaves out a patient without any observation left, ChronicleDataset
  .ablate_features, and has no prediction with a target for one whose days with a target are all gone).
- The num_eval_draws random cut days per test patient are drawn again from the days left, with the run's seed
  (ChronicleDataset.draw_eval_cuts), as Chronicle samples its random days from the days it predicted. With nothing
  removed they are exactly the draws of the training run.
- A cut whose history has no time-series observation left (only a static feature left on it and before it) is
  predicted from the static features alone, with the time-series embedding set to 0: STraTS cannot attend over an
  empty sequence. Their share is reported (empty_frac).

Output in output_dir:
    ablation_results.csv             one row per model, p and mask draw: metric, std, share of masked observations,
                                     patients not scored, share of cuts without observations, source of the
                                     masked features (mask_source: "drawn" or the Chronicle file), ...
    ablation_overall_evaluation.csv  with cohort: the metrics of overall_evaluation.py (C-index for OS / TTNT,
                                     AUROC with Chronicle's thresholds for the labs, ...) of the same predictions
    log.txt                          (log_task<i>.txt for SLURM array task i)
    <run folder>/                    per model: results.csv, overall_evaluation.csv, masked_features.csv
                                     (p, mask_draw, feature, in_model), log.txt and, with save_predictions, one folder
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
import re
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
KEYS = ['results_dir', 'targets', 'runs', 'chronicle_ablation', 'percentages', 'num_mask_draws', 'include_static',
        'seed', 'eval_batch_size', 'save_predictions', 'cohort', 'output_dir']
# Chronicle's feature ablation (chronicle-evaluate, chronicle.cli.evaluate.ablate_features): one file per step
CHRONICLE_NAME, CHRONICLE_COLUMNS = 'dropped_features.csv', ['fold', 'ablation_percent', 'feature']
CHRONICLE_DIR = re.compile(r'^(?P<key>percent|fold)_(?P<value>[0-9]+(?:\.[0-9]+)?)$')


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
    for k in ['cohort', 'chronicle_ablation']:
        if cfg[k] is not None and not (isinstance(cfg[k], str) and cfg[k]):
            fail(k+' must be null or a path')
    targets = cfg['targets']
    if targets!='all' and not (isinstance(targets, list) and len(targets)>0
                               and all(isinstance(t, str) and t for t in targets) and len(set(targets))==len(targets)):
        fail('targets must be "all" or a non-empty list of distinct target names')
    runs = cfg['runs']
    if runs is not None and not (isinstance(runs, list) and len(runs)>0
                                 and all(isinstance(r, str) and r for r in runs) and len(set(runs))==len(runs)):
        fail('runs must be null or a non-empty list of distinct run tags such as "1o5"')
    p = cfg['percentages']
    if p is None:
        if cfg['chronicle_ablation'] is None:
            fail('percentages: null (every step of Chronicle) needs chronicle_ablation; without it, a list')
    else:
        if not (isinstance(p, list) and len(p)>0 and all(isinstance(x, (int, float)) and not isinstance(x, bool)
                                                         and 0<=x<=100 for x in p)):
            fail('percentages must be a non-empty list of numbers from 0 to 100'
                 +(' or null' if cfg['chronicle_ablation'] is not None else ''))
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


def as_number(x):
    """A Fraction as int if whole, else float (10 rather than 10.0 in folder names and CSVs)."""
    return int(x) if x.denominator==1 else float(x)


def chronicle_steps(root, percentages):
    """The steps of Chronicle's feature ablation below root: [{percentage, mask_draw (= Chronicle's fold), features
    (the removed ones, as Chronicle wrote them), source (the file)}], sorted by percentage and fold. percentages: None
    = every step found; a list = the steps of these percentages (each must have been run by Chronicle, except 0, a
    step with nothing masked and no file)."""
    root = Path(root)
    if not root.is_dir():
        raise SystemExit('chronicle_ablation '+str(root)+' does not exist')
    files = sorted(root.rglob(CHRONICLE_NAME))
    if not files:
        raise SystemExit('chronicle_ablation '+str(root)+': no '+CHRONICLE_NAME+' below it (written by '
                         'chronicle-evaluate ... feature_ablation_percent=<p> feature_ablation_fold=<f>)')
    steps = {}
    for path in files:
        def fail(message):
            raise SystemExit(str(path)+': '+message)
        df = pd.read_csv(path)
        if missing := [c for c in CHRONICLE_COLUMNS if c not in df.columns]:
            fail('no column(s) '+str(missing)+' (expected '+str(CHRONICLE_COLUMNS)+', as chronicle-evaluate writes)')
        from_dirs = {}
        for part in path.relative_to(root).parent.parts:
            if m := CHRONICLE_DIR.match(part):
                from_dirs[m['key']] = Fraction(m['value'])
        from_csv = {}
        for col, key in [('ablation_percent', 'percent'), ('fold', 'fold')]:
            values = df[col].dropna().unique()
            if len(values)>1:
                fail('more than one '+col+': '+str(sorted(values)))
            if len(values)==1:
                from_csv[key] = Fraction(str(values[0]))
        step = {}
        for key in ['percent', 'fold']:
            if key in from_csv and key in from_dirs and from_csv[key]!=from_dirs[key]:
                fail(key+' '+str(as_number(from_csv[key]))+' in the file, but the folder name says '
                     +str(as_number(from_dirs[key])))
            if key not in from_csv and key not in from_dirs:
                fail('no '+key+': neither in the file (no rows) nor in a folder name '+key+'_<number>')
            step[key] = from_csv.get(key, from_dirs.get(key))
        percent, fold = step['percent'], step['fold']
        if not (0<percent<100 and fold.denominator==1 and fold>=0):
            fail('percent '+str(as_number(percent))+', fold '+str(as_number(fold))
                 +': expected 0 < percent < 100 and an integer fold >= 0')
        features = df['feature'].astype(str).tolist()
        if len(set(features))<len(features):
            fail('a feature occurs more than once')
        key = (percent, int(fold))
        if key in steps:
            fail('percent '+str(as_number(percent))+', fold '+str(int(fold))+' also in '+str(steps[key]['source']))
        steps[key] = {'percentage':as_number(percent), 'mask_draw':int(fold), 'features':features, 'source':path}
    if percentages is not None:
        wanted = {Fraction(str(p)) for p in percentages}
        found = {p for p, _ in steps}
        if missing := sorted(wanted-found-{0}):
            raise SystemExit('percentages '+str([as_number(p) for p in missing])+' are not in chronicle_ablation '
                             +str(root)+', which has '+str([as_number(p) for p in sorted(found)]))
        steps = {k:s for k,s in steps.items() if k[0] in wanted}
        if 0 in wanted:
            steps[(Fraction(0), 0)] = {'percentage':0, 'mask_draw':0, 'features':[], 'source':None}
    return [steps[k] for k in sorted(steps)]


class MaskedDataset:
    """The test split of a ChronicleDataset with some features masked, scored as Chronicle scores it: Chronicle
    predicts a day through the <CLS> token after that day's observations, so a day without any observation left is
    not predicted, and a patient without such a day is not scored. Everything else is the dataset's own (labels,
    the other splits); the random cut days of the test draws are drawn again from the days left (with the run's
    seed, so with nothing removed they are the draws of the training run). Only get_batch reads the observations."""
    def __init__(self, dataset, masked):
        masked = set(masked)
        var_to_ind = {v:i for i,v in enumerate(dataset.norm['variables'])}
        masked_ind = np.array(sorted(var_to_ind[v] for v in masked if v in var_to_ind), dtype=np.int64)
        masked_static = [j for j,v in enumerate(STATIC_VARIS) if v in masked]
        assert len(masked_ind)+len(masked_static)==len(masked), 'Masked features the model does not know'
        self.__dict__.update(dataset.__dict__)
        self.days, self.times, self.values, self.varis, self.cut_days, self.cut_labels = (list(x) for x in [
            dataset.days, dataset.times, dataset.values, dataset.varis, dataset.cut_days, dataset.cut_labels])
        scored = []
        num_obs, num_kept, num_cuts, num_cuts_kept = 0, 0, 0, 0
        for i in dataset.splits['test']:
            keep = ~np.isin(dataset.varis[i], masked_ind)
            for name in ['days', 'times', 'values', 'varis']:
                getattr(self, name)[i] = getattr(dataset, name)[i][keep]
            num_obs, num_kept = num_obs+len(keep), num_kept+int(keep.sum())
            # days with an observation left: time series, or a static feature (an observation in Chronicle)
            static_kept = ~np.isin(dataset.static_varis[i], masked_static)
            observed = np.concatenate((self.days[i], dataset.static_days[i][static_kept]))
            has_obs = np.isin(dataset.cut_days[i], observed)
            self.cut_days[i], self.cut_labels[i] = dataset.cut_days[i][has_obs], dataset.cut_labels[i][has_obs]
            num_cuts, num_cuts_kept = num_cuts+len(has_obs), num_cuts_kept+int(has_obs.sum())
            if has_obs.any():
                scored.append(i)
        self.obs_masked_frac = 1-num_kept/num_obs if num_obs>0 else np.nan
        self.days_dropped_frac = 1-num_cuts_kept/num_cuts if num_cuts>0 else np.nan
        self.num_patients_dropped = len(dataset.splits['test'])-len(scored)
        # new dicts: the dataset's own splits and draws stay as they are for the next mask
        self.splits = {**dataset.splits, 'test':scored}
        self.eval_cuts = self.draw_eval_cuts()
        # a masked static feature = the training mean = 0 after normalization, as a missing one
        self.demo = dataset.demo.copy()
        self.demo[:, masked_static] = 0.0
        self.num_masked_static = len(masked_static)

    draw_eval_cuts = ChronicleDataset.draw_eval_cuts
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
    def __init__(self, cfg, logger, device, overall_tasks, chronicle=None):
        """chronicle: the steps of Chronicle's feature ablation (chronicle_steps), or None to draw the features."""
        self.cfg, self.logger, self.device, self.overall_tasks = cfg, logger, device, overall_tasks
        self.chronicle = chronicle
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
        known = list(dataset.norm['variables'])+STATIC_VARIS # every feature the model knows
        assert len(set(known))==len(known), 'Feature names occur more than once'
        if self.chronicle is not None: # Chronicle's removed features, the same for every model
            pool, steps = known, self.chronicle
        else:
            pool = list(dataset.norm['variables'])+(STATIC_VARIS if cfg['include_static'] else [])
            steps = [{'percentage':p, 'mask_draw':mask_draw, 'features':draw_masked(pool, p, mask_draw, cfg['seed']),
                      'source':None}
                     for p in cfg['percentages'] for mask_draw in range(1, cfg['num_mask_draws']+1)]
        overall = self.overall_tasks.get(target)
        if self.overall_tasks and overall is None:
            args.logger.write('No overall evaluation for '+target+' (not one of the labels of overall_evaluation.py)')

        rows, overall_rows, masked_rows = [], [], []
        for step in steps:
            p, mask_draw, features = step['percentage'], step['mask_draw'], step['features']
            in_model = set(known)
            masked = [f for f in features if f in in_model]
            unknown = [f for f in features if f not in in_model]
            source = 'drawn' if step['source'] is None else str(step['source'])
            masked_rows += [{'percentage':p, 'mask_draw':mask_draw, 'feature':f, 'in_model':f in in_model}
                            for f in features]
            write_csv(masked_rows, out_dir/MASKED_NAME)
            data = MaskedDataset(dataset, masked)
            num_scored = len(data.splits['test'])
            args.logger.write('\n'+target+', run '+run+': '+str(p)+' % masked (mask draw '+str(mask_draw)+', '+source
                              +'): '+str(len(masked))+' of '+str(len(pool))+' features ('+str(data.num_masked_static)
                              +' static), '+str(data.obs_masked_frac)+' of the test observations; '
                              +str(data.days_dropped_frac)+' of the days with a target without any observation left '
                              'and '+str(data.num_patients_dropped)+' test patients without such a day, not scored'
                              +(('; '+str(len(unknown))+' removed features the model does not know (not in its input '
                                 'anyway): '+str(unknown)) if unknown else ''))
            row = {'target':target, 'target_type':args.target_type, 'run':run, 'run_dir':run_dir.name,
                   'percentage':p, 'mask_draw':mask_draw, 'mask_source':source, 'num_features':len(pool),
                   'num_masked':len(masked), 'num_masked_static':data.num_masked_static,
                   'num_masked_unknown':len(unknown), 'obs_masked_frac':data.obs_masked_frac,
                   'days_dropped_frac':data.days_dropped_frac, 'num_patients':num_scored,
                   'num_patients_dropped':data.num_patients_dropped,
                   'metric':evaluator.metric}
            if num_scored==0:
                args.logger.write('WARNING: no test patient has a day with a target and an observation left, '
                                  'nothing scored')
                rows.append({**row, 'value':np.nan, 'std':np.nan, 'empty_frac':np.nan, 'value_all_days':np.nan,
                             'empty_frac_all_days':np.nan})
                write_csv(rows, out_dir/RESULTS_NAME)
                continue
            step_dir = out_dir/('mask'+format(p, 'g')+'_draw'+str(mask_draw))
            if cfg['save_predictions']:
                step_dir.mkdir(exist_ok=True)
                for name in [DRAWS_NAME, ALL_DAYS_NAME]: # of an earlier ablation of this run
                    if (step_dir/name).exists():
                        (step_dir/name).unlink()
                with open(step_dir/'log.txt', 'w') as f: # overall_evaluation.py reads the units from here
                    f.write(str(p)+' % of the features masked (mask draw '+str(mask_draw)+', '+source+'), model '
                            +str(run_dir)+'\n'
                            +('WARNING: '+SCALED_UNITS_WARNING+' (as in the training run)\n' if scaled_units else ''))
            save = (lambda name:str(step_dir/name)) if cfg['save_predictions'] else (lambda name:None)
            res = evaluator.evaluate(model, data, 'test', train_step=None, save_path=save(DRAWS_NAME))
            test_ind = np.asarray(data.splits['test']) # the scored patients
            row.update({'value':res[evaluator.metric], 'std':res[evaluator.metric+'_std'],
                        'empty_frac':np.mean([data.empty_frac(test_ind, cuts) for cuts in data.eval_cuts['test']])})
            res_all = evaluator.evaluate(model, data, 'test', train_step=None, all_days=True,
                                         save_path=save(ALL_DAYS_NAME))
            # the metric over all days for regression / binary only, as main.py logs it (a C-index over every
            # day of every patient is not Chronicle's C-index)
            row.update({'value_all_days':res_all[evaluator.metric] if args.target_type!='survival' else np.nan,
                        'empty_frac_all_days':data.empty_frac(*data.all_cuts('test'))})
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

    chronicle = None
    if cfg['chronicle_ablation'] is not None:
        chronicle = chronicle_steps(cfg['chronicle_ablation'], cfg['percentages'])
        logger.write('Masked features from Chronicle\'s feature ablation '+str(cfg['chronicle_ablation'])
                     +' (num_mask_draws, include_static and seed are not used), '+str(len(chronicle))+' steps '
                     '(percentage, fold = mask draw, number of removed features, file):\n'
                     +'\n'.join(str((s['percentage'], s['mask_draw'], len(s['features']), str(s['source'])))
                                for s in chronicle))

    ablation = Ablation(cfg, logger, device, overall_tasks, chronicle)
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
