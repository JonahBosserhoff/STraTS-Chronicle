"""Training runs of a training config (configs/train_chronicle.yaml): every target x run is one task, i.e. one
call of main.py --dataset chronicle on one GPU. Used by scripts/train_chronicle.sbatch; from src/:

    python train_chronicle.py --config ../configs/train_chronicle.yaml --count    # prints "<#tasks> <max_parallel>"
    python train_chronicle.py --config ../configs/train_chronicle.yaml --list     # the tasks
    python train_chronicle.py --config ../configs/train_chronicle.yaml --task 3   # runs task 3
    python train_chronicle.py --config ../configs/train_chronicle.yaml            # runs the only task (1 target, 1 run)

Task i trains target targets[i // #runs] with run runs[i % #runs] (as --run <r>o<num_runs>).
"""
import argparse
import os
import sys

import yaml

from preprocess_chronicle import cohort_targets

KEYS = ['targets', 'num_runs', 'runs', 'max_parallel', 'hyperparameters']
# set by this script (or fixed for the Chronicle benchmark), not in hyperparameters
RESERVED = ['dataset', 'target', 'run', 'train_frac', 'output_dir']


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
    targets = cfg['targets']
    if targets!='all' and not (isinstance(targets, list) and len(targets)>0
                               and all(isinstance(t, str) and t for t in targets)):
        fail('targets must be "all" or a non-empty list of target names')
    if isinstance(targets, list) and len(set(targets))<len(targets):
        fail('targets contains a target more than once')
    if not (is_int(cfg['num_runs']) and cfg['num_runs']>=1):
        fail('num_runs must be an integer >= 1')
    runs = cfg['runs']
    if runs is not None and not (isinstance(runs, list) and len(runs)>0 and all(is_int(r) for r in runs)
                                 and len(set(runs))==len(runs) and all(1<=r<=cfg['num_runs'] for r in runs)):
        fail('runs must be null or a non-empty list of distinct integers from 1 to num_runs')
    if not (is_int(cfg['max_parallel']) and cfg['max_parallel']>=1):
        fail('max_parallel must be an integer >= 1')
    hyper = cfg['hyperparameters'] or {}
    if not isinstance(hyper, dict):
        fail('hyperparameters must be a mapping (main.py argument -> value)')
    for k, v in hyper.items():
        if k in RESERVED:
            fail('hyperparameters: '+str(k)+' is set by train_chronicle.py, not in the config')
        if not (isinstance(k, str) and (v is None or isinstance(v, str)
                                        or (isinstance(v, (int, float)) and not isinstance(v, bool)))):
            fail('hyperparameters: '+str(k)+': '+repr(v)+' (a main.py argument needs a number or a string)')
    cfg['hyperparameters'] = hyper
    return cfg


def get_tasks(cfg):
    """(target, run) of every task, in task order."""
    targets = cohort_targets() if cfg['targets']=='all' else cfg['targets']
    runs = cfg['runs'] if cfg['runs'] is not None else list(range(1, cfg['num_runs']+1))
    return [(target, run) for target in targets for run in runs]


def main_args(cfg, target, run):
    """Arguments of main.py for one task (a hyperparameter set to null keeps the default of main.py)."""
    argv = ['--dataset', 'chronicle', '--target', target, '--train_frac', '1.0',
            '--run', str(run)+'o'+str(cfg['num_runs'])]
    for k, v in cfg['hyperparameters'].items():
        if v is not None:
            argv += ['--'+k, str(v)]
    return argv


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='../configs/train_chronicle.yaml')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--count', action='store_true', help='print "<number of tasks> <max_parallel>"')
    mode.add_argument('--list', action='store_true', help='print the tasks')
    mode.add_argument('--task', type=int, default=None, help='run this task (SLURM array task id)')
    # bounds of the SLURM array: must be exactly 0 to #tasks-1, so that no task is left out or run twice
    parser.add_argument('--array_min', type=int, default=None)
    parser.add_argument('--array_max', type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    tasks = get_tasks(cfg)
    if args.count:
        print(len(tasks), min(cfg['max_parallel'], len(tasks)))
        sys.exit(0)
    if args.list:
        for i, (target, run) in enumerate(tasks):
            print(i, 'python main.py '+' '.join(main_args(cfg, target, run)))
        sys.exit(0)

    if args.task is None:
        if len(tasks)!=1:
            raise SystemExit(args.config+' has '+str(len(tasks))+' runs (targets x runs): submit them as an '
                             'array with "bash scripts/train_chronicle.sbatch" (not sbatch), see the header of '
                             'that script')
        task = 0
    else:
        task = args.task
        if args.array_min is not None or args.array_max is not None:
            if (args.array_min, args.array_max)!=(0, len(tasks)-1):
                raise SystemExit('The SLURM array is '+str(args.array_min)+'-'+str(args.array_max)+', but '
                                 +args.config+' has '+str(len(tasks))+' runs (array 0-'+str(len(tasks)-1)+'). '
                                 'Submit with "bash scripts/train_chronicle.sbatch", which sets the array.')
        if not 0<=task<len(tasks):
            raise SystemExit('Task '+str(task)+' out of range: '+args.config+' has '+str(len(tasks))+' runs')
    target, run = tasks[task]
    argv = ['main.py']+main_args(cfg, target, run)
    print('Task '+str(task)+' of '+str(len(tasks))+': target '+target+', run '+str(run)+'o'+str(cfg['num_runs']), flush=True)
    print('python '+' '.join(argv), flush=True)
    os.execv(sys.executable, [sys.executable]+argv)
