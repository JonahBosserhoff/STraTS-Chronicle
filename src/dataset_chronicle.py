"""Chronicle cohort for STraTS: one Chronicle target (--target) predicted at a cut day of the history.

STraTS attends in both directions, so a sequence must never contain anything after the
time point it predicts for. Every sample is therefore a patient's history cut at one of
their observed days: all observations up to and including that day go in, and the target
is taken at that day. This is the same prediction point as a <CLS> token in Chronicle,
which carries the targets of that day's first observation.

Targets (any of the cohort, one per run, single head; see preprocess_chronicle.py):
- survival (OS, TTNT, ...: <T>_TIME / <T>_EVENT): (remaining time at the cut day, event),
  Cox loss, C-index.
- binary (a label_* column holding only 0 and 1, which is how chronicle-preprocess decides
  which labels to leave unscaled): BCE with logits, ROC AUC.
- regression (any other label_* column, as scaled by chronicle-preprocess): MSE, Pearson r.
Only days with a target are cut days, as Chronicle scores only <CLS> tokens with a target
(NaN = e.g. no future measurement in the window, or TTNT before the first therapy).
Patients without any such day are dropped for this target.

- Training: a new random cut day every time a patient is drawn, i.e. once per epoch.
- val / test / eval_train: num_eval_draws random cut days per patient, drawn once with a
  fixed seed, so every evaluation (and every run) scores the same cuts. Chronicle reports
  the C-index the same way (c_index_samples random days per patient, mean ± std).
- all_cuts(): every cut day of every patient, for the final evaluation of regression /
  binary targets, which Chronicle scores at every <CLS> token with a target.

If more than max_obs observations remain before the cut, the most recent max_obs are kept.

Otherwise as in dataset.py: only variables seen in the training set, values z-scored per
variable with train statistics, static variables in a separate demo vector. The training
statistics (self.norm) are saved with the model, so external_validation.py can map an
external cohort the same way.
"""
import pickle

import numpy as np
import torch

from preprocess_chronicle import target_columns
from utils import CycleIndex


# Chronicle's baseline features with an STraTS-like counterpart (MIMIC-III: Age, Gender).
# All other Chronicle features (also those recorded at day 0) stay time-series triplets.
STATIC_VARIS = ['age_at_diagnosis', 'gender_female']


class ChronicleDataset:
    def __init__(self, args, external=None) -> None:
        """external: None to train on the Chronicle splits, whose training statistics are kept in
        self.norm (saved with the model). For external validation (data, oc, norm): a cohort in the
        same format, normalized with the norm of the training run, as one split 'external'."""
        if external is None:
            assert args.train_frac==1, 'The Chronicle benchmark uses the full Chronicle splits: pass --train_frac 1.0'
            filepath = '../data/processed/'+args.dataset+'.pkl'
            data, oc, train_ids, val_ids, test_ids = pickle.load(open(filepath,'rb'))
            assert 'day' in oc.columns, filepath+' is in the old OS-only format: re-run preprocess_chronicle.py'
            args.logger.write('\nPreparing dataset '+args.dataset)
            split_ids = {'train':train_ids, 'val':val_ids, 'test':test_ids}
            norm = None
            known_variables = data.loc[data.ts_id.isin(train_ids)].variable.unique()
        else:
            data, oc, norm = external
            args.logger.write('\nPreparing external dataset')
            split_ids = {'external':oc.ts_id.unique()}
            known_variables = np.concatenate((norm['variables'], STATIC_VARIS))

        # target: its columns, and its type (fixed by the training run for external validation)
        targets = target_columns(oc.columns)
        assert args.target in targets, 'Target '+str(args.target)+' not in the data. Targets: '+str(list(targets))
        label_cols = targets[args.target]
        if external is None:
            if len(label_cols)==2:
                args.target_type = 'survival'
            else:
                values = oc[args.target].dropna()
                args.target_type = 'binary' if len(values)>0 and values.isin([0,1]).all() else 'regression'
        assert (len(label_cols)==2)==(args.target_type=='survival'), \
            'Target '+args.target+' is '+args.target_type+' in the training run, but has columns '+str(label_cols)
        args.logger.write('Target: '+args.target+' ('+args.target_type+', columns '+str(label_cols)+')')
        oc = oc[['ts_id', 'day']+label_cols]
        assert not oc.duplicated(['ts_id', 'day']).any(), 'More than one target row per patient and day'

        # keep variables seen in training set only
        delete_variables = np.setdiff1d(data.variable.unique(), known_variables)
        args.logger.write('Removing variables not in training set: '+str(delete_variables))
        data = data.loc[data.variable.isin(known_variables)]

        # split off static variables
        static_ii = data.variable.isin(STATIC_VARIS)
        demo = data.loc[static_ii].groupby(['ts_id', 'variable']).value.last().unstack()
        missing = np.setdiff1d(STATIC_VARIS, demo.columns)
        args.logger.write('Static variables not in the data (filled with the training mean): '+str(missing))
        assert norm is not None or len(missing)==0, 'Static variables not in the data: '+str(missing)
        data = data.loc[~static_ii]

        # A patient without any time-series observation has no day to cut at (and an
        # empty sequence would make attention NaN), so they are dropped, as dataset.py
        # drops time series without data. So is a patient without a target on any of
        # those days (e.g. TTNT for a patient who never started a therapy).
        curr_ids = data.ts_id.unique()
        num_before = [len(ids) for ids in split_ids.values()]
        split_ids = {s:np.intersect1d(ids, curr_ids) for s,ids in split_ids.items()}
        args.logger.write('# patients in '+str(list(split_ids))+': '+str([len(ids) for ids in split_ids.values()])
                          +' (before dropping patients without time-series data: '+str(num_before)+')')
        # training statistics from all of these training patients, whatever the target
        norm_train_ids = split_ids.get('train')
        # cut days = days with a time-series observation and a target
        cuts = data[['ts_id', 'day']].drop_duplicates().merge(oc, on=['ts_id', 'day'], how='inner')
        cuts = cuts.loc[cuts[label_cols].notna().all(axis=1)]
        num_before = [len(ids) for ids in split_ids.values()]
        split_ids = {s:np.intersect1d(ids, cuts.ts_id.unique()) for s,ids in split_ids.items()}
        args.logger.write('# patients with a target in '+str(list(split_ids))+': '
                          +str([len(ids) for ids in split_ids.values()])
                          +' (before dropping patients without a day with a target: '+str(num_before)+')')
        self.ts_ids = np.concatenate(list(split_ids.values()))
        N = len(self.ts_ids)
        ts_id_to_ind = {ts_id:i for i,ts_id in enumerate(self.ts_ids)}
        self.splits = {s:[ts_id_to_ind[i] for i in ids] for s,ids in split_ids.items()}
        if 'train' in self.splits:
            self.splits['eval_train'] = self.splits['train'][:2000]

        # training statistics: static data mean-filled, then normalized; values normalized
        # per variable. With many sparse variables, one seen only once in train has std NaN;
        # it is treated like a constant one (std 1).
        if norm is None:
            demo_train = demo.reindex(index=norm_train_ids, columns=STATIC_VARIS)
            demo_std = demo_train.fillna(demo_train.mean()).std(ddof=0)
            demo_std[demo_std==0] = 1
            means_stds = data.loc[data.ts_id.isin(norm_train_ids)].groupby('variable').value.agg(['mean', 'std'])
            means_stds['std'] = means_stds['std'].fillna(1).replace(0, 1)
            norm = {'demo_mean':demo_train.mean(), 'demo_std':demo_std, 'means_stds':means_stds,
                    'max_day':data.loc[data.ts_id.isin(norm_train_ids), 'day'].max(),
                    'variables':data.variable.unique()}
        self.norm = norm
        demo = demo.reindex(index=self.ts_ids, columns=STATIC_VARIS)

        self.demo = ((demo.fillna(norm['demo_mean'])-norm['demo_mean'])/norm['demo_std']).to_numpy()
        args.D = len(STATIC_VARIS)
        args.logger.write('# static features: '+str(args.D))

        data = data.merge(norm['means_stds'].reset_index(), on='variable', how='left')
        data['value'] = (data['value']-data['mean'])/data['std']
        max_day = norm['max_day']

        var_to_ind = {v:i for i,v in enumerate(norm['variables'])}
        args.V = len(var_to_ind)
        args.logger.write('# TS variables: '+str(args.V))

        # one array per patient, sorted by day
        def per_patient(df, cols):
            """rows of df (patients in self.ts_ids) -> one array per patient and column, sorted by day"""
            df = df.loc[df.ts_id.isin(self.ts_ids)].copy()
            df['ts_ind'] = df['ts_id'].map(ts_id_to_ind)
            df = df.sort_values(['ts_ind', 'day'], kind='stable')
            bounds = np.searchsorted(df['ts_ind'].to_numpy(), np.arange(N+1))
            return [[x[bounds[i]:bounds[i+1]] for i in range(N)] for x in [col(df) for col in cols]]
        self.days, self.times, self.values, self.varis = per_patient(data, [
            lambda df:df['day'].to_numpy(), lambda df:(df['day']/max_day*2-1).to_numpy(),
            lambda df:df['value'].to_numpy(), lambda df:df['variable'].map(var_to_ind).to_numpy()])

        # cut days per patient (sorted) and the target at each: [n, 2] = (remaining time, event)
        # for survival, [n, 1] otherwise
        self.cut_days, self.cut_labels = per_patient(cuts, [
            lambda df:df['day'].to_numpy(), lambda df:df[label_cols].to_numpy(dtype=float)])
        self.survival = args.target_type=='survival'
        def describe(split):
            y = np.concatenate([self.cut_labels[i] for i in self.splits[split]])
            if args.target_type=='regression':
                return 'mean '+str(y[:,0].mean())+', std '+str(y[:,0].std())
            return ('% events: ' if self.survival else '% positive: ')+str(y[:,-1].mean())
        args.logger.write('# days with a target in '+str(list(split_ids))+': '
                          +str([sum(len(self.cut_days[i]) for i in self.splits[s]) for s in split_ids])
                          +'; over these days: '+str({s:describe(s) for s in split_ids}))

        rng = np.random.default_rng(args.seed)
        self.eval_cuts = {split:[np.array([rng.choice(self.cut_days[i]) for i in self.splits[split]])
                                 for _ in range(args.num_eval_draws)]
                          for split in ['eval_train', 'val', 'test', 'external'] if split in self.splits}
        self.args = args
        if 'train' in self.splits:
            self.train_cycler = CycleIndex(self.splits['train'], args.train_batch_size)

    def all_cuts(self, split):
        """(patient index, cut day) for every cut day of every patient of a split."""
        ind = self.splits[split]
        return (np.repeat(ind, [len(self.cut_days[i]) for i in ind]),
                np.concatenate([self.cut_days[i] for i in ind]))

    def get_batch(self, ind=None, cut_days=None):
        if ind is None:
            ind = self.train_cycler.get_batch_ind()
            cut_days = [np.random.choice(self.cut_days[i]) for i in ind]
        values, times, varis, labels = [], [], [], []
        for i, cut in zip(ind, cut_days):
            end = np.searchsorted(self.days[i], cut, side='right') # observations up to and incl. cut day
            start = max(0, end-self.args.max_obs)
            values.append(self.values[i][start:end])
            times.append(self.times[i][start:end])
            varis.append(self.varis[i][start:end])
            pos = np.searchsorted(self.cut_days[i], cut)
            assert pos<len(self.cut_days[i]) and self.cut_days[i][pos]==cut, 'Not a cut day of patient '+str(i)
            labels.append(self.cut_labels[i][pos])
        num_obs = [len(v) for v in values]
        max_obs = max(num_obs)
        def pad(seqs):
            return np.stack([np.pad(s, (0, max_obs-len(s))) for s in seqs])
        obs_mask = [[1]*l+[0]*(max_obs-l) for l in num_obs]
        labels = np.stack(labels)
        if not self.survival:
            labels = labels[:,0]
        return {'values':torch.FloatTensor(pad(values)), 'times':torch.FloatTensor(pad(times)),
                'varis':torch.IntTensor(pad(varis)), 'obs_mask':torch.IntTensor(obs_mask),
                'demo':torch.FloatTensor(self.demo[ind]),
                'labels':torch.FloatTensor(labels)} # survival: [N, 2] = (remaining time in days, event); else [N]
