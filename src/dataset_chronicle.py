"""Chronicle cohort for STraTS: overall survival (OS) predicted at a cut day of the history.

STraTS attends in both directions, so a sequence must never contain anything after the
time point it predicts for. Every sample is therefore a patient's history cut at one of
their observed days: all observations up to and including that day go in, and the OS
label is taken at that day (remaining time = os_end_day - cut day, event). This is the
same prediction point as a <CLS> token in Chronicle.

- Training: a new random cut day every time a patient is drawn, i.e. once per epoch.
- val / test / eval_train: num_eval_draws random cut days per patient, drawn once with a
  fixed seed, so every evaluation (and every run) scores the same cuts. Chronicle reports
  the C-index the same way (c_index_samples random days per patient, mean ± std).

If more than max_obs observations remain before the cut, the most recent max_obs are kept.

Otherwise as in dataset.py: only variables seen in the training set, values z-scored per
variable with train statistics, static variables in a separate demo vector.
"""
import pickle

import numpy as np
import torch

from utils import CycleIndex


# Chronicle's baseline features with an STraTS-like counterpart (MIMIC-III: Age, Gender).
# All other Chronicle features (also those recorded at day 0) stay time-series triplets.
STATIC_VARIS = ['age_at_diagnosis', 'gender_female']


class ChronicleDataset:
    def __init__(self, args) -> None:
        assert args.train_frac==1, 'The Chronicle benchmark uses the full Chronicle splits: pass --train_frac 1.0'
        filepath = '../data/processed/'+args.dataset+'.pkl'
        data, oc, train_ids, val_ids, test_ids = pickle.load(open(filepath,'rb'))
        args.logger.write('\nPreparing dataset '+args.dataset)

        # keep variables seen in training set only
        train_variables = data.loc[data.ts_id.isin(train_ids)].variable.unique()
        delete_variables = np.setdiff1d(data.variable.unique(), train_variables)
        args.logger.write('Removing variables not in training set: '+str(delete_variables))
        data = data.loc[data.variable.isin(train_variables)]

        # split off static variables
        static_ii = data.variable.isin(STATIC_VARIS)
        demo = data.loc[static_ii].groupby(['ts_id', 'variable']).value.last().unstack()
        missing = np.setdiff1d(STATIC_VARIS, demo.columns)
        assert len(missing)==0, 'Static variables not in the data: '+str(missing)
        data = data.loc[~static_ii]

        # A patient without any time-series observation has no day to cut at (and an
        # empty sequence would make attention NaN), so they are dropped, as dataset.py
        # drops time series without data.
        curr_ids = data.ts_id.unique()
        num_before = [len(train_ids), len(val_ids), len(test_ids)]
        train_ids, val_ids, test_ids = [np.intersect1d(ids, curr_ids) for ids in (train_ids, val_ids, test_ids)]
        args.logger.write('# train, val, test patients: '+str([len(train_ids), len(val_ids), len(test_ids)])
                          +' (before dropping patients without time-series data: '+str(num_before)+')')
        self.ts_ids = np.concatenate((train_ids, val_ids, test_ids))
        N = len(self.ts_ids)
        ts_id_to_ind = {ts_id:i for i,ts_id in enumerate(self.ts_ids)}
        self.splits = {'train':[ts_id_to_ind[i] for i in train_ids],
                       'val':[ts_id_to_ind[i] for i in val_ids],
                       'test':[ts_id_to_ind[i] for i in test_ids]}
        self.splits['eval_train'] = self.splits['train'][:2000]

        # static data: mean-fill missing values, then normalize (train statistics)
        demo = demo.reindex(index=self.ts_ids, columns=STATIC_VARIS)
        demo = demo.fillna(demo.loc[train_ids].mean())
        means, stds = demo.loc[train_ids].mean(), demo.loc[train_ids].std(ddof=0)
        stds[stds==0] = 1
        self.demo = ((demo-means)/stds).to_numpy()
        args.D = len(STATIC_VARIS)
        args.logger.write('# static features: '+str(args.D))

        # normalize values. With many sparse variables, one seen only once in train has
        # std NaN; it is treated like a constant one (std 1).
        means_stds = data.loc[data.ts_id.isin(train_ids)].groupby('variable').value.agg(['mean', 'std'])
        means_stds['std'] = means_stds['std'].fillna(1).replace(0, 1)
        data = data.merge(means_stds.reset_index(), on='variable', how='left')
        data['value'] = (data['value']-data['mean'])/data['std']
        max_day = data.loc[data.ts_id.isin(train_ids), 'day'].max()

        variables = data.variable.unique()
        var_to_ind = {v:i for i,v in enumerate(variables)}
        args.V = len(variables)
        args.logger.write('# TS variables: '+str(args.V))

        # one array per patient, sorted by day
        data['ts_ind'] = data['ts_id'].map(ts_id_to_ind)
        data = data.sort_values(['ts_ind', 'day'], kind='stable')
        bounds = np.searchsorted(data['ts_ind'].to_numpy(), np.arange(N+1))
        def per_patient(x):
            return [x[bounds[i]:bounds[i+1]] for i in range(N)]
        self.days = per_patient(data['day'].to_numpy())
        self.times = per_patient((data['day']/max_day*2-1).to_numpy())
        self.values = per_patient(data['value'].to_numpy())
        self.varis = per_patient(data['variable'].map(var_to_ind).to_numpy())
        self.cut_days = [np.unique(d) for d in self.days]

        # OS label: remaining time at a cut day = os_end_day - cut day
        oc = oc.set_index('ts_id').loc[self.ts_ids]
        self.os_end_day = oc['os_end_day'].to_numpy()
        self.os_event = oc['os_event'].to_numpy()
        args.logger.write('% events in train, val, test: '
                          +str([self.os_event[self.splits[s]].mean() for s in ['train', 'val', 'test']]))

        rng = np.random.default_rng(args.seed)
        self.eval_cuts = {split:[np.array([rng.choice(self.cut_days[i]) for i in self.splits[split]])
                                 for _ in range(args.num_eval_draws)]
                          for split in ['eval_train', 'val', 'test']}
        self.args = args
        self.train_cycler = CycleIndex(self.splits['train'], args.train_batch_size)

    def get_batch(self, ind=None, cut_days=None):
        if ind is None:
            ind = self.train_cycler.get_batch_ind()
            cut_days = [np.random.choice(self.cut_days[i]) for i in ind]
        values, times, varis = [], [], []
        for i, cut in zip(ind, cut_days):
            end = np.searchsorted(self.days[i], cut, side='right') # observations up to and incl. cut day
            start = max(0, end-self.args.max_obs)
            values.append(self.values[i][start:end])
            times.append(self.times[i][start:end])
            varis.append(self.varis[i][start:end])
        num_obs = [len(v) for v in values]
        max_obs = max(num_obs)
        def pad(seqs):
            return np.stack([np.pad(s, (0, max_obs-len(s))) for s in seqs])
        obs_mask = [[1]*l+[0]*(max_obs-l) for l in num_obs]
        labels = np.stack([self.os_end_day[ind]-np.asarray(cut_days), self.os_event[ind]], axis=1)
        return {'values':torch.FloatTensor(pad(values)), 'times':torch.FloatTensor(pad(times)),
                'varis':torch.IntTensor(pad(varis)), 'obs_mask':torch.IntTensor(obs_mask),
                'demo':torch.FloatTensor(self.demo[ind]),
                'labels':torch.FloatTensor(labels)} # labels: [N, 2] = (remaining OS time in days, event)
