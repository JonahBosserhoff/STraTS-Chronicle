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
variable with train statistics, static variables in a separate demo vector. The training
statistics (self.norm) are saved with the model, so external_validation.py can map an
external cohort the same way.
"""
import pickle

import numpy as np
import torch

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
            args.logger.write('\nPreparing dataset '+args.dataset)
            split_ids = {'train':train_ids, 'val':val_ids, 'test':test_ids}
            norm = None
            known_variables = data.loc[data.ts_id.isin(train_ids)].variable.unique()
        else:
            data, oc, norm = external
            args.logger.write('\nPreparing external dataset')
            split_ids = {'external':oc.ts_id.unique()}
            known_variables = np.concatenate((norm['variables'], STATIC_VARIS))

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
        # drops time series without data.
        curr_ids = data.ts_id.unique()
        num_before = [len(ids) for ids in split_ids.values()]
        split_ids = {s:np.intersect1d(ids, curr_ids) for s,ids in split_ids.items()}
        args.logger.write('# patients in '+str(list(split_ids))+': '+str([len(ids) for ids in split_ids.values()])
                          +' (before dropping patients without time-series data: '+str(num_before)+')')
        self.ts_ids = np.concatenate(list(split_ids.values()))
        N = len(self.ts_ids)
        ts_id_to_ind = {ts_id:i for i,ts_id in enumerate(self.ts_ids)}
        self.splits = {s:[ts_id_to_ind[i] for i in ids] for s,ids in split_ids.items()}
        if 'train' in self.splits:
            self.splits['eval_train'] = self.splits['train'][:2000]

        # training statistics: static data mean-filled, then normalized; values normalized
        # per variable. With many sparse variables, one seen only once in train has std NaN;
        # it is treated like a constant one (std 1).
        demo = demo.reindex(index=self.ts_ids, columns=STATIC_VARIS)
        if norm is None:
            train_ids = split_ids['train']
            demo_train = demo.loc[train_ids]
            demo_std = demo_train.fillna(demo_train.mean()).std(ddof=0)
            demo_std[demo_std==0] = 1
            means_stds = data.loc[data.ts_id.isin(train_ids)].groupby('variable').value.agg(['mean', 'std'])
            means_stds['std'] = means_stds['std'].fillna(1).replace(0, 1)
            norm = {'demo_mean':demo_train.mean(), 'demo_std':demo_std, 'means_stds':means_stds,
                    'max_day':data.loc[data.ts_id.isin(train_ids), 'day'].max(),
                    'variables':data.variable.unique()}
        self.norm = norm

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
        args.logger.write('% events in '+str(list(split_ids))+': '
                          +str([self.os_event[self.splits[s]].mean() for s in split_ids]))

        rng = np.random.default_rng(args.seed)
        self.eval_cuts = {split:[np.array([rng.choice(self.cut_days[i]) for i in self.splits[split]])
                                 for _ in range(args.num_eval_draws)]
                          for split in ['eval_train', 'val', 'test', 'external'] if split in self.splits}
        self.args = args
        if 'train' in self.splits:
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
