"""Convert a preprocessed Chronicle cohort into the STraTS pickle format.

Input is the output of `chronicle-preprocess`: <prefix>_{train,val,test}.csv in one
directory (already split by patient and scaled with scalers fitted on train). Reading
these files, instead of re-splitting, keeps the train/val/test patients identical to the
Chronicle run.

As in Chronicle, only features listed in the global-format schema are kept.

Output: ../data/processed/chronicle.pkl = [data, oc, train_ids, val_ids, test_ids]
    data: ts_id, day, variable, value     one row per observation (ts_id = patient_id)
    oc:   ts_id, day, <target columns>    one row per patient and observed day, with every
                                          target of the cohort: <T>_TIME / <T>_EVENT for
                                          survival targets (OS, TTNT, ...) and the label_*
                                          columns. The target to train on is chosen in
                                          main.py (--target), so one preprocessing serves all.
The targets of a day are taken from its first row, as Chronicle's <CLS> token of that day
copies the labels of the day's first observation (they are constant within a day).
"""
import argparse
import os
import pickle
from pathlib import Path

import pandas as pd


def target_columns(columns):
    """Target name -> its columns in a Chronicle CSV: survival targets as the pair
    <T>_TIME / <T>_EVENT (as Chronicle finds them), all other targets as their label_* column."""
    columns = list(columns)
    targets = {c[:-len('_TIME')]:[c, c[:-len('_TIME')]+'_EVENT'] for c in columns
               if c.endswith('_TIME') and c[:-len('_TIME')]+'_EVENT' in columns}
    survival_cols = {c for cols in targets.values() for c in cols}
    targets.update({c:[c] for c in columns if c.startswith('label') and c not in survival_cols})
    return targets


def to_strats(df):
    """Chronicle long-format frame -> (data, oc) as above, all features kept."""
    targets = target_columns(df.columns)
    assert len(targets)>0, 'No targets (<T>_TIME / <T>_EVENT or label_* columns) in the data'
    label_cols = [c for cols in targets.values() for c in cols]
    oc = df.drop_duplicates(['patient_id', 'time_since_initial_diagnosis'], keep='first') \
           [['patient_id', 'time_since_initial_diagnosis']+label_cols] \
           .rename(columns={'patient_id':'ts_id', 'time_since_initial_diagnosis':'day'}) \
           .reset_index(drop=True)
    data = df[['patient_id', 'time_since_initial_diagnosis', 'feature_name', 'feature_value']] \
             .rename(columns={'patient_id':'ts_id', 'time_since_initial_diagnosis':'day',
                              'feature_name':'variable', 'feature_value':'value'})
    return data, oc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True,
                        help='directory with <prefix>_{train,val,test}.csv from chronicle-preprocess')
    parser.add_argument('--global_format', type=str, required=True,
                        help='Chronicle schema CSV (feature vocabulary)')
    parser.add_argument('--out', type=str, default='../data/processed/chronicle.pkl')
    args = parser.parse_args()

    schema = pd.read_csv(args.global_format)
    features = set(schema.loc[schema['type']!='special_token', 'feature_name'])

    data, oc, split_ids = [], [], {}
    for split in ['train', 'val', 'test']:
        paths = sorted(Path(args.data_dir).glob('*_'+split+'.csv'))
        assert len(paths)==1, 'Expected one *_'+split+'.csv in '+args.data_dir+', found '+str(paths)
        df = pd.read_csv(paths[0])
        print('Read', paths[0], len(df), 'rows,', df.patient_id.nunique(), 'patients')
        # schema features only, as Chronicle (also for the day -> targets table)
        df = df.loc[df.feature_name.isin(features)]
        split_data, split_oc = to_strats(df)
        data.append(split_data)
        oc.append(split_oc)
        split_ids[split] = split_data.ts_id.unique()

    data, oc = pd.concat(data), pd.concat(oc, ignore_index=True)
    print('# observations, patients, variables:', len(data), data.ts_id.nunique(), data.variable.nunique())
    print('Targets (choose one with main.py --target):', list(target_columns(oc.columns)))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    pickle.dump([data, oc, split_ids['train'], split_ids['val'], split_ids['test']], open(args.out, 'wb'))
    print('Saved', args.out)
