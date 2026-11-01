"""Convert a preprocessed Chronicle cohort into the STraTS pickle format.

Input is the output of `chronicle-preprocess`: <prefix>_{train,val,test}.csv in one
directory (already split by patient and scaled with scalers fitted on train). Reading
these files, instead of re-splitting, keeps the train/val/test patients identical to the
Chronicle run.

As in Chronicle, only features listed in the global-format schema are kept.

Output: ../data/processed/chronicle.pkl = [data, oc, train_ids, val_ids, test_ids]
    data: ts_id, day, variable, value     one row per observation (ts_id = patient_id)
    oc:   ts_id, os_end_day, os_event     day + OS_TIME is constant per patient, so the
                                          OS label at any cut day is os_end_day - day
"""
import argparse
import os
import pickle
from pathlib import Path

import pandas as pd


def to_strats(df):
    """Chronicle long-format frame -> (data, oc) as above, all features kept."""
    df = df.assign(os_end_day=df['time_since_initial_diagnosis']+df['OS_TIME'])
    oc = df.groupby('patient_id')[['os_end_day', 'OS_EVENT']].first().reset_index() \
           .rename(columns={'patient_id':'ts_id', 'OS_EVENT':'os_event'})
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
        split_data, split_oc = to_strats(df)
        split_data = split_data.loc[split_data.variable.isin(features)]
        data.append(split_data)
        oc.append(split_oc)
        split_ids[split] = split_data.ts_id.unique()

    data, oc = pd.concat(data), pd.concat(oc)
    print('# observations, patients, variables:', len(data), data.ts_id.nunique(), data.variable.nunique())

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    pickle.dump([data, oc, split_ids['train'], split_ids['val'], split_ids['test']], open(args.out, 'wb'))
    print('Saved', args.out)
