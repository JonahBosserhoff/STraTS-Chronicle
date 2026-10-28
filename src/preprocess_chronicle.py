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
    df['os_end_day'] = df['time_since_initial_diagnosis']+df['OS_TIME']
    oc.append(df.groupby('patient_id')[['os_end_day', 'OS_EVENT']].first())
    df = df.loc[df.feature_name.isin(features)]
    data.append(df[['patient_id', 'time_since_initial_diagnosis', 'feature_name', 'feature_value']])
    split_ids[split] = df.patient_id.unique()

data = pd.concat(data).rename(columns={'patient_id':'ts_id', 'time_since_initial_diagnosis':'day',
                                       'feature_name':'variable', 'feature_value':'value'})
oc = pd.concat(oc).reset_index().rename(columns={'patient_id':'ts_id', 'OS_EVENT':'os_event'})
print('# observations, patients, variables:', len(data), data.ts_id.nunique(), data.variable.nunique())

os.makedirs(os.path.dirname(args.out), exist_ok=True)
pickle.dump([data, oc, split_ids['train'], split_ids['val'], split_ids['test']], open(args.out, 'wb'))
print('Saved', args.out)
