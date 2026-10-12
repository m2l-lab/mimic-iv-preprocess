from __future__ import absolute_import
from __future__ import print_function

import argparse
import os
import pandas as pd 
import sys
from tqdm import tqdm
import traceback

from mimic3benchmark.subject import read_stays, read_diagnoses, read_events, get_events_for_stay,\
    add_hours_elpased_to_events, read_events_s, get_events_for_stay_s, add_hours_elpased_to_events_s
from mimic3benchmark.subject import convert_events_to_timeseries, convert_events_to_timeseries_s, get_first_valid_from_timeseries
from mimic3benchmark.preprocessing import read_itemid_to_variable_map, map_itemids_to_variables, clean_events
from mimic3benchmark.preprocessing import assemble_episodic_data


parser = argparse.ArgumentParser(description='Extract episodes from per-subject data.')
parser.add_argument('subjects_root_path', type=str, help='Directory containing subject sub-directories.')
parser.add_argument('--variable_map_file', type=str,
                    default=os.path.join(os.path.dirname(__file__), '../resources/itemid_to_variable_map.csv'),
                    help='CSV containing ITEMID-to-variable map.')
parser.add_argument('--reference_range_file', type=str,
                    default=os.path.join(os.path.dirname(__file__), '../resources/variable_ranges.csv'),
                    help='CSV containing reference ranges for variables.')
args, _ = parser.parse_known_args()

var_map = read_itemid_to_variable_map(args.variable_map_file)
variables = var_map.variable.unique()

for subject_dir in tqdm(os.listdir(args.subjects_root_path), desc='Iterating over subjects'):
    dn = os.path.join(args.subjects_root_path, subject_dir)
    try:
        subject_id = int(subject_dir)
        if not os.path.isdir(dn):
            raise Exception
    except:
        continue

    try:
        # reading tables of this subject
        stays = read_stays(os.path.join(args.subjects_root_path, subject_dir))
        
        diagnoses = read_diagnoses(os.path.join(args.subjects_root_path, subject_dir))
        events = read_events(os.path.join(args.subjects_root_path, subject_dir))
        events_s = read_events_s(os.path.join(args.subjects_root_path, subject_dir))

    except:
        sys.stderr.write('Error reading from disk for subject: {}\n'.format(subject_id))
        traceback.print_exc()
        continue
    
    #add overall mortality
    stays['dod'] = pd.to_datetime(stays.dod)
    stays['intime'] = pd.to_datetime(stays.intime)
    stays['mortality_1yr'] = (
            stays.dod.notnull() & (stays.intime <= stays.dod) & (stays.dod <= stays.intime + pd.Timedelta(days=365))).astype(int)
    
    episodic_data = assemble_episodic_data(stays, diagnoses)

    # cleaning and converting to time series

    events = map_itemids_to_variables(events, var_map)

    events = clean_events(events)

    events_s = map_itemids_to_variables(events_s, var_map)

    events_s = clean_events(events_s)

    if events.shape[0] == 0:
        # no valid events for this subject
        continue
    # else:
    #     print(f'events for subject_id {events.shape[0]}')

    timeseries = convert_events_to_timeseries(events, variables=variables)

    # extracting separate episodes
    # import pdb; pdb.set_trace()      

    for i in range(stays.shape[0]):
        stay_id = stays.stay_id.iloc[i]
        intime = stays.intime.iloc[i]
        outtime = stays.outtime.iloc[i]

        episode = get_events_for_stay(timeseries, stay_id, intime, outtime)
        if episode.shape[0] == 0:
            # no data for this episode
            continue

        episode = add_hours_elpased_to_events(episode, intime).set_index('HOURS').sort_index(axis=0)
        if stay_id in episodic_data.index:
            episodic_data.loc[stay_id, 'Weight'] = get_first_valid_from_timeseries(episode, 'Weight')
            episodic_data.loc[stay_id, 'Height'] = get_first_valid_from_timeseries(episode, 'Height')
        episodic_data.loc[episodic_data.index == stay_id].to_csv(os.path.join(args.subjects_root_path, subject_dir,
                                                                              'episode{}.csv'.format(i+1)),
                                                                 index_label='Icustay')
        episode = episode.drop(columns=['Weight', 'Height'], errors='ignore')
        columns = list(episode.columns)
        columns_sorted = sorted(columns, key=(lambda x: "" if x == "Hours" else x))
        episode = episode[columns_sorted]
        episode.to_csv(os.path.join(args.subjects_root_path, subject_dir, 'episode{}_timeseries.csv'.format(i+1)),
                       index_label='Hours')
        
    if events_s.shape[0] == 0:
        # no valid events for this subject
        continue
    # else:
    #     print(f'events for subject_id {events.shape[0]}')

    timeseries_s = convert_events_to_timeseries_s(events_s, variables=variables)

    # extracting separate episodes
    # import pdb; pdb.set_trace()      

    for i in range(stays.shape[0]):
        stay_id = stays.stay_id.iloc[i]
        intime = stays.intime.iloc[i]
        outtime = stays.outtime.iloc[i]

        episode_s = get_events_for_stay_s(timeseries_s, stay_id, intime, outtime)
        if episode_s.shape[0] == 0:
            # no data for this episode
            continue

        episode_s = add_hours_elpased_to_events_s(episode_s, intime).set_index('HOURS').sort_index(axis=0)
        if stay_id in episodic_data.index:
            episodic_data.loc[stay_id, 'Weight'] = get_first_valid_from_timeseries(episode, 'Weight')
            episodic_data.loc[stay_id, 'Height'] = get_first_valid_from_timeseries(episode, 'Height')
        episodic_data.loc[episodic_data.index == stay_id].to_csv(os.path.join(args.subjects_root_path, subject_dir,
                                                                              'episode{}_s.csv'.format(i+1)),
                                                                 index_label='Icustay')
        columns = list(episode_s.columns)
        columns_sorted = sorted(columns, key=(lambda x: "" if x == "Hours" else x))
        episode_s = episode_s[columns_sorted]
        episode_s.to_csv(os.path.join(args.subjects_root_path, subject_dir, 'episode{}_timeseries_s.csv'.format(i+1)),
                       index_label='Hours')

