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
from mimic3benchmark.timestamp_families import load_itemid_families, itemids_with_timestamp, print_family_table


parser = argparse.ArgumentParser(description='Extract episodes from per-subject data.')
parser.add_argument('subjects_root_path', type=str, help='Directory containing subject sub-directories.')
parser.add_argument('--variable_map_file', type=str,
                    default=os.path.join(os.path.dirname(__file__), '../resources/itemid_to_variable_map.csv'),
                    help='CSV containing ITEMID-to-variable map.')
parser.add_argument('--reference_range_file', type=str,
                    default=os.path.join(os.path.dirname(__file__), '../resources/variable_ranges.csv'),
                    help='CSV containing reference ranges for variables.')
parser.add_argument('--variants', type=str, nargs='+', default=['charttime', 'storetime'],
                    choices=['charttime', 'storetime', 'hybrid'],
                    help="Which episode files to write: 'charttime' (episode{i}_timeseries.csv), "
                         "'storetime' (..._s.csv) and/or 'hybrid' (..._h.csv: timestamp chosen per "
                         "feature family, see mimic3benchmark/timestamp_families.py). "
                         "Default: charttime storetime.")
args, _ = parser.parse_known_args()

# Height/Weight (tabular features) may only come from the prediction window
# of the tasks (first 48 h of the ICU stay, as in create_in_hospital_mortality /
# create_phenotyping); later values would be look-ahead leakage.
PREDICTION_WINDOW_HOURS = 48

var_map = read_itemid_to_variable_map(args.variable_map_file)
variables = var_map.variable.unique()

if 'hybrid' in args.variants:
    families = load_itemid_families(args.variable_map_file)
    storetime_itemids = itemids_with_timestamp(families, 'storetime')
    print_family_table(families)


def build_hybrid_events(events, events_s, storetime_itemids):
    """Combine charttime and storetime events per itemid: itemids of a
    storetime family come from events_s (storetime renamed to charttime so
    the charttime helpers can be reused), all others from events."""
    from_chart = events[~events.itemid.isin(storetime_itemids)]
    from_store = events_s[events_s.itemid.isin(storetime_itemids)].rename(columns={'storetime': 'charttime'})
    # rows without a storetime cannot be placed on the time axis
    from_store = from_store[from_store.charttime.notnull()]
    return pd.concat([from_chart, from_store], ignore_index=True, sort=False)


def write_hybrid_episodes(events_h, stays, episodic_data, subject_path):
    timeseries_h = convert_events_to_timeseries(events_h, variables=variables)
    for i in range(stays.shape[0]):
        stay_id = stays.stay_id.iloc[i]
        intime = stays.intime.iloc[i]
        outtime = stays.outtime.iloc[i]

        episode_h = get_events_for_stay(timeseries_h, stay_id, intime, outtime)
        if episode_h.shape[0] == 0:
            continue

        episode_h = add_hours_elpased_to_events(episode_h, intime).set_index('HOURS').sort_index(axis=0)
        episode_data = episodic_data.loc[episodic_data.index == stay_id].copy()
        if stay_id in episode_data.index:
            # Height/Weight are bedside items, so they are charttime-anchored here
            episode_data.loc[stay_id, 'Weight'] = get_first_valid_from_timeseries(episode_h, 'Weight', max_hours=PREDICTION_WINDOW_HOURS)
            episode_data.loc[stay_id, 'Height'] = get_first_valid_from_timeseries(episode_h, 'Height', max_hours=PREDICTION_WINDOW_HOURS)
        episode_data.to_csv(os.path.join(subject_path, 'episode{}_h.csv'.format(i+1)), index_label='Icustay')
        episode_h = episode_h.drop(columns=['Weight', 'Height'], errors='ignore')
        columns_sorted = sorted(episode_h.columns, key=(lambda x: "" if x == "Hours" else x))
        episode_h[columns_sorted].to_csv(os.path.join(subject_path, 'episode{}_timeseries_h.csv'.format(i+1)),
                                         index_label='Hours')

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

    if 'hybrid' in args.variants:
        events_h = build_hybrid_events(events, events_s, storetime_itemids)
        if events_h.shape[0] > 0:
            write_hybrid_episodes(events_h, stays, episodic_data, dn)

    if events.shape[0] == 0:
        # no valid events for this subject
        continue
    # else:
    #     print(f'events for subject_id {events.shape[0]}')

    if 'charttime' in args.variants:
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
                episodic_data.loc[stay_id, 'Weight'] = get_first_valid_from_timeseries(episode, 'Weight', max_hours=PREDICTION_WINDOW_HOURS)
                episodic_data.loc[stay_id, 'Height'] = get_first_valid_from_timeseries(episode, 'Height', max_hours=PREDICTION_WINDOW_HOURS)
            episodic_data.loc[episodic_data.index == stay_id].to_csv(os.path.join(args.subjects_root_path, subject_dir,
                                                                                  'episode{}.csv'.format(i+1)),
                                                                     index_label='Icustay')
            episode = episode.drop(columns=['Weight', 'Height'], errors='ignore')
            columns = list(episode.columns)
            columns_sorted = sorted(columns, key=(lambda x: "" if x == "Hours" else x))
            episode = episode[columns_sorted]
            episode.to_csv(os.path.join(args.subjects_root_path, subject_dir, 'episode{}_timeseries.csv'.format(i+1)),
                           index_label='Hours')
        
    if 'storetime' not in args.variants or events_s.shape[0] == 0:
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
            # taken from the storetime episode of THIS stay (previously read from
            # `episode`, the last charttime episode with Height/Weight already
            # dropped, so these were always NaN)
            episodic_data.loc[stay_id, 'Weight'] = get_first_valid_from_timeseries(episode_s, 'Weight', max_hours=PREDICTION_WINDOW_HOURS)
            episodic_data.loc[stay_id, 'Height'] = get_first_valid_from_timeseries(episode_s, 'Height', max_hours=PREDICTION_WINDOW_HOURS)
        episodic_data.loc[episodic_data.index == stay_id].to_csv(os.path.join(args.subjects_root_path, subject_dir,
                                                                              'episode{}_s.csv'.format(i+1)),
                                                                 index_label='Icustay')
        # Height/Weight are tabular features only, not timeseries
        episode_s = episode_s.drop(columns=['Weight', 'Height'], errors='ignore')
        columns = list(episode_s.columns)
        columns_sorted = sorted(columns, key=(lambda x: "" if x == "Hours" else x))
        episode_s = episode_s[columns_sorted]
        episode_s.to_csv(os.path.join(args.subjects_root_path, subject_dir, 'episode{}_timeseries_s.csv'.format(i+1)),
                       index_label='Hours')

