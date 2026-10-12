"""Match CXR images to ICU stays within window, and assign
train/val/test split membership .

Adapted from MedFuse's loadmetadata()/load_cxr_ehr() (nyuad-cai/MedFuse)
"""

import pandas as pd


AP_VIEW = "AP"
WINDOW_HOURS = 48


def match_images_to_stays(cxr_metadata_path, all_stays_path, window_hours=WINDOW_HOURS,
                           view_position=AP_VIEW):
    """Return one row per stay_id that has >=1 qualifying CXR image,
    keeping only latest img per stay.
    """
    cxr_metadata = pd.read_csv(cxr_metadata_path)
    all_stays = pd.read_csv(all_stays_path)

    required_stay_cols = {'subject_id', 'stay_id', 'intime', 'outtime'}
    missing = required_stay_cols - set(all_stays.columns)
    if missing:
        raise ValueError("all_stays.csv is missing required column(s): {}".format(missing))

    merged = cxr_metadata.merge(
        all_stays[['subject_id', 'stay_id', 'intime', 'outtime']],
        how='inner', on='subject_id')

    merged['StudyTime'] = merged['StudyTime'].apply(lambda x: '{:06}'.format(int(float(x))))
    merged['StudyDateTime'] = pd.to_datetime(
        merged['StudyDate'].astype(str) + ' ' + merged['StudyTime'].astype(str),
        format='%Y%m%d %H%M%S')

    merged['intime'] = pd.to_datetime(merged['intime'])
    merged['outtime'] = pd.to_datetime(merged['outtime'])

    end_time = merged['intime'] + pd.Timedelta(hours=window_hours)

    in_window = merged.loc[
        (merged['StudyDateTime'] >= merged['intime']) &
        (merged['StudyDateTime'] <= end_time)
    ]

    ap_only = in_window.loc[in_window['ViewPosition'] == view_position]

    selected = []
    for stay_id, group in ap_only.groupby('stay_id'):
        latest = group.sort_values('StudyDateTime').tail(1)
        selected.append(latest)

    if not selected:
        return ap_only.iloc[0:0]  # empty df with correct columns

    result = pd.concat(selected, ignore_index=True)
    return result


def assign_matched_split(matched_images, train_listfile, val_listfile, test_listfile):
    out = {}
    for split_name, listfile in [('train', train_listfile), ('val', val_listfile), ('test', test_listfile)]:
        merged = matched_images.merge(listfile, how='inner', on='stay_id')
        merged['split'] = split_name
        out[split_name] = merged
    return out


def print_matching_summary(matched_images, split_dict, train_listfile, val_listfile, test_listfile):
    print("Stays with >=1 qualifying AP image in window: {}".format(matched_images['stay_id'].nunique()))
    yield_stats = {}
    for split_name, listfile in [('train', train_listfile), ('val', val_listfile), ('test', test_listfile)]:
        n_listfile = listfile['stay_id'].nunique()
        n_matched = split_dict[split_name]['stay_id'].nunique()
        pct = 100.0 * n_matched / n_listfile if n_listfile > 0 else 0.0
        print("  {}: {} / {} EHR stays have a matched image ({:.1f}%)".format(
            split_name, n_matched, n_listfile, pct))
        yield_stats[split_name] = {
            'n_ehr_eligible_stays': int(n_listfile),
            'n_matched_stays': int(n_matched),
            'matched_yield_pct': round(pct, 2),
        }
    return yield_stats
