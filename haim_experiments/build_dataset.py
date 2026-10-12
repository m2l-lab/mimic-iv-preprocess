import os

import pandas as pd

from haim_experiments.haim_features import extract_ts_features, extract_tabular_features, compute_train_tabular_stats


def _patient_and_episode_filename(stay_filename):
    patient = stay_filename.split('_')[0]
    rest = stay_filename[len(patient) + 1:]           # 'episode1_timeseries[_s].csv'
    episode_filename = rest.replace('_timeseries', '')  # 'episode1[_s].csv'
    return patient, episode_filename


def filter_first_stay_per_patient(listfile_df): # if needed 
    df = listfile_df.copy()
    patient_ids = df['stay'].apply(lambda s: _patient_and_episode_filename(s)[0])

    def _episode_num(stay):
        episode_filename = _patient_and_episode_filename(stay)[1]  # 'episode3[_s].csv'
        digits = ''.join(ch for ch in episode_filename.split('_')[0] if ch.isdigit())
        return int(digits) if digits else 0

    episode_nums = df['stay'].apply(_episode_num)
    df = df.assign(_patient=patient_ids, _episode_num=episode_nums)
    df = df.sort_values('_episode_num').drop_duplicates(subset='_patient', keep='first')
    df = df.drop(columns=['_patient', '_episode_num']).sort_index().reset_index(drop=True)
    return df


def episode_paths_from_listfile(listfile_df, root_physical_dir):
    paths = []
    for stay in listfile_df['stay']:
        patient, episode_filename = _patient_and_episode_filename(stay)
        paths.append(os.path.join(root_physical_dir, patient, episode_filename))
    return paths


def build_feature_matrix(listfile_df, binned_physical_dir, root_physical_dir,
                          train_stats, fallback_mode="clinical", label_col="y_true",
                          include_measured_count=True, tabular_train_stats=None,
                          include_tabular_missing_indicator=True):

    rows = []
    labels = []
    meta_rows = []

    for _, r in listfile_df.iterrows():
        stay = r['stay']
        binned_csv = os.path.join(binned_physical_dir, stay)
        if not os.path.exists(binned_csv):
            raise FileNotFoundError(
                "Binned timeseries file not found: {}. Check that "
                "binned_physical_dir matches this listfile's partition.".format(binned_csv))

        ts_feats = extract_ts_features(binned_csv, train_stats, fallback_mode=fallback_mode,
                                        include_measured_count=include_measured_count)
        ts_feats = {"ts_{}".format(k): v for k, v in ts_feats.items()}

        patient, episode_filename = _patient_and_episode_filename(stay)
        episode_csv = os.path.join(root_physical_dir, patient, episode_filename)
        if not os.path.exists(episode_csv):
            raise FileNotFoundError(
                "Episode file not found: {}. Check root_physical_dir.".format(episode_csv))


        tab_feats = extract_tabular_features(episode_csv, train_stats=tabular_train_stats,
                                              include_missing_indicator=include_tabular_missing_indicator)
        tab_feats = {"tab_{}".format(k): v for k, v in tab_feats.items()}

        row = {}
        row.update(ts_feats)
        row.update(tab_feats)
        rows.append(row)
        labels.append(r[label_col])
        meta_rows.append({'stay': stay, 'stay_id': r.get('stay_id', None)})

    X = pd.DataFrame(rows)
    y = pd.Series(labels, name=label_col).values
    meta = pd.DataFrame(meta_rows)

    # Convert non-numeric tabular columns 
    for col in X.columns:
        if str(X[col].dtype) == 'category':
            continue
        if X[col].dtype == object or pd.api.types.is_string_dtype(X[col]):
            X[col] = X[col].astype('category')

    return X, y, meta


def align_columns(X_train, X_other):
    return X_other.reindex(columns=X_train.columns)
