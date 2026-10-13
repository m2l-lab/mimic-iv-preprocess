import os

import numpy as np
import pandas as pd

from haim_experiments.haim_features import extract_ts_features, extract_tabular_features
from haim_experiments.build_dataset import _patient_and_episode_filename



LISTFILE_NON_LABEL_COLUMNS = {'stay', 'period_length', 'stay_id'}


def infer_label_columns(listfile_df):
    return [c for c in listfile_df.columns if c not in LISTFILE_NON_LABEL_COLUMNS]


def build_phenotyping_feature_matrix(matched_df, binned_physical_dir, root_physical_dir,
                                      image_embeddings_df, train_stats, tabular_train_stats,
                                      fallback_mode="clinical", include_measured_count=True,
                                      include_tabular_missing_indicator=True, label_columns=None):
    if label_columns is None:
        raise ValueError(
            "label_columns must be given explicitly. Auto-inferring from matched_df is unsafe.")

    img_lookup = None
    if image_embeddings_df is not None:
        img_lookup = image_embeddings_df.set_index('stay_id')
        img_lookup = img_lookup[~img_lookup.index.duplicated(keep='first')]

    rows = []
    label_rows = []
    meta_rows = []
    n_missing_embedding = 0

    for _, r in matched_df.iterrows():
        stay = r['stay']
        binned_csv = os.path.join(binned_physical_dir, stay)
        if not os.path.exists(binned_csv):
            raise FileNotFoundError(
                "Binned timeseries file not found: {}. Check that "
                "binned_physical_dir matches this partition.".format(binned_csv))

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

        img_feats = {}
        stay_id = r['stay_id']
        if img_lookup is not None and stay_id in img_lookup.index:
            img_row = img_lookup.loc[stay_id]
            img_feats = dict(img_row.items())
        else:
            n_missing_embedding += 1

        row = {}
        row.update(ts_feats)
        row.update(tab_feats)
        row.update(img_feats)
        rows.append(row)
        label_rows.append([int(r[c]) if pd.notnull(r[c]) else 0 for c in label_columns])
        meta_rows.append({'stay': stay, 'stay_id': stay_id})

    if image_embeddings_df is not None and n_missing_embedding > 0:
        print("Warning: {} / {} matched stays had no precomputed image embedding "
              "(image features left as NaN for those rows).".format(n_missing_embedding, len(matched_df)))

    X = pd.DataFrame(rows)
    Y = pd.DataFrame(label_rows, columns=label_columns)
    meta = pd.DataFrame(meta_rows)

    for col in X.columns:
        if str(X[col].dtype) == 'category':
            continue
        if X[col].dtype == object or pd.api.types.is_string_dtype(X[col]):
            X[col] = X[col].astype('category')

    return X, Y, meta, label_columns
