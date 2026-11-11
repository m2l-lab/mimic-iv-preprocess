"""Adds microbiology features (from extract_microbiology_features.py) for one anchor as micro_* columns."""

import pandas as pd


MICRO_ANCHORS = ('charttime', 'storetime')


def add_micro_features(X, meta, micro_csv, anchor):
    micro = pd.read_csv(micro_csv).set_index('stay_id')
    cols = [c for c in micro.columns if c.startswith(anchor + '_')]
    feats = micro[cols].reindex(meta['stay_id'].values).fillna(0)
    feats.columns = ['micro_' + c[len(anchor) + 1:] for c in cols]
    return pd.concat([X.reset_index(drop=True), feats.reset_index(drop=True)], axis=1)
