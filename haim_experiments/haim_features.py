"""HAIM-style feature extraction 
"""

import glob
import os

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


CATEGORICAL_VARIABLES = [
    "Glascow coma scale eye opening",
    "Glascow coma scale motor response",
    "Glascow coma scale verbal response",
]


_GCS_EYE_MAP = {
    "0 none": 0, "none": 0,
    "1 no response": 1, "no response": 1,
    "2 to pain": 2, "to pain": 2,
    "3 to speech": 3, "to speech": 3,
    "4 spontaneously": 4, "spontaneously": 4,
}
_GCS_MOTOR_MAP = {
    "1 no response": 1, "no response": 1,
    "2 abnorm extensn": 2, "abnormal extension": 2,
    "3 abnorm flexion": 3, "abnormal flexion": 3,
    "4 flex-withdraws": 4, "flex-withdraws": 4,
    "5 localizes pain": 5, "localizes pain": 5,
    "6 obeys commands": 6, "obeys commands": 6,
}
_GCS_VERBAL_MAP = {
    "1 no response": 1, "no response": 1,
    "no response-ett": 1, "1.0 et/trach": 1,  
    "2 incomp sounds": 2, "incomprehensible sounds": 2,
    "3 inapprop words": 3, "inappropriate words": 3,
    "4 confused": 4, "confused": 4,
    "5 oriented": 5, "oriented": 5,
}

CATEGORICAL_STRING_MAPS = {
    "Glascow coma scale eye opening": _GCS_EYE_MAP,
    "Glascow coma scale motor response": _GCS_MOTOR_MAP,
    "Glascow coma scale verbal response": _GCS_VERBAL_MAP,
}


def _coerce_variable_to_numeric(raw_series, var_name):
    if var_name in CATEGORICAL_STRING_MAPS:
        mapping = CATEGORICAL_STRING_MAPS[var_name]
        normalized = raw_series.astype(str).str.strip().str.lower()
        mapped = normalized.map(mapping)
        still_missing = mapped.isna() & raw_series.notna()
        if still_missing.any():
            mapped.loc[still_missing] = pd.to_numeric(
                raw_series.loc[still_missing], errors="coerce")
        return mapped
    return pd.to_numeric(raw_series, errors="coerce")


TIMESERIES_VARIABLES = [
    "Bicarbonate", "Bilirubin", "Blood urea nitrogen",
    "Chloride", "Creatinine", "Diastolic blood pressure", "Fraction inspired oxygen",
    "Glascow coma scale eye opening", "Glascow coma scale motor response",
    "Glascow coma scale verbal response", "Glucose",
    "Heart Rate", "Mean blood pressure", "Oxygen saturation",
    "Potassium", "Respiratory rate", "Systolic blood pressure", "Temperature",
    "pH",
]


TRAIN_MEAN_ONLY_VARIABLES = set()


FALLBACK_VALUES = {
    "Bicarbonate": 27.0,
    "Bilirubin": 0.7,
    "Blood urea nitrogen": 14.0,
    "Chloride": 100.0,
    "Creatinine": 1.0,
    "Diastolic blood pressure": 80.0,
    "Fraction inspired oxygen": 0.21,     # room air
    "Glascow coma scale eye opening": 4,  # categorical: spontaneous 
    "Glascow coma scale motor response": 6,  # categorical: obeys commands 
    "Glascow coma scale verbal response": 5,  # categorical: oriented
    "Glucose": 85.0,
    "Heart Rate": 80.0,
    "Mean blood pressure": 90.0,
    "Oxygen saturation": 99.0,
    "Potassium": 4.0,
    "Respiratory rate": 15.0,
    "Systolic blood pressure": 120.0,
    "Temperature": 36.6,
    "pH": 7.4
}


def compute_train_variable_stats(binned_dir, stay_filenames=None, variables=None,
                                  categorical_vars=None):
    if variables is None:
        variables = TIMESERIES_VARIABLES
    if categorical_vars is None:
        categorical_vars = CATEGORICAL_VARIABLES

    all_values = {v: [] for v in variables}

    if stay_filenames is not None:
        csv_files = [os.path.join(binned_dir, f) for f in stay_filenames]
        missing = [f for f in csv_files if not os.path.exists(f)]
        if missing:
            raise FileNotFoundError(
                "{} stay file(s) from the provided list were not found in {}, "
                "e.g. {}".format(len(missing), binned_dir, missing[:3]))
    else:
        csv_files = glob.glob(os.path.join(binned_dir, "*.csv"))
        csv_files = [f for f in csv_files if os.path.basename(f) != "listfile.csv"]

    for path in csv_files:
        df = pd.read_csv(path, keep_default_na=False, na_values=[''])
        for v in variables:
            if v not in df.columns:
                continue
            vals = _coerce_variable_to_numeric(df[v], v).dropna()
            if len(vals) > 0:
                all_values[v].extend(vals.tolist())

    stats = {}
    for v in variables:
        if len(all_values[v]) > 0:
            s = pd.Series(all_values[v])
            stats[v] = {'mean': float(s.mean()), 'mode': float(s.mode().iloc[0])}
        else:
            stats[v] = {'mean': np.nan, 'mode': np.nan}
    return stats


def _compute_haim_stats(dense_values):
    vals = np.asarray(dense_values, dtype=float)
    n = len(vals)
    s = pd.Series(vals)

    stats = {
        "max": float(np.max(vals)),
        "min": float(np.min(vals)),
        "mean": float(np.mean(vals)),
        "variance": float(s.var(skipna=True)) if n > 1 else 0.0,  # ddof=1, matches pandas default
    }

    diffs = s.diff().dropna()
    if len(diffs) > 0:
        stats["meandiff"] = float(diffs.mean())
        stats["meanabsdiff"] = float(diffs.abs().mean())
        stats["maxdiff"] = float(diffs.abs().max())
        stats["sumabsdiff"] = float(diffs.abs().sum())
    else:
        stats["meandiff"] = 0.0
        stats["meanabsdiff"] = 0.0
        stats["maxdiff"] = 0.0
        stats["sumabsdiff"] = 0.0

    stats["diff"] = float(vals[-1] - vals[0])

    peaks, _ = find_peaks(vals)
    stats["npeaks"] = float(len(peaks))

    if n > 1:
        stats["trend"] = float(np.polyfit(np.arange(n), vals, 1)[0])
    else:
        stats["trend"] = 0.0

    return stats


def extract_ts_features(binned_csv_path, train_stats, variables=None,
                         categorical_vars=None, fallback_mode="clinical",
                         include_measured_count=True):
    if variables is None:
        variables = TIMESERIES_VARIABLES
    if categorical_vars is None:
        categorical_vars = CATEGORICAL_VARIABLES
    if fallback_mode not in ("clinical", "train_mean"):
        raise ValueError("fallback_mode must be 'clinical' or 'train_mean'")

    df = pd.read_csv(binned_csv_path, keep_default_na=False, na_values=[''])
    n_bins = len(df)

    features = {}
    for var in variables:
        if var in df.columns:
            raw = _coerce_variable_to_numeric(df[var], var)
        else:
            raw = pd.Series([np.nan] * n_bins)

        measured_count = int(raw.notnull().sum())

        if measured_count > 0:
            dense = raw.ffill().bfill()
        else:
            var_stats = train_stats.get(var, {})
            if var in TRAIN_MEAN_ONLY_VARIABLES or fallback_mode == "train_mean":
                if var in categorical_vars:
                    fill_value = var_stats.get('mode', np.nan)
                else:
                    fill_value = var_stats.get('mean', np.nan)
            else:
                fill_value = FALLBACK_VALUES.get(var, var_stats.get('mean', np.nan))
            dense = pd.Series([fill_value] * n_bins)

        stats = _compute_haim_stats(dense.values)
        if include_measured_count:
            stats["measured_count"] = float(measured_count)

        for stat_name, stat_val in stats.items():
            features["{}_{}".format(var, stat_name)] = stat_val

    return features



TABULAR_EXCLUDE_PREFIXES = ("Diagnosis ",)
TABULAR_EXCLUDE_COLUMNS = {
    "Icustay", "Mortality", "Mortality_1yr", "Length of Stay",
}


TABULAR_UNKNOWN_CATEGORY_OVERRIDE = {
    "Gender": 0,  
    "Race": 0,    
}


def compute_train_tabular_stats(root_train_phys, episode_paths, numeric_agg="median"):
    if numeric_agg not in ("median", "mean"):
        raise ValueError("numeric_agg must be 'median' or 'mean'")

    frames = []
    for path in episode_paths:
        df = pd.read_csv(path)
        if len(df) == 0:
            continue
        row = df.iloc[[0]]
        keep_cols = [c for c in row.columns if c not in TABULAR_EXCLUDE_COLUMNS
                     and not any(c.startswith(p) for p in TABULAR_EXCLUDE_PREFIXES)]
        frames.append(row[keep_cols])

    if not frames:
        return {}
    all_rows = pd.concat(frames, ignore_index=True)

    stats = {}
    for col in all_rows.columns:
        if col in TABULAR_UNKNOWN_CATEGORY_OVERRIDE:
            stats[col] = TABULAR_UNKNOWN_CATEGORY_OVERRIDE[col]
            continue
        series = all_rows[col].dropna()
        if len(series) == 0:
            continue
        if pd.api.types.is_numeric_dtype(series):
            stats[col] = float(series.median()) if numeric_agg == "median" else float(series.mean())
        else:
            stats[col] = series.mode().iloc[0]
    return stats


def extract_tabular_features(episode_csv_path, train_stats=None, include_missing_indicator=True):
    df = pd.read_csv(episode_csv_path)
    if len(df) == 0:
        return {}
    row = df.iloc[0]

    features = {}
    for col in df.columns:
        if col in TABULAR_EXCLUDE_COLUMNS:
            continue
        if any(col.startswith(p) for p in TABULAR_EXCLUDE_PREFIXES):
            continue
        val = row[col]
        was_missing = bool(pd.isnull(val))
        if was_missing and train_stats is not None and col in train_stats:
            val = train_stats[col]
        features[col] = val
        if include_missing_indicator:
            features["{}_was_missing".format(col)] = int(was_missing)

    return features
