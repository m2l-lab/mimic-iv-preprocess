"""Cohort statistics reporting
"""

import numpy as np
import pandas as pd

from haim_experiments.haim_features import TIMESERIES_VARIABLES


# Human-readable labels for mimic3-benchmark's pre-encoded demographic
# codes (preprocessing.py: transform_gender()'s g_map, transform_ethnicity()'s
# e_map). Several raw category strings collapse to the same code (e.g. both
# 'BLACK' and 'CARIBBEAN ISLAND' -> 2), so these labels are representative
# of the group, not a perfect 1:1 reconstruction of the original string.
GENDER_CODE_LABELS = {0: 'Unknown', 1: 'Female', 2: 'Male', 3: 'Other'}
RACE_CODE_LABELS = {0: 'Unknown/Other/Declined', 1: 'Asian', 2: 'Black', 3: 'Hispanic', 4: 'White'}
CATEGORICAL_CODE_COLUMNS = {
    'Gender': GENDER_CODE_LABELS,
    'Race': RACE_CODE_LABELS,
}


def _patient_id_from_stay(stay_filename):
    return stay_filename.split('_')[0]


def compute_cohort_stats(X, meta, n_bins=48, ts_variables=None,
                          label_data=None, label_columns=None):
    if ts_variables is None:
        ts_variables = TIMESERIES_VARIABLES

    n_stays = int(len(X))
    patients = meta['stay'].apply(_patient_id_from_stay)
    n_patients = int(patients.nunique())
    stays_per_patient = patients.value_counts()

    stats = {
        'n_stays': n_stays,
        'n_unique_patients': n_patients,
        'stays_per_patient_mean': round(n_stays / n_patients, 4) if n_patients else None,
        'n_patients_with_multiple_stays': int((stays_per_patient > 1).sum()),
    }

    # Label prevalence 
    if label_data is not None:
        if label_columns is None:
            arr = np.asarray(label_data)
            stats['label_positive_rate'] = round(float(np.mean(arr)), 4)
            stats['n_label_positive'] = int(np.sum(arr))
        else:
            rates = {col: round(float(label_data[col].mean()), 4) for col in label_columns}
            counts = {col: int(label_data[col].sum()) for col in label_columns}
            stats['label_positive_rates'] = rates
            stats['label_positive_counts'] = counts
            vals = list(rates.values())
            stats['label_positive_rate_min'] = round(min(vals), 4)
            stats['label_positive_rate_max'] = round(max(vals), 4)
            stats['label_positive_rate_mean'] = round(float(np.mean(vals)), 4)

    # Timeseries missingness (partial, complete) per variable
    ts_missingness = {}
    for var in ts_variables:
        count_col = 'ts_{}_measured_count'.format(var)
        if count_col not in X.columns:
            continue
        counts = X[count_col]
        complete_missing_rate = float((counts == 0).mean())
        fully_measured_rate = float((counts >= n_bins).mean())
        partial_missing_rate = 1.0 - complete_missing_rate - fully_measured_rate
        ts_missingness[var] = {
            'complete_missingness_rate': round(complete_missing_rate, 4),
            'partial_missingness_rate': round(max(partial_missing_rate, 0.0), 4),
            'fully_measured_rate': round(fully_measured_rate, 4),
            'mean_measured_bins_of_{}'.format(n_bins): round(float(counts.mean()), 2),
        }
    stats['timeseries_missingness'] = ts_missingness
    if ts_missingness:
        complete_rates = [v['complete_missingness_rate'] for v in ts_missingness.values()]
        stats['timeseries_complete_missingness_rate_min'] = round(min(complete_rates), 4)
        stats['timeseries_complete_missingness_rate_max'] = round(max(complete_rates), 4)
        stats['timeseries_complete_missingness_rate_mean'] = round(float(np.mean(complete_rates)), 4)

    #Tabular missingness
    tab_missingness = {}
    for col in X.columns:
        if col.startswith('tab_') and col.endswith('_was_missing'):
            base = col[len('tab_'):-len('_was_missing')]
            tab_missingness[base] = round(float(X[col].mean()), 4)
    stats['tabular_missingness'] = tab_missingness

    #Demographics
    demo = {}
    for col in X.columns:
        if not col.startswith('tab_') or col.endswith('_was_missing'):
            continue
        base_name = col[len('tab_'):]
        series = X[col]

        if base_name in CATEGORICAL_CODE_COLUMNS:
            code_labels = CATEGORICAL_CODE_COLUMNS[base_name]
            vc = series.value_counts(normalize=True, dropna=True)
            readable = {}
            raw = {}
            for k, v in vc.items():
                raw[str(k)] = round(float(v), 4)
                try:
                    label = code_labels.get(int(k), 'code_{}'.format(k))
                except (ValueError, TypeError):
                    label = str(k)
                readable[label] = round(float(v), 4)
            demo[base_name] = {'distribution_pct': readable, 'raw_code_counts_pct': raw}
        elif pd.api.types.is_numeric_dtype(series):
            demo[base_name] = {'mean': round(float(series.mean()), 2), 'std': round(float(series.std()), 2)}
        elif str(series.dtype) == 'category' or series.dtype == object:
            vc = series.value_counts(normalize=True, dropna=True)
            demo[base_name] = {str(k): round(float(v), 4) for k, v in vc.items()}
    stats['tabular_summary'] = demo

    return stats


def print_cohort_stats_summary(partition_name, stats):
    print("\n--- Cohort summary: {} ---".format(partition_name))
    print("  Stays: {}  |  Unique patients: {}  |  Stays/patient: {}  |  Patients w/ >1 stay: {}".format(
        stats['n_stays'], stats['n_unique_patients'], stats['stays_per_patient_mean'],
        stats['n_patients_with_multiple_stays']))
    if 'label_positive_rate' in stats:
        print("  Label positive rate: {:.1%} ({} / {})".format(
            stats['label_positive_rate'], stats['n_label_positive'], stats['n_stays']))
    elif 'label_positive_rate_mean' in stats:
        print("  Label positive rate across phenotypes: min={:.1%} mean={:.1%} max={:.1%}".format(
            stats['label_positive_rate_min'], stats['label_positive_rate_mean'], stats['label_positive_rate_max']))
    if 'timeseries_complete_missingness_rate_mean' in stats:
        print("  Timeseries complete-missingness rate: min={:.1%} mean={:.1%} max={:.1%}".format(
            stats['timeseries_complete_missingness_rate_min'],
            stats['timeseries_complete_missingness_rate_mean'],
            stats['timeseries_complete_missingness_rate_max']))
