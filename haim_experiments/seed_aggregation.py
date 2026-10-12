"""multi-seed training runs
"""

import numpy as np
from scipy import stats as scipy_stats


def mean_std_ci(values, confidence=0.95):
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    mean = float(np.mean(arr))
    if n <= 1:
        return {'mean': mean, 'std': 0.0, 'n': n, 'ci_lower': mean, 'ci_upper': mean, 'confidence': confidence}

    std = float(np.std(arr, ddof=1))  # sample std
    sem = std / np.sqrt(n)
    t_crit = float(scipy_stats.t.ppf((1 + confidence) / 2, df=n - 1))
    half_width = t_crit * sem
    return {
        'mean': mean, 'std': std, 'n': n,
        'ci_lower': mean - half_width, 'ci_upper': mean + half_width,
        'confidence': confidence,
    }


def aggregate_feature_importance(per_seed_importance_dfs, confidence=0.95):
    import pandas as pd

    all_features = per_seed_importance_dfs[0]['feature'].tolist()
    modality_lookup = dict(zip(per_seed_importance_dfs[0]['feature'], per_seed_importance_dfs[0]['modality']))

    per_feature_values = {f: [] for f in all_features}
    for df in per_seed_importance_dfs:
        value_lookup = dict(zip(df['feature'], df['mean_abs_shap']))
        for f in all_features:
            per_feature_values[f].append(value_lookup.get(f, 0.0))

    rows = []
    for f in all_features:
        stats = mean_std_ci(per_feature_values[f], confidence=confidence)
        rows.append({
            'feature': f,
            'modality': modality_lookup.get(f, 'other'),
            'mean_abs_shap': stats['mean'],
            'mean_abs_shap_std': stats['std'],
            'mean_abs_shap_ci_lower': max(stats['ci_lower'], 0.0),
            'mean_abs_shap_ci_upper': stats['ci_upper'],
            'n_seeds': stats['n'],
        })
    out = pd.DataFrame(rows).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)
    return out


def aggregate_modality_importance(per_seed_modality_series, confidence=0.95):
    import pandas as pd

    all_modalities = per_seed_modality_series[0].index.tolist()
    rows = []
    for m in all_modalities:
        values = [s.get(m, 0.0) for s in per_seed_modality_series]
        stats = mean_std_ci(values, confidence=confidence)
        rows.append({
            'modality': m, 'mean_pct': stats['mean'], 'std_pct': stats['std'],
            'ci_lower': max(stats['ci_lower'], 0.0), 'ci_upper': stats['ci_upper'], 'n_seeds': stats['n'],
        })
    return pd.DataFrame(rows).sort_values('mean_pct', ascending=False).reset_index(drop=True)
