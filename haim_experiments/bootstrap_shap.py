"""Patient-clustered bootstrap CIs for SHAP-based importance
"""

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
import shap
import xgboost as xgb

from haim_experiments.bootstrap_compare import bootstrap_weights, _ci

IMAGE_SHARE = 'image_share_pct'


def _safe_name(phenotype):
    return re.sub(r'[^A-Za-z0-9]+', '_', phenotype).strip('_')


def _model_paths(results_dir, prefix):
    pattern = re.compile(r'{}_seed\d+_model\.json'.format(re.escape(prefix)))
    paths = sorted(p for p in glob.glob(os.path.join(results_dir, '{}_seed*_model.json'.format(prefix)))
                   if pattern.fullmatch(os.path.basename(p)))
    if paths:
        return paths
    single = os.path.join(results_dir, '{}_model.json'.format(prefix))
    if os.path.exists(single):
        return [single]
    raise FileNotFoundError('No model files for {} in {}'.format(prefix, results_dir))


def _abs_shap_per_seed(results_dir, prefix):
    X = pd.read_csv(os.path.join(results_dir, '{}_test_features.csv'.format(prefix)))
    for col in X.columns:
        if X[col].dtype == object:
            X[col] = X[col].astype('category')
    out = []
    for path in _model_paths(results_dir, prefix):
        model = xgb.XGBClassifier()
        model.load_model(path)
        names = model.get_booster().feature_names or list(X.columns)
        sv = shap.TreeExplainer(model).shap_values(X[names])
        if isinstance(sv, list):
            sv = sv[1]
        out.append(pd.DataFrame(np.abs(sv), columns=names)[list(X.columns)].values)
    return list(X.columns), out


def _subjects(results_dir, label, task, n_rows):
    preds = pd.read_csv(os.path.join(results_dir, '{}_test_predictions.csv'.format(label)))
    preds = preds[preds.task.astype(str) == task]
    rows = preds[preds.seed == preds.seed.iloc[0]]
    if len(rows) != n_rows:
        raise ValueError('{} / {}: {} prediction rows vs {} feature rows'.format(label, task, len(rows), n_rows))
    return rows.subject_id.astype(str).values


def _row(label, task, feature, point, boot, n_boot, n_seeds, subjects):
    lo, hi = _ci(boot)
    return {'run': label, 'task': task, 'feature': feature, 'point': point, 'ci_lower': lo, 'ci_upper': hi,
            'n_boot': n_boot, 'n_seeds': n_seeds, 'n_stays': len(subjects), 'n_patients': len(set(subjects))}


def _warn_if_differs(what, recomputed, reported, tol):
    dev = float(np.nanmax(np.abs(np.asarray(recomputed) - np.asarray(reported))))
    if dev > tol:
        print('  WARNING {}: recomputed value differs from saved results by up to {:.3g}'.format(what, dev))


def mortality_rows(results_dir, label, n_boot, seed):
    features, abs_shap = _abs_shap_per_seed(results_dir, label)
    subjects = _subjects(results_dir, label, 'mortality', abs_shap[0].shape[0])
    W, _ = bootstrap_weights(subjects, n_boot, np.random.default_rng(seed))
    norm = W.sum(axis=1, keepdims=True)

    point = np.mean([a.mean(axis=0) for a in abs_shap], axis=0)
    boot = np.mean([(W @ a) / norm for a in abs_shap], axis=0)

    reported = pd.read_csv(os.path.join(results_dir, '{}_feature_importance.csv'.format(label)))
    reported = reported.set_index('feature')['mean_abs_shap'].reindex(features).values
    _warn_if_differs(label, point, reported, 1e-6 * max(1.0, float(point.max())))

    return [_row(label, 'mortality', f, point[j], boot[:, j], n_boot, len(abs_shap), subjects)
            for j, f in enumerate(features)]


def phenotype_rows(results_dir, label, n_boot, seed):
    preds = pd.read_csv(os.path.join(results_dir, '{}_test_predictions.csv'.format(label)), usecols=['task'])
    rows = []
    for phenotype in sorted(preds.task.astype(str).unique()):
        prefix = '{}_{}'.format(label, _safe_name(phenotype))
        features, abs_shap = _abs_shap_per_seed(results_dir, prefix)
        img = np.array([f.startswith('img_') for f in features])
        if not img.any():
            raise ValueError('{}: no img_ features in {}_test_features.csv'.format(label, prefix))
        subjects = _subjects(results_dir, label, phenotype, abs_shap[0].shape[0])
        W, _ = bootstrap_weights(subjects, n_boot, np.random.default_rng(seed))

        point, boot = [], []
        for a in abs_shap:
            img_sum, total = a[:, img].sum(axis=1), a.sum(axis=1)
            point.append(100 * img_sum.sum() / total.sum())
            boot.append(100 * (W @ img_sum) / (W @ total))
        point, boot = float(np.mean(point)), np.mean(boot, axis=0)

        detailed = os.path.join(results_dir, '{}_modality_importance_detailed.csv'.format(prefix))
        if os.path.exists(detailed):
            mdf = pd.read_csv(detailed)
            img_row = mdf[mdf.modality == 'image']
            if len(img_row):
                _warn_if_differs(prefix, [point], [img_row.mean_pct.iloc[0]], 0.05)

        rows.append(_row(label, phenotype, IMAGE_SHARE, point, boot, n_boot, len(abs_shap), subjects))
        print('  {}: image share {:.1f}% [{:.1f}, {:.1f}]'.format(phenotype, point, rows[-1]['ci_lower'],
                                                                  rows[-1]['ci_upper']))
    return rows


def main():
    parser = argparse.ArgumentParser(description='Patient-clustered bootstrap CIs for SHAP importance.')
    parser.add_argument('task', choices=['mortality', 'phenotype'])
    parser.add_argument('--run', nargs=2, action='append', required=True, metavar=('RESULTS_DIR', 'LABEL'))
    parser.add_argument('--n_boot', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    compute = mortality_rows if args.task == 'mortality' else phenotype_rows
    rows = []
    for results_dir, label in args.run:
        print('Bootstrapping SHAP for {} ...'.format(label))
        rows += compute(results_dir, label, args.n_boot, args.seed)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print('Saved {} rows to {}'.format(len(rows), args.output))


if __name__ == '__main__':
    main()
