import argparse
import os

import numpy as np
import pandas as pd
import shap
import xgboost as xgb

from haim_experiments.bootstrap_shap import _model_paths

MIN_Z = 3.0


def load_predictions(results_dir, label, task):
    preds = pd.read_csv(os.path.join(results_dir, '{}_test_predictions.csv'.format(label)))
    preds = preds[preds.task.astype(str) == task]
    first = preds[preds.seed == preds.seed.iloc[0]]
    wide = preds.pivot_table(index='stay_id', columns='seed', values='y_pred')
    info = first.set_index('stay_id')[['subject_id', 'y_true']]
    out = info.join(pd.DataFrame({'pred': wide.mean(axis=1), 'se': wide.std(axis=1, ddof=1) / np.sqrt(wide.shape[1])}))
    return out, list(first.stay_id.values)


def load_features(results_dir, label, row_order):
    X = pd.read_csv(os.path.join(results_dir, '{}_test_features.csv'.format(label)))
    for col in X.columns:
        if X[col].dtype == object:
            X[col] = X[col].astype('category')
    if len(X) != len(row_order):
        raise ValueError('{}: {} feature rows vs {} prediction rows'.format(label, len(X), len(row_order)))
    X.index = row_order
    return X


def case_shap(results_dir, label, X, stay_ids):
    rows = X.loc[stay_ids]
    total = 0.0
    paths = _model_paths(results_dir, label)
    for path in paths:
        model = xgb.XGBClassifier()
        model.load_model(path)
        names = model.get_booster().feature_names or list(X.columns)
        sv = shap.TreeExplainer(model).shap_values(rows[names])
        if isinstance(sv, list):
            sv = sv[1]
        total = total + pd.DataFrame(sv, columns=names, index=stay_ids)[list(X.columns)]
    return total / len(paths)


def main():
    parser = argparse.ArgumentParser(description='Stays with different predictions in two runs')
    parser.add_argument('--a', nargs=2, required=True, metavar=('RESULTS_DIR', 'LABEL'))
    parser.add_argument('--b', nargs=2, required=True, metavar=('RESULTS_DIR', 'LABEL'))
    parser.add_argument('--task', default='mortality')
    parser.add_argument('--top_cases', type=int, default=20)
    parser.add_argument('--top_features', type=int, default=8)
    parser.add_argument('--output_dir', required=True)
    args = parser.parse_args()
    (dir_a, lab_a), (dir_b, lab_b) = args.a, args.b
    os.makedirs(args.output_dir, exist_ok=True)

    pa, order_a = load_predictions(dir_a, lab_a, args.task)
    pb, order_b = load_predictions(dir_b, lab_b, args.task)
    common = pa.index.intersection(pb.index)
    if len(common) != len(pa) or len(common) != len(pb):
        print('WARNING: test stays differ ({} / {} stays, {} common)'.format(len(pa), len(pb), len(common)))
    cases = pa.loc[common].join(pb.loc[common], lsuffix='_a', rsuffix='_b')
    if (cases.y_true_a != cases.y_true_b).any():
        print('WARNING: y_true differs for {} common stays'.format(int((cases.y_true_a != cases.y_true_b).sum())))
    cases = cases.rename(columns={'subject_id_a': 'subject_id', 'y_true_a': 'y_true',
                                  'pred_a': 'pred_a', 'pred_b': 'pred_b'}).drop(columns=['subject_id_b', 'y_true_b'])
    cases['diff'] = cases.pred_b - cases.pred_a
    cases['z'] = cases['diff'] / np.sqrt(cases.se_a ** 2 + cases.se_b ** 2 + 1e-12)
    cases['closer_to_truth'] = np.where((cases.y_true - cases.pred_a).abs() < (cases.y_true - cases.pred_b).abs(), 'a', 'b')
    cases.insert(0, 'stay_id', cases.index)

    candidates = cases[cases.z.abs() >= MIN_Z].copy()
    candidates = candidates.reindex(candidates['diff'].abs().sort_values(ascending=False).index)
    candidates.to_csv(os.path.join(args.output_dir, 'cases.csv'), index=False)
    print('{} of {} stays differ by >= {} standard errors of the seed-averaged prediction'.format(
        len(candidates), len(cases), MIN_Z))
    if candidates.empty:
        return

    top = candidates.head(args.top_cases)
    ids = list(top.stay_id)
    Xa, Xb = load_features(dir_a, lab_a, order_a), load_features(dir_b, lab_b, order_b)
    features = list(Xa.columns) + [f for f in Xb.columns if f not in set(Xa.columns)]
    Xa, Xb = Xa.reindex(columns=features), Xb.reindex(columns=features)
    sa = case_shap(dir_a, lab_a, load_features(dir_a, lab_a, order_a), ids).reindex(columns=features).fillna(0.0)
    sb = case_shap(dir_b, lab_b, load_features(dir_b, lab_b, order_b), ids).reindex(columns=features).fillna(0.0)

    rows = []
    for stay in ids:
        delta = (sb.loc[stay] - sa.loc[stay])
        for rank, feat in enumerate(delta.abs().sort_values(ascending=False).index[:args.top_features], 1):
            va, vb = Xa.loc[stay, feat], Xb.loc[stay, feat]
            rows.append({'stay_id': stay, 'rank': rank, 'feature': feat, 'value_a': va, 'value_b': vb,
                         'value_differs': not (pd.isna(va) and pd.isna(vb)) and va != vb,
                         'shap_a': sa.loc[stay, feat], 'shap_b': sb.loc[stay, feat], 'delta_shap': delta[feat]})
    details = pd.DataFrame(rows)
    details.to_csv(os.path.join(args.output_dir, 'case_details.csv'), index=False)

    all_delta = (sb - sa)
    summary = pd.DataFrame({
        'n_cases_in_top3': details[details['rank'] <= 3].groupby('feature').size(),
        'n_cases_value_differs': details[details['rank'] <= 3].groupby('feature').value_differs.sum(),
        'mean_abs_delta_shap': all_delta.abs().mean(),
    }).fillna(0).sort_values(['n_cases_in_top3', 'mean_abs_delta_shap'], ascending=False)
    summary = summary[summary.n_cases_in_top3 > 0]
    summary.index.name = 'feature'
    summary.to_csv(os.path.join(args.output_dir, 'driver_summary.csv'))

    print('\nTop drivers over the {} most different stays (a={}, b={}):'.format(len(ids), lab_a, lab_b))
    print(summary.head(10).round(4).to_string())
    for stay in ids[:3]:
        c = cases.loc[stay]
        print('\nstay {}  y_true={}  pred_a={:.3f}  pred_b={:.3f}  z={:.1f}'.format(
            stay, c.y_true, c.pred_a, c.pred_b, c.z))
        print(details[details.stay_id == stay][['feature', 'value_a', 'value_b', 'shap_a', 'shap_b', 'delta_shap']]
              .round(3).to_string(index=False))
    print('\nSaved to {}'.format(args.output_dir))


if __name__ == '__main__':
    main()
