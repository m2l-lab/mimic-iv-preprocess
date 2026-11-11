"""
  shift_feature_pct     half the L1 distance between normalised mean |SHAP| profiles over features
                        = % of total importance that sits on different features (0 = identical)
  shift_variable_pct    same, after summing the statistics of one variable (mean/min/trend/...)
  overlap_feature_pct   % of the baseline's top-k features that are also top-k in the run
  overlap_variable_pct  same on the variable level
  cosine_patient        mean over test patients of the cosine between the two models' SHAP vectors

"""

import argparse
import os

import numpy as np
import pandas as pd
import shap
import xgboost as xgb

from haim_experiments.bootstrap_compare import bootstrap_weights, _ci
from haim_experiments.bootstrap_shap import _model_paths

METRICS = ['shift_feature_pct', 'shift_variable_pct', 'overlap_feature_pct', 'overlap_variable_pct',
           'cosine_patient']
TIMESTAMPS = {'hybrid': 'hybrid', 'store': 'storetime', 'chart': 'charttime'}
IMPUTATION = {'static': 'static value', 'statistical': 'statistical'}


def variable_group(feature):
    if feature.startswith('ts_'):
        name = feature[3:]
        if name.endswith('_measured_count'):
            return name[:-len('_measured_count')] + ' [count]'
        for suffix in ('_ett_count', '_ett_flag'):
            if name.endswith(suffix):
                return name[:-len(suffix)] + ' [ETT]'
        return name.rpartition('_')[0] or name
    if feature.startswith('tab_'):
        name = feature[4:]
        if name.endswith('_was_missing'):
            return name[:-len('_was_missing')] + ' [missing]'
        return name
    return feature


def load_run(results_dir, label, task='mortality'):
    X = pd.read_csv(os.path.join(results_dir, '{}_test_features.csv'.format(label)))
    for col in X.columns:
        if X[col].dtype == object:
            X[col] = X[col].astype('category')
    preds = pd.read_csv(os.path.join(results_dir, '{}_test_predictions.csv'.format(label)))
    preds = preds[preds.task.astype(str) == task]
    rows = preds[preds.seed == preds.seed.iloc[0]]
    if len(rows) != len(X):
        raise ValueError('{}: {} prediction rows vs {} feature rows'.format(label, len(rows), len(X)))

    shap_values, mean_abs = [], []
    for path in _model_paths(results_dir, label):
        model = xgb.XGBClassifier()
        model.load_model(path)
        names = model.get_booster().feature_names or list(X.columns)
        sv = shap.TreeExplainer(model).shap_values(X[names])
        if isinstance(sv, list):
            sv = sv[1]
        sv = pd.DataFrame(sv, columns=names)[list(X.columns)].values
        mean_abs.append(np.abs(sv).mean(axis=0))
        shap_values.append(sv.astype(np.float32))

    reported = pd.read_csv(os.path.join(results_dir, '{}_feature_importance.csv'.format(label)))
    reported = reported.set_index('feature')['mean_abs_shap'].reindex(X.columns).values
    point = np.mean(mean_abs, axis=0)
    if np.nanmax(np.abs(point - reported)) > 1e-5 * max(1.0, float(point.max())):
        print('  WARNING {}: recomputed mean |SHAP| differs from feature_importance.csv by up to {:.2e}'.format(
            label, np.nanmax(np.abs(point - reported))))

    return {'label': label, 'features': list(X.columns), 'stay_id': rows.stay_id.values,
            'subject_id': rows.subject_id.astype(str).values, 'shap': shap_values}


def project(S, features, space, rows):
    index = {f: i for i, f in enumerate(features)}
    out = np.zeros((len(rows), len(space)), dtype=np.float32)
    src = [(j, index[f]) for j, f in enumerate(space) if f in index]
    out[:, [j for j, _ in src]] = S[rows][:, [i for _, i in src]]
    return out


def group_matrix(space):
    groups = [variable_group(f) for f in space]
    names = list(dict.fromkeys(groups))
    G = np.zeros((len(space), len(names)))
    for j, g in enumerate(groups):
        G[j, names.index(g)] = 1.0
    return G


def top_mask(values, k):
    k = min(k, values.shape[1])
    idx = np.argpartition(-values, k - 1, axis=1)[:, :k]
    mask = np.zeros(values.shape, dtype=bool)
    np.put_along_axis(mask, idx, True, axis=1)
    return mask, k


def model_stats(S, Wn, G, k):
    imp = (Wn @ np.abs(S)).astype(np.float64)
    var = imp @ G
    top_f, k_f = top_mask(imp, k)
    top_v, k_v = top_mask(var, k)
    return {'S': S, 'norm': np.linalg.norm(S, axis=1), 'imp': imp, 'var': var,
            'top_f': top_f, 'k_f': k_f, 'top_v': top_v, 'k_v': k_v}


def pair_metrics(a, b, Wn):
    share = lambda x: x / x.sum(axis=1, keepdims=True)
    cos = np.einsum('ij,ij->i', a['S'], b['S']) / (a['norm'] * b['norm'] + 1e-12)
    return {
        'shift_feature_pct': 50 * np.abs(share(a['imp']) - share(b['imp'])).sum(axis=1),
        'shift_variable_pct': 50 * np.abs(share(a['var']) - share(b['var'])).sum(axis=1),
        'overlap_feature_pct': 100 * (a['top_f'] & b['top_f']).sum(axis=1) / a['k_f'],
        'overlap_variable_pct': 100 * (a['top_v'] & b['top_v']).sum(axis=1) / a['k_v'],
        'cosine_patient': Wn @ cos.astype(np.float64),
    }


def mean_over_pairs(models_a, models_b, Wn, within):
    total, n = {m: 0.0 for m in METRICS}, 0
    for i, a in enumerate(models_a):
        for j, b in enumerate(models_b):
            if within and j <= i:
                continue
            for m, v in pair_metrics(a, b, Wn).items():
                total[m] = total[m] + v
            n += 1
    return {m: v / n for m, v in total.items()}, n


def summarize(values, meta):
    row = dict(meta)
    for m in METRICS:
        lo, hi = _ci(values[m][1:])
        row[m], row[m + '_ci_lower'], row[m + '_ci_upper'] = float(values[m][0]), lo, hi
    return row


def weights(subjects, n_boot, seed):
    W, n_pat = bootstrap_weights(subjects, n_boot, np.random.default_rng(seed))
    W = np.vstack([np.ones((1, W.shape[1])), W]).astype(np.float32)
    return W / W.sum(axis=1, keepdims=True), n_pat


def display_name(label):
    parts = label.split('_')
    if len(parts) >= 2 and parts[0] in TIMESTAMPS and parts[1] in IMPUTATION:
        return '{}, {}{}'.format(TIMESTAMPS[parts[0]], IMPUTATION[parts[1]],
                                 ' + missing' if parts[2:] == ['miss'] else '')
    return label.replace('_', r'\_')


def latex_table(rows, space):
    wanted = [r for r in rows if r['run'] == 'seed_reference' and space in r['feature_space'].split('/')]
    wanted = wanted or [r for r in rows if r['run'] == 'seed_reference'][:1]
    by_run = {}
    for r in rows:
        if r['run'] == 'seed_reference':
            continue
        if r['feature_space'] == space or (r['feature_space'] == 'same' and r['run'] not in by_run):
            by_run[r['run']] = r
    lines = [r'\begin{table}', r'\centering',
             r'\caption{Change of the feature importance relative to the baseline.}',
             r'\label{tbl-explanation-shift}', r'\resizebox{\linewidth}{!}{',
             r'\begin{tabular}{lccccc}', r'\toprule',
             r' & Importance shift, features (\%) & Importance shift, variables (\%) & '
             r'Top-$k$ overlap, features (\%) & Top-$k$ overlap, variables (\%) & Patient-level cosine \\',
             r'\midrule']
    ordered = [('Seed reference (baseline)', wanted[0])] if wanted else []
    ordered += [(display_name(run), r) for run, r in by_run.items()]
    for i, (name, r) in enumerate(ordered):
        cells = []
        for m in METRICS:
            digits = 3 if m == 'cosine_patient' else 1
            cells.append('{:.{d}f} [{:.{d}f}, {:.{d}f}]'.format(r[m], r[m + '_ci_lower'], r[m + '_ci_upper'], d=digits))
        lines.append(('\\rowcolor{gray!8}' if i % 2 == 0 else '') + name + ' & ' + ' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'}', r'\end{table}']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description='Change of SHAP-based model logic relative to a baseline run.')
    parser.add_argument('--baseline', nargs=2, required=True, metavar=('RESULTS_DIR', 'LABEL'))
    parser.add_argument('--run', nargs=2, action='append', required=True, metavar=('RESULTS_DIR', 'LABEL'))
    parser.add_argument('--top_k', type=int, default=10)
    parser.add_argument('--n_boot', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--task', default='mortality')
    parser.add_argument('--output', required=True)
    parser.add_argument('--latex', default=None)
    parser.add_argument('--latex_space', choices=['all', 'shared'], default='all')
    args = parser.parse_args()


    base = load_run(args.baseline[0], args.baseline[1], args.task)
    base_order = np.argsort(base['stay_id'])
    references, rows = {}, []

    def compute(space, run=None):
        if run is None:
            stay_pos_b = base_order
            subjects = base['subject_id'][stay_pos_b]
        else:
            common = sorted(set(base['stay_id']) & set(run['stay_id']))
            if not common:
                raise ValueError('No common stays between {} and {}'.format(base['label'], run['label']))
            stay_pos_b = pd.Series(np.arange(len(base['stay_id'])), index=base['stay_id']).reindex(common).values
            stay_pos_r = pd.Series(np.arange(len(run['stay_id'])), index=run['stay_id']).reindex(common).values
            subjects = base['subject_id'][stay_pos_b]
            if len(common) != len(set(base['stay_id'])) or len(common) != len(set(run['stay_id'])):
                print('  WARNING {}: test stays differ from baseline ({} common)'.format(run['label'], len(common)))
        Wn, n_pat = weights(subjects, args.n_boot, args.seed)
        G = group_matrix(space)
        stats_b = [model_stats(project(S, base['features'], space, stay_pos_b), Wn, G, args.top_k)
                   for S in base['shap']]
        if run is None:
            values, n_pairs = mean_over_pairs(stats_b, stats_b, Wn, within=True)
        else:
            stats_r = [model_stats(project(S, run['features'], space, stay_pos_r), Wn, G, args.top_k)
                       for S in run['shap']]
            values, n_pairs = mean_over_pairs(stats_b, stats_r, Wn, within=False)
        return values, {'n_features_space': len(space), 'n_boot': args.n_boot, 'n_seed_pairs': n_pairs,
                        'n_stays': len(subjects), 'n_patients': n_pat}

    def ensure_reference(space, space_label, run_label):
        key = tuple(space)
        if key not in references:
            values, meta = compute(space)
            references[key] = {'values': values, 'meta': meta, 'labels': set(), 'runs': []}
        references[key]['labels'].add(space_label)
        references[key]['runs'].append(run_label)

    ensure_reference(base['features'], 'same', args.baseline[1])
    variant_rows = []
    for results_dir, label in args.run:
        print('Computing SHAP for {} ...'.format(label))
        run = load_run(results_dir, label, args.task)
        if set(run['features']) == set(base['features']):
            spaces = [('same', list(base['features']))]
        else:
            union = list(base['features']) + [f for f in run['features'] if f not in set(base['features'])]
            shared = [f for f in base['features'] if f in set(run['features'])]
            spaces = [('all', union), ('shared', shared)]
        for space_label, space in spaces:
            values, meta = compute(space, run)
            meta.update({'run': label, 'baseline': base['label'], 'feature_space': space_label,
                         'n_features_baseline': len(base['features']), 'n_features_run': len(run['features']),
                         'n_features_shared': len(set(base['features']) & set(run['features']))})
            variant_rows.append(summarize(values, meta))
            ensure_reference(space, space_label, label)
        del run

    for key, ref in references.items():
        meta = dict(ref['meta'])
        meta.update({'run': 'seed_reference', 'baseline': base['label'],
                     'feature_space': '/'.join(sorted(ref['labels'])), 'reference_for': ','.join(ref['runs']),
                     'n_features_baseline': len(base['features'])})
        rows.append(summarize(ref['values'], meta))
    rows += variant_rows

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    df.to_csv(args.output, index=False)
    print('Saved {} rows to {}'.format(len(df), args.output))
    show = ['run', 'feature_space'] + METRICS
    print(df[show].round(3).to_string(index=False))
    if args.latex:
        with open(args.latex, 'w') as f:
            f.write(latex_table(rows, args.latex_space) + '\n')
        print('Saved table to {}'.format(args.latex))


if __name__ == '__main__':
    main()
