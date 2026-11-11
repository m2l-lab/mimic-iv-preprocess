"""Patient-level bootstrap of test-set AUROC/AUPRC, paired or unpaired (models are not retrained).
Usage examples are in the README."""

from __future__ import absolute_import
from __future__ import print_function

import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy import sparse


PRED_COLUMNS = ['stay', 'stay_id', 'subject_id', 'task', 'seed', 'y_true', 'y_pred']


def test_predictions_long(meta_test, task, y_test, per_seed_results):
    stays = meta_test['stay'].astype(str).values
    subject_ids = [s.split('_')[0] for s in stays]
    frames = []
    for r in per_seed_results:
        frames.append(pd.DataFrame({
            'stay': stays,
            'stay_id': meta_test['stay_id'].values,
            'subject_id': subject_ids,
            'task': task,
            'seed': r['seed'],
            'y_true': np.asarray(y_test).astype(int),
            'y_pred': np.asarray(r['test_pred'], dtype=float),
        }))
    return pd.concat(frames, ignore_index=True)[PRED_COLUMNS]


class _RankedScores(object):
    """Weighted AUROC/AP for many bootstrap weight vectors at once (equal to sklearn on duplicated data)."""

    def __init__(self, y_true, y_score):
        y_true = np.asarray(y_true).astype(int)
        y_score = np.asarray(y_score, dtype=float)
        # tie groups, sorted by descending score
        uniq, inv = np.unique(-y_score, return_inverse=True)
        n, g = len(y_score), len(uniq)
        pos = (y_true == 1).astype(float)
        neg = 1.0 - pos
        rows = np.arange(n)
        self.G_pos = sparse.csr_matrix((pos, (rows, inv)), shape=(n, g))
        self.G_neg = sparse.csr_matrix((neg, (rows, inv)), shape=(n, g))

    def metrics(self, W):
        wp = np.asarray(W @ self.G_pos)
        wn = np.asarray(W @ self.G_neg)
        P = wp.sum(axis=1)
        N = wn.sum(axis=1)
        valid = (P > 0) & (N > 0)

        # AUROC = P(pos > neg) + 0.5 P(tie)
        cum_n = np.cumsum(wn, axis=1)
        neg_below = N[:, None] - cum_n
        auc_num = (wp * (neg_below + 0.5 * wn)).sum(axis=1)

        # AP as in sklearn: sum of (recall_k - recall_k-1) * precision_k
        tp = np.cumsum(wp, axis=1)
        fp = cum_n
        with np.errstate(invalid='ignore', divide='ignore'):
            precision = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
            auroc = auc_num / (P * N)
            auprc = (wp * precision).sum(axis=1) / P
        auroc[~valid] = np.nan
        auprc[~valid] = np.nan
        return auroc, auprc


def load_predictions(path):
    df = pd.read_csv(path)
    missing = [c for c in PRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError('{} is missing columns {}'.format(path, missing))
    df['subject_id'] = df['subject_id'].astype(str)
    df['task'] = df['task'].astype(str)
    return df


def _wide(df, task, stay_ids):
    sub = df[df.task == task]
    piv = sub.pivot_table(index='stay_id', columns='seed', values='y_pred', aggfunc='first').reindex(stay_ids)
    y = sub.drop_duplicates('stay_id').set_index('stay_id')['y_true'].reindex(stay_ids).values
    return y.astype(int), piv.values.T


def bootstrap_weights(subject_of_stay, n_boot, rng):
    patients, patient_idx = np.unique(subject_of_stay, return_inverse=True)
    n_pat = len(patients)
    draws = rng.integers(0, n_pat, size=(n_boot, n_pat))
    counts = np.zeros((n_boot, n_pat), dtype=np.int32)
    np.add.at(counts, (np.repeat(np.arange(n_boot), n_pat), draws.ravel()), 1)
    return counts[:, patient_idx].astype(float), n_pat


def _seed_mean_metrics(y, preds, W):
    full = np.ones((1, len(y)))
    pt_auc, pt_ap, bs_auc, bs_ap = [], [], [], []
    for p in preds:
        ranked = _RankedScores(y, p)
        a, b = ranked.metrics(full)
        pt_auc.append(a[0]); pt_ap.append(b[0])
        a, b = ranked.metrics(W)
        bs_auc.append(a); bs_ap.append(b)
    return (np.nanmean(pt_auc), np.nanmean(pt_ap),
            np.mean(bs_auc, axis=0), np.mean(bs_ap, axis=0))


def _ci(x):
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan
    return float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))


def _p_two_sided(d):
    d = d[~np.isnan(d)]
    if len(d) == 0:
        return np.nan
    return float(min(1.0, 2 * min(np.mean(d <= 0), np.mean(d >= 0))))


def _fmt_p(p, n_boot):
    if p != p:
        return 'nan'
    return '<{:.4f}'.format(1.0 / n_boot) if p == 0 else '{:.4f}'.format(p)


def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    if not ok.any():
        return out
    idx = np.where(ok)[0]
    order = idx[np.argsort(p[idx])]
    m = len(order)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        out[i] = running
    return out


def _metrics_per_resample(y, preds, W, n_boot, chunk):
    auc_parts, ap_parts = [], []
    for start in range(0, n_boot, chunk):
        p_auc, p_ap, b_auc, b_ap = _seed_mean_metrics(y, preds, W[start:start + chunk])
        auc_parts.append(b_auc); ap_parts.append(b_ap)
    return {'auroc': (p_auc, np.concatenate(auc_parts)), 'auprc': (p_ap, np.concatenate(ap_parts))}


def _subjects(df, stay_ids):
    return df.drop_duplicates('stay_id').set_index('stay_id')['subject_id'].reindex(stay_ids).values


def compare_pair(path_a, path_b, n_boot=2000, seed=0, chunk=500, groups=None, unpaired=False):
    a = load_predictions(path_a)
    b = load_predictions(path_b)

    common = sorted(set(a.stay_id) & set(b.stay_id))
    if not common and not unpaired:
        raise ValueError('No common stay_id between {} and {} (use --unpaired)'.format(path_a, path_b))
    n_drop_a = a.stay_id.nunique() - len(common)
    n_drop_b = b.stay_id.nunique() - len(common)

    tasks = sorted(set(a.task) & set(b.task))
    only = sorted(set(a.task) ^ set(b.task))
    if only:
        print('  tasks only in one file (ignored): {}'.format(only))

    rng = np.random.default_rng(seed)
    if unpaired:
        # different test sets (e.g. stay-level split): each file resampled over its own patients
        stays = {'A': sorted(set(a.stay_id)), 'B': sorted(set(b.stay_id))}
        W = {'A': bootstrap_weights(_subjects(a, stays['A']), n_boot, rng)[0],
             'B': bootstrap_weights(_subjects(b, stays['B']), n_boot, rng)[0]}
        n_pat = a.subject_id.nunique()
    else:
        stays = {'A': common, 'B': common}
        W_common, n_pat = bootstrap_weights(_subjects(a, common), n_boot, rng)
        W = {'A': W_common, 'B': W_common}

    per_task = {}
    for task in tasks:
        ya, pa = _wide(a, task, stays['A'])
        yb, pb = _wide(b, task, stays['B'])
        if not unpaired and not np.array_equal(ya, yb):
            raise ValueError("Labels differ between files for task '{}' on common stays".format(task))
        per_task[task] = {'A': _metrics_per_resample(ya, pa, W['A'], n_boot, chunk),
                          'B': _metrics_per_resample(yb, pb, W['B'], n_boot, chunk)}

    # macro/group averages only over tasks scorable in every resample
    scorable = [t for t in tasks
                if not any(np.isnan(per_task[t][k]['auroc'][1]).any() for k in ('A', 'B'))]
    excluded = sorted(set(tasks) - set(scorable))
    if len(tasks) > 1 and excluded:
        print('  excluded from macro/group averages (no positives or negatives in some resamples): {}'.format(
            excluded))

    def _macro(task_list):
        out = {}
        for key in ('A', 'B'):
            out[key] = {}
            for metric in ('auroc', 'auprc'):
                pts = [per_task[t][key][metric][0] for t in task_list]
                bss = np.vstack([per_task[t][key][metric][1] for t in task_list])
                out[key][metric] = (float(np.mean(pts)), bss.mean(axis=0))
        return out

    n_tasks = {t: 1 for t in tasks}
    if len(tasks) > 1 and scorable:
        per_task['macro'] = _macro(scorable)
        n_tasks['macro'] = len(scorable)

    group_names = []
    for name, members in (groups or {}).items():
        unknown = sorted(set(members) - set(tasks))
        if unknown:
            raise ValueError("Group '{}': phenotypes not found in both prediction files: {}\n"
                             "Available: {}".format(name, unknown, tasks))
        used = [t for t in members if t in scorable]
        if not used:
            print("  group '{}' has no phenotype scorable in every resample, skipped".format(name))
            continue
        per_task['group:' + name] = _macro(used)
        n_tasks['group:' + name] = len(used)
        group_names.append(name)

    rows = []

    def _row(task, metric, pa_, ba, pb_, bb, n):
        d = ba - bb
        return {
            'a': path_a, 'b': path_b, 'task': task, 'metric': metric,
            'a_point': pa_, 'a_ci_lower': _ci(ba)[0], 'a_ci_upper': _ci(ba)[1],
            'b_point': pb_, 'b_ci_lower': _ci(bb)[0], 'b_ci_upper': _ci(bb)[1],
            'diff_point': pa_ - pb_, 'diff_ci_lower': _ci(d)[0], 'diff_ci_upper': _ci(d)[1],
            'p_value': _p_two_sided(d),
            'n_boot': n_boot, 'n_invalid_resamples': int(np.isnan(d).sum()), 'n_tasks': n,
            'n_common_stays': len(common), 'n_patients': n_pat,
            'n_stays_only_in_a': n_drop_a, 'n_stays_only_in_b': n_drop_b, 'paired': not unpaired,
        }

    for task, res in per_task.items():
        for metric in ('auroc', 'auprc'):
            pa_, ba = res['A'][metric]
            pb_, bb = res['B'][metric]
            rows.append(_row(task, metric, pa_, ba, pb_, bb, n_tasks[task]))

    # group contrasts: a_* = (A - B) in g1, b_* = (A - B) in g2
    for i, g1 in enumerate(group_names):
        for g2 in group_names[i + 1:]:
            r1, r2 = per_task['group:' + g1], per_task['group:' + g2]
            for metric in ('auroc', 'auprc'):
                d1 = (r1['A'][metric][0] - r1['B'][metric][0], r1['A'][metric][1] - r1['B'][metric][1])
                d2 = (r2['A'][metric][0] - r2['B'][metric][0], r2['A'][metric][1] - r2['B'][metric][1])
                rows.append(_row('contrast:{} - {}'.format(g1, g2), metric, d1[0], d1[1], d2[0], d2[1],
                                 n_tasks['group:' + g1] + n_tasks['group:' + g2]))
    return rows


def main():
    parser = argparse.ArgumentParser(description='Paired patient-level bootstrap of test-set AUROC/AUPRC.')
    parser.add_argument('--pair', nargs=2, action='append', required=True, metavar=('A', 'B'),
                        help='Two *_test_predictions.csv files to compare (A - B). Repeatable.')
    parser.add_argument('--unpaired', action='store_true', default=False,
                        help='Bootstrap each file over its own test patients (for different test sets, '
                             'e.g. the stay-level split).')
    parser.add_argument('--n_boot', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=0, help='RNG seed of the bootstrap.')
    parser.add_argument('--output', type=str, required=True, help='CSV with one row per pair/task/metric.')
    parser.add_argument('--groups', type=str, default=None,
                        help='JSON file {group_name: [phenotype, ...]} (phenotyping only). Adds the macro '
                             'average per group and the contrast of the A - B gain between groups, '
                             'e.g. cxr_phenotype_groups.json.')
    args = parser.parse_args()

    groups = None
    if args.groups:
        with open(args.groups) as f:
            groups = {k: v for k, v in json.load(f).items() if not k.startswith('_')}

    rows = []
    for path_a, path_b in args.pair:
        print('Comparing\n  A: {}\n  B: {}'.format(path_a, path_b))
        rows.extend(compare_pair(path_a, path_b, n_boot=args.n_boot, seed=args.seed, groups=groups,
                                 unpaired=args.unpaired))
    out = pd.DataFrame(rows)
    out['p_holm'] = np.nan
    for _, idx in out.groupby(['task', 'metric']).groups.items():
        out.loc[idx, 'p_holm'] = holm(out.loc[idx, 'p_value'].values)

    out_dir = os.path.dirname(args.output)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out.to_csv(args.output, index=False)

    summary_mask = out.task.isin(['mortality', 'macro']) | out.task.str.startswith(('group:', 'contrast:'))
    show = out[summary_mask] if summary_mask.any() else out
    for _, r in show.iterrows():
        print('{} | {} {}: A={:.4f} [{:.4f}, {:.4f}]  B={:.4f} [{:.4f}, {:.4f}]  '
              'A-B={:+.4f} [{:+.4f}, {:+.4f}]  p={} (Holm {})  stays={} patients={}'.format(
                  os.path.basename(r.a) + ' vs ' + os.path.basename(r.b), r.task, r.metric.upper(),
                  r.a_point, r.a_ci_lower, r.a_ci_upper, r.b_point, r.b_ci_lower, r.b_ci_upper,
                  r.diff_point, r.diff_ci_lower, r.diff_ci_upper,
                  _fmt_p(r.p_value, r.n_boot), _fmt_p(r.p_holm, r.n_boot),
                  r.n_common_stays, r.n_patients))
    print('\nSaved {}'.format(args.output))


if __name__ == '__main__':
    main()
