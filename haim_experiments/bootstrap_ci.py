"""95% patient-level bootstrap confidence intervals for single prediction files (AUROC, AUPRC in %).
"""

from __future__ import absolute_import
from __future__ import print_function

import argparse
import os

import numpy as np
import pandas as pd

from haim_experiments.bootstrap_compare import (load_predictions, _wide, _subjects, bootstrap_weights,
                                                _metrics_per_resample, _ci)


def file_rows(path, n_boot, seed, chunk=500):
    df = load_predictions(path)
    stays = sorted(df.stay_id.unique())
    W, n_pat = bootstrap_weights(_subjects(df, stays), n_boot, np.random.default_rng(seed))
    tasks = sorted(df.task.unique())
    per_task = {}
    for task in tasks:
        y, preds = _wide(df, task, stays)
        per_task[task] = _metrics_per_resample(y, preds, W, n_boot, chunk)

    scorable = [t for t in tasks if not np.isnan(per_task[t]['auroc'][1]).any()]
    if len(tasks) > 1 and scorable:
        per_task['macro'] = {m: (float(np.mean([per_task[t][m][0] for t in scorable])),
                                 np.vstack([per_task[t][m][1] for t in scorable]).mean(axis=0))
                             for m in ('auroc', 'auprc')}

    label = os.path.basename(path).replace('_test_predictions.csv', '')
    rows = []
    for task, res in per_task.items():
        row = {'run': label, 'task': task, 'n_stays': len(stays), 'n_patients': n_pat, 'n_boot': n_boot}
        for metric in ('auroc', 'auprc'):
            point, boot = res[metric]
            lo, hi = _ci(boot)
            row.update({metric: point, metric + '_ci_lower': lo, metric + '_ci_upper': hi,
                        metric + '_text': '{:.2f} [{:.2f}, {:.2f}]'.format(100 * point, 100 * lo, 100 * hi)})
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description='Patient-level bootstrap CIs for single prediction files.')
    parser.add_argument('files', nargs='+', help='*_test_predictions.csv files')
    parser.add_argument('--n_boot', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', required=True, help='CSV with all tasks')
    args = parser.parse_args()

    rows = []
    for path in args.files:
        rows.extend(file_rows(path, args.n_boot, args.seed))
    out = pd.DataFrame(rows)
    out_dir = os.path.dirname(args.output)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out.to_csv(args.output, index=False)

    show = out[out.task.isin(['mortality', 'macro'])]
    width = show.run.str.len().max()
    print('{:{w}}  {:>24}  {:>24}  stays  patients'.format('run', 'AUROC % [95% CI]', 'AUPRC % [95% CI]', w=width))
    for r in show.itertuples():
        print('{:{w}}  {:>24}  {:>24}  {:>5}  {:>8}'.format(r.run, r.auroc_text, r.auprc_text, r.n_stays,
                                                          r.n_patients, w=width))
    print('\nSaved {}'.format(args.output))


if __name__ == '__main__':
    main()
