"""One test stay in two runs (e.g. charttime vs storetime): SHAP values of the features that differ most
and the predicted risk. Example:

python -m haim_experiments.plot_case \
    --a results/mortality_chart chart_static\
    --b results/mortality_store store_static\
    --stay_id 00000 --output_path results/case_comparison.pdf --top_n 6
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb

from haim_experiments.bootstrap_shap import _model_paths
from haim_experiments.plot_results import VARIANT_PALETTE, _pretty_feature_name
from haim_experiments.seed_aggregation import mean_std_ci
from haim_experiments.timestamp_cases import load_features, load_predictions


def short_name(feature):
    name = _pretty_feature_name(feature)
    for prefix in ('ts_', 'tab_', 'micro_', 'icd_', 'img_'):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def shap_per_seed(results_dir, label, X, stay_id):
    row = X.loc[[stay_id]]
    out = []
    for path in _model_paths(results_dir, label):
        model = xgb.XGBClassifier()
        model.load_model(path)
        names = model.get_booster().feature_names or list(X.columns)
        sv = shap.TreeExplainer(model).shap_values(row[names])
        if isinstance(sv, list):
            sv = sv[1]
        out.append(pd.Series(np.asarray(sv)[0], index=names).reindex(X.columns))
    return pd.DataFrame(out).reset_index(drop=True)


def predictions_per_seed(results_dir, label, task, stay_id):
    preds = pd.read_csv(os.path.join(results_dir, '{}_test_predictions.csv'.format(label)))
    preds = preds[(preds.task.astype(str) == task) & (preds.stay_id == stay_id)]
    return preds.sort_values('seed').y_pred.values


def interval(values):
    s = mean_std_ci(values)
    return s['mean'], s['ci_lower'], s['ci_upper']


def plot_case(shap_a, shap_b, pred_a, pred_b, names, output_path, top_n):
    features = list(shap_a.columns) + [f for f in shap_b.columns if f not in set(shap_a.columns)]
    shap_a, shap_b = shap_a.reindex(columns=features).fillna(0.0), shap_b.reindex(columns=features).fillna(0.0)

    delta = (shap_b.mean() - shap_a.mean()).abs().sort_values(ascending=False)
    top = list(delta.index[:top_n])
    rest = [f for f in features if f not in set(top)]
    rows_a = [shap_a[f].values for f in top] + [shap_a[rest].sum(axis=1).values]
    rows_b = [shap_b[f].values for f in top] + [shap_b[rest].sum(axis=1).values]
    labels = [short_name(f) for f in top] + ['All other features']
    y = np.arange(len(labels))[::-1]

    fig = plt.figure(figsize=(10.5, 0.3 * len(labels) + 1.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[3.2, 1], wspace=0.08)
    ax, ax2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    for yi in y:
        ax.axhline(yi, color='#BBBBBB', linewidth=0.8, linestyle=':', zorder=0)
    ax.axvline(0, color='0.3', lw=0.8, zorder=0)
    for rows, color, name in ((rows_a, VARIANT_PALETTE[0], names[0]), (rows_b, VARIANT_PALETTE[1], names[1])):
        stats = np.array([interval(r) for r in rows])
        ax.errorbar(stats[:, 0], y, xerr=[np.clip(stats[:, 0] - stats[:, 1], 0, None),
                                          np.clip(stats[:, 2] - stats[:, 0], 0, None)],
                    fmt='none', ecolor=color, elinewidth=1.0, capsize=2.5, alpha=0.6, zorder=1.5)
        ax.scatter(stats[:, 0], y, color=color, s=70, label=name, zorder=2, edgecolor='white', linewidth=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.get_yticklabels()[-1].set_style('italic')
    ax.tick_params(axis='y', length=0)
    ax.grid(False)
    ax.set_xlabel('mean SHAP value (log-odds contribution to predicted risk)')
    ax.set_title('(a)', loc='left', fontsize=11)
    ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=2)

    for x, (p, color) in enumerate(((pred_a, VARIANT_PALETTE[0]), (pred_b, VARIANT_PALETTE[1]))):
        mean, lo, hi = interval(p)
        lo, hi = max(lo, 0.0), min(hi, 1.0)
        ax2.errorbar([x], [mean], yerr=[[max(mean - lo, 0)], [max(hi - mean, 0)]], fmt='none', ecolor=color,
                     elinewidth=1.0, capsize=2.5, alpha=0.6, zorder=1.5)
        ax2.scatter([x], [mean], color=color, s=70, zorder=2, edgecolor='white', linewidth=0.6)
        ax2.text(x + 0.12, mean, '{:.2f}'.format(mean), va='center')
    ax2.set_xlim(-0.5, 1.7)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(names)
    ax2.set_ylim(0, 1)
    ax2.set_ylabel('mean mortality prediction')
    ax2.yaxis.set_label_position('right')
    ax2.yaxis.tick_right()
    ax2.grid(axis='x', visible=False)
    ax2.set_title('(b)', loc='left', fontsize=11)

    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    print('Saved {}'.format(output_path))


def main():
    parser = argparse.ArgumentParser(description='SHAP values and prediction of one stay in two runs.')
    parser.add_argument('--a', nargs=2, required=True, metavar=('RESULTS_DIR', 'LABEL'))
    parser.add_argument('--b', nargs=2, required=True, metavar=('RESULTS_DIR', 'LABEL'))
    parser.add_argument('--stay_id', type=int, required=True)
    parser.add_argument('--task', default='mortality')
    parser.add_argument('--top_n', type=int, default=8)
    parser.add_argument('--name_a', default='charttime')
    parser.add_argument('--name_b', default='storetime')
    parser.add_argument('--output_path', required=True)
    args = parser.parse_args()

    shaps, preds = [], []
    for results_dir, label in (args.a, args.b):
        _, order = load_predictions(results_dir, label, args.task)
        if args.stay_id not in order:
            raise ValueError('stay_id {} not in test set of {}'.format(args.stay_id, label))
        X = load_features(results_dir, label, order)
        shaps.append(shap_per_seed(results_dir, label, X, args.stay_id))
        preds.append(predictions_per_seed(results_dir, label, args.task, args.stay_id))
        print('{}: {} seeds'.format(label, len(shaps[-1])))
    plot_case(shaps[0], shaps[1], preds[0], preds[1], (args.name_a, args.name_b), args.output_path, args.top_n)


if __name__ == '__main__':
    main()
