"""Train an XGBoost mortality model (HAIM-style: tabular + timeseries HAIM
features) for ONE variant (chart or store), and compute SHAP feature +
modality importance.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score

from haim_experiments.build_dataset import (build_feature_matrix, align_columns, episode_paths_from_listfile,
                             filter_first_stay_per_patient)
from haim_experiments.haim_features import compute_train_variable_stats, compute_train_tabular_stats
from haim_experiments.cohort_stats import compute_cohort_stats, print_cohort_stats_summary
from haim_experiments.seed_aggregation import mean_std_ci, aggregate_feature_importance, aggregate_modality_importance


def main():
    parser = argparse.ArgumentParser(description="Train HAIM-style mortality model + SHAP.")
    parser.add_argument('--root_path', type=str, required=True,
                         help="Root path.csv")
    parser.add_argument('--binned_dir', type=str, required=True,
                         help="Output dir of create_in_hospital_mortality.py")
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--fallback_mode', type=str, default='clinical', choices=['clinical', 'train_mean'])
    parser.add_argument('--variant_label', type=str, default='variant',
                         help="Free-text label for this run, e.g. 'chart' or 'store' (used in output filenames only).")
    parser.add_argument('--include_measured_count', dest='include_measured_count',
                         action='store_true', default=True,
                         help="Add the measured_count / was_missing features ")
    parser.add_argument('--no-include_measured_count', dest='include_measured_count',
                         action='store_false',
                         help="Run without the measured_count / was_missing features ")
    parser.add_argument('--n_estimators', type=int, default=350,
                         help="Fixed number of boosting rounds (default: 350)")
    parser.add_argument('--early_stopping', dest='early_stopping', action='store_true', default=False,
                         help="Enable early stopping on the validation set (default: OFF, for comparability.")
    parser.add_argument('--no-early_stopping', dest='early_stopping', action='store_false')
    parser.add_argument('--n_jobs', type=int, default=1,
                         help="XGBoost thread count (default: 1). Matters for bit-exact reproducibility")
    parser.add_argument('--seeds', type=str, default='42',
                         help="Comma-separated list of XGBoost random_state values to train with")
    parser.add_argument('--one_stay_per_patient', action='store_true', default=False,
                         help="Restrict each of train/val/test to each patient's chronologically first "
                              "ICU episode only (default: off, keeps all stays)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    train_listfile = pd.read_csv(os.path.join(args.binned_dir, 'train_listfile.csv'))
    val_listfile = pd.read_csv(os.path.join(args.binned_dir, 'val_listfile.csv'))
    test_listfile = pd.read_csv(os.path.join(args.binned_dir, 'test_listfile.csv'))

    if args.one_stay_per_patient:
        n_before = len(train_listfile) + len(val_listfile) + len(test_listfile)
        train_listfile = filter_first_stay_per_patient(train_listfile)
        val_listfile = filter_first_stay_per_patient(val_listfile)
        test_listfile = filter_first_stay_per_patient(test_listfile)
        n_after = len(train_listfile) + len(val_listfile) + len(test_listfile)
        print("--one_stay_per_patient set: restricted to each patient's first ICU episode "
              "({} -> {} stays total).".format(n_before, n_after))

    binned_train_phys = os.path.join(args.binned_dir, 'train')  # train AND val stays physically here
    binned_test_phys = os.path.join(args.binned_dir, 'test')
    root_train_phys = os.path.join(args.root_path, 'train')
    root_test_phys = os.path.join(args.root_path, 'test')

    print("Computing train-set variable means (train_listfile stays only, no val leakage)...")
    train_stats = compute_train_variable_stats(
        binned_train_phys, stay_filenames=train_listfile['stay'].tolist())

    print("Computing train-set tabular stats (train_listfile stays only, no val leakage)...")
    train_episode_paths = episode_paths_from_listfile(train_listfile, root_train_phys)
    tabular_train_stats = compute_train_tabular_stats(
        root_train_phys, train_episode_paths,
        numeric_agg='mean' if args.fallback_mode == 'train_mean' else 'median')

    print("Building feature matrices...")
    X_train, y_train, meta_train = build_feature_matrix(
        train_listfile, binned_train_phys, root_train_phys, train_stats, args.fallback_mode,
        include_measured_count=args.include_measured_count, tabular_train_stats=tabular_train_stats,
        include_tabular_missing_indicator=args.include_measured_count)
    X_val, y_val, meta_val = build_feature_matrix(
        val_listfile, binned_train_phys, root_train_phys, train_stats, args.fallback_mode,
        include_measured_count=args.include_measured_count, tabular_train_stats=tabular_train_stats,
        include_tabular_missing_indicator=args.include_measured_count)
    X_test, y_test, meta_test = build_feature_matrix(
        test_listfile, binned_test_phys, root_test_phys, train_stats, args.fallback_mode,
        include_measured_count=args.include_measured_count, tabular_train_stats=tabular_train_stats,
        include_tabular_missing_indicator=args.include_measured_count)

    X_val = align_columns(X_train, X_val)
    X_test = align_columns(X_train, X_test)
    for col in X_train.columns:
        if str(X_train[col].dtype) == 'category':
            X_val[col] = X_val[col].astype('category')
            X_test[col] = X_test[col].astype('category')

    print("Train: {} rows, {} features | Val: {} rows | Test: {} rows".format(
        X_train.shape[0], X_train.shape[1], X_val.shape[0], X_test.shape[0]))

    print("\nComputing cohort statistics (patient counts, label prevalence, missingness)...")
    cohort_stats = {}
    for part_name, X_part, meta_part, y_part in [
        ('train', X_train, meta_train, y_train), ('val', X_val, meta_val, y_val), ('test', X_test, meta_test, y_test)
    ]:
        part_stats = compute_cohort_stats(X_part, meta_part, label_data=y_part)
        cohort_stats[part_name] = part_stats
        print_cohort_stats_summary(part_name, part_stats)

    seeds = [int(s.strip()) for s in args.seeds.split(',') if s.strip()]
    print("\nTraining with {} seed(s): {}".format(len(seeds), seeds))

    per_seed_results = []  # list of dicts, one per seed
    for seed in seeds:
        xgb_kwargs = dict(
            n_estimators=args.n_estimators,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='auc',
            enable_categorical=True,
            random_state=seed,
            n_jobs=args.n_jobs,
        )
        if args.early_stopping:
            xgb_kwargs['early_stopping_rounds'] = 30
        model = xgb.XGBClassifier(**xgb_kwargs)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        trees_used = int(model.get_booster().num_boosted_rounds())

        test_pred = model.predict_proba(X_test)[:, 1]
        auroc = roc_auc_score(y_test, test_pred)
        auprc = average_precision_score(y_test, test_pred)
        print("[seed={}] Test AUROC: {:.4f}  AUPRC: {:.4f}  trees_used={}".format(seed, auroc, auprc, trees_used))

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        if isinstance(shap_values, list):  # some xgboost/shap version combos return a list for binary clf
            shap_values = shap_values[1]

        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        importance_df = pd.DataFrame({
            'feature': X_test.columns,
            'mean_abs_shap': mean_abs_shap,
        }).sort_values('mean_abs_shap', ascending=False)

        def _modality(col):
            if col.startswith('ts_'):
                return 'timeseries'
            if col.startswith('tab_'):
                return 'tabular'
            return 'other'

        importance_df['modality'] = importance_df['feature'].apply(_modality)
        modality_importance = importance_df.groupby('modality')['mean_abs_shap'].sum().sort_values(ascending=False)
        modality_importance_pct = (modality_importance / modality_importance.sum() * 100).round(2)

        per_seed_results.append({
            'seed': seed, 'model': model, 'auroc': auroc, 'auprc': auprc, 'trees_used': trees_used,
            'shap_values': shap_values, 'importance_df': importance_df,
            'modality_importance_pct': modality_importance_pct,
        })

    prefix = args.variant_label

    #Per-seed raw results, always saved
    perseed_records = [{'seed': r['seed'], 'test_auroc': float(r['auroc']), 'test_auprc': float(r['auprc']),
                         'trees_used': r['trees_used']} for r in per_seed_results]
    with open(os.path.join(args.output_dir, '{}_perseed_metrics.json'.format(prefix)), 'w') as f:
        json.dump(perseed_records, f, indent=2)

    auroc_stats = mean_std_ci([r['auroc'] for r in per_seed_results])
    auprc_stats = mean_std_ci([r['auprc'] for r in per_seed_results])
    print("\nAcross {} seed(s): AUROC = {:.4f} +/- {:.4f} (95% CI [{:.4f}, {:.4f}])".format(
        len(seeds), auroc_stats['mean'], auroc_stats['std'], auroc_stats['ci_lower'], auroc_stats['ci_upper']))
    print("Across {} seed(s): AUPRC = {:.4f} +/- {:.4f} (95% CI [{:.4f}, {:.4f}])".format(
        len(seeds), auprc_stats['mean'], auprc_stats['std'], auprc_stats['ci_lower'], auprc_stats['ci_upper']))

    #Aggregated (or, with a single seed, equivalent-to-before)
    importance_df_agg = aggregate_feature_importance([r['importance_df'] for r in per_seed_results])
    modality_df_agg = aggregate_modality_importance([r['modality_importance_pct'] for r in per_seed_results])

    importance_df_agg.to_csv(os.path.join(args.output_dir, '{}_feature_importance.csv'.format(prefix)), index=False)
    modality_df_agg.to_csv(os.path.join(args.output_dir, '{}_modality_importance_detailed.csv'.format(prefix)),
                            index=False)

    modality_df_agg.set_index('modality')['mean_pct'].to_csv(
        os.path.join(args.output_dir, '{}_modality_importance.csv'.format(prefix)))

    shap_values_avg = np.mean([r['shap_values'] for r in per_seed_results], axis=0)
    np.save(os.path.join(args.output_dir, '{}_shap_values.npy'.format(prefix)), shap_values_avg)
    meta_test.to_csv(os.path.join(args.output_dir, '{}_test_meta.csv'.format(prefix)), index=False)
    X_test.to_csv(os.path.join(args.output_dir, '{}_test_features.csv'.format(prefix)), index=False)

    for r in per_seed_results:
        seed_suffix = '' if len(seeds) == 1 else '_seed{}'.format(r['seed'])
        r['model'].save_model(os.path.join(args.output_dir, '{}{}_model.json'.format(prefix, seed_suffix)))

    metrics = {
        'variant_label': args.variant_label,
        'fallback_mode': args.fallback_mode,
        'n_train': int(X_train.shape[0]),
        'n_val': int(X_val.shape[0]),
        'n_test': int(X_test.shape[0]),
        'n_features': int(X_train.shape[1]),
        'n_estimators_requested': args.n_estimators,
        'early_stopping': args.early_stopping,
        'seeds': seeds,
        'n_seeds': len(seeds),
        'test_auroc': auroc_stats['mean'],
        'test_auprc': auprc_stats['mean'],
        'test_auroc_ci': {k: auroc_stats[k] for k in ('mean', 'std', 'ci_lower', 'ci_upper', 'n')},
        'test_auprc_ci': {k: auprc_stats[k] for k in ('mean', 'std', 'ci_lower', 'ci_upper', 'n')},
        'modality_importance_pct': dict(zip(modality_df_agg['modality'], modality_df_agg['mean_pct'])),
    }
    with open(os.path.join(args.output_dir, '{}_metrics.json'.format(prefix)), 'w') as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(args.output_dir, '{}_cohort_stats.json'.format(prefix)), 'w') as f:
        json.dump(cohort_stats, f, indent=2)

    print("\nSaved results to {}".format(args.output_dir))


if __name__ == '__main__':
    main()
