
import argparse
import json
import os
import re

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score

from haim_experiments.build_dataset import align_columns, episode_paths_from_listfile, filter_first_stay_per_patient
from haim_experiments.build_phenotyping_dataset import build_phenotyping_feature_matrix, infer_label_columns
from haim_experiments.cxr_matching import match_images_to_stays, assign_matched_split, print_matching_summary
from haim_experiments.haim_features import compute_train_variable_stats, compute_train_tabular_stats
from haim_experiments.cohort_stats import compute_cohort_stats, print_cohort_stats_summary
from haim_experiments.seed_aggregation import mean_std_ci, aggregate_feature_importance, aggregate_modality_importance



def _safe_name(phenotype):
    return re.sub(r'[^A-Za-z0-9]+', '_', phenotype).strip('_')


def _modality(col):
    if col.startswith('ts_'):
        return 'timeseries'
    if col.startswith('tab_'):
        return 'tabular'
    if col.startswith('img_'):
        return 'image'
    return 'other'


def main():
    parser = argparse.ArgumentParser(description="Train HAIM-style phenotyping models + SHAP (matched subset).")
    parser.add_argument('--root_path', type=str, required=True,
                         help="Root path with ")
    parser.add_argument('--binned_dir', type=str, required=True,
                         help="Output dir of create_phenotyping.py")
    parser.add_argument('--cxr_metadata', type=str, required=True,
                         help="Path to mimic-cxr-2.0.0-metadata.csv.")
    parser.add_argument('--all_stays', type=str, default=None,
                         help="Path to all_stays.csv. ")
    parser.add_argument('--image_embeddings', type=str, default=None,
                         help="Precomputed image embeddings CSV")
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--fallback_mode', type=str, default='clinical', choices=['clinical', 'train_mean'])
    parser.add_argument('--variant_label', type=str, default='variant',
                         help="Free-text label for this run, e.g. 'chart' or 'store' (used in output filenames).")
    parser.add_argument('--n_estimators', type=int, default=350,
                         help="Fixed number of boosting rounds (default: 350).")
    parser.add_argument('--early_stopping', dest='early_stopping', action='store_true', default=False,
                         help="Enable early stopping on the validation set (default: OFF).")
    parser.add_argument('--no-early_stopping', dest='early_stopping', action='store_false')
    parser.add_argument('--include_measured_count', dest='include_measured_count',
                         action='store_true', default=True,
                         help="Add the measured_count / was_missing features.")
    parser.add_argument('--no-include_measured_count', dest='include_measured_count', action='store_false',
                         help="Run without the measured_count / was_missing features.")
    parser.add_argument('--min_positives', type=int, default=10,
                         help="Skip a phenotype in train/val/test evaluation if it has fewer than this many "
                              "positive cases in the TRAIN split ")
    parser.add_argument('--n_jobs', type=int, default=1,
                         help="XGBoost thread count (default: 1)")
    parser.add_argument('--seeds', type=str, default='42',
                         help="Comma-separated list of XGBoost random_state values to train with")
    parser.add_argument('--no_image_features', action='store_true', default=False,
                         help="Ablation: keep the same matched-subset cohort but DO NOT attach any image features")
    parser.add_argument('--image_feature_set', choices=['dense', 'predictions', 'both'], default='dense',
                         help="Which columns of image_embeddings to actually use (default: 'dense')")
    parser.add_argument('--one_stay_per_patient', action='store_true', default=False,
                         help="Restrict each of train/val/test to each patient's chronologically first "
                              "ICU episode only (default: off, keeps all stays).")
    args = parser.parse_args()

    if not args.no_image_features and not args.image_embeddings:
        raise ValueError("--image_embeddings is required unless --no_image_features is set.")

    os.makedirs(args.output_dir, exist_ok=True)
    all_stays_path = args.all_stays or os.path.join(args.root_path, 'all_stays.csv')

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

    # Computed from the RAW listfile, before merging with CXR metadata --
    # see infer_label_columns()'s docstring for why this must not be done
    # on the post-merge matched_df instead.
    label_columns = infer_label_columns(train_listfile)

    print("Matching CXR images to ICU stays (48h window, AP view, latest image)...")
    matched_images = match_images_to_stays(args.cxr_metadata, all_stays_path)
    split_dict = assign_matched_split(matched_images, train_listfile, val_listfile, test_listfile)
    matching_yield_stats = print_matching_summary(matched_images, split_dict, train_listfile, val_listfile, test_listfile)

    if args.no_image_features:
        print("--no_image_features set: keeping the matched-image-subset cohort, but NOT attaching "
              "any image features to X (ablation).")
        image_embeddings_df = None
    else:
        image_embeddings_df = pd.read_csv(args.image_embeddings)
        emb_cols = [c for c in image_embeddings_df.columns if c.startswith('img_emb_')]
        pred_cols = [c for c in image_embeddings_df.columns if c.startswith('img_pred_')]

        if args.image_feature_set == 'dense':
            keep_cols = emb_cols
        elif args.image_feature_set == 'predictions':
            keep_cols = pred_cols
        else:  # both
            keep_cols = emb_cols + pred_cols

        if args.image_feature_set in ('predictions', 'both') and not pred_cols:
            raise ValueError(
                "--image_feature_set '{}' requires prediction columns (img_pred_*) in --image_embeddings, "
                "but none were found. Regenerate the embeddings CSV with "
                "image_embeddings.py's --include_predictions flag.".format(args.image_feature_set))
        if not keep_cols:
            raise ValueError("No image feature columns found matching --image_feature_set='{}' in {}.".format(
                args.image_feature_set, args.image_embeddings))

        print("--image_feature_set='{}': using {} image column(s) ({}).".format(
            args.image_feature_set, len(keep_cols),
            "dense only" if args.image_feature_set == 'dense' else
            "predictions only" if args.image_feature_set == 'predictions' else "dense + predictions"))
        image_embeddings_df = image_embeddings_df[['stay_id'] + keep_cols]

    binned_train_phys = os.path.join(args.binned_dir, 'train')  # train AND val stays physically live here
    binned_test_phys = os.path.join(args.binned_dir, 'test')
    root_train_phys = os.path.join(args.root_path, 'train')
    root_test_phys = os.path.join(args.root_path, 'test')

    print("Computing train-set stats (matched train stays only, no val/test leakage)...")
    train_stats = compute_train_variable_stats(
        binned_train_phys, stay_filenames=split_dict['train']['stay'].tolist())
    train_episode_paths = episode_paths_from_listfile(split_dict['train'], root_train_phys)
    tabular_train_stats = compute_train_tabular_stats(
        root_train_phys, train_episode_paths,
        # Tied to --fallback_mode, same "naive vs. careful" axis as the
        # timeseries fallback, rather than a separate flag: 'train_mean'
        # (default/naive) -> mean; 'clinical' (preferred/careful) -> median.
        numeric_agg='mean' if args.fallback_mode == 'train_mean' else 'median')

    print("Building feature matrices...")
    X_train, Y_train, meta_train, _ = build_phenotyping_feature_matrix(
        split_dict['train'], binned_train_phys, root_train_phys, image_embeddings_df,
        train_stats, tabular_train_stats, args.fallback_mode,
        args.include_measured_count, args.include_measured_count, label_columns=label_columns)
    X_val, Y_val, meta_val, _ = build_phenotyping_feature_matrix(
        split_dict['val'], binned_train_phys, root_train_phys, image_embeddings_df,
        train_stats, tabular_train_stats, args.fallback_mode,
        args.include_measured_count, args.include_measured_count, label_columns=label_columns)
    X_test, Y_test, meta_test, _ = build_phenotyping_feature_matrix(
        split_dict['test'], binned_test_phys, root_test_phys, image_embeddings_df,
        train_stats, tabular_train_stats, args.fallback_mode,
        args.include_measured_count, args.include_measured_count, label_columns=label_columns)

    X_val = align_columns(X_train, X_val)
    X_test = align_columns(X_train, X_test)
    for col in X_train.columns:
        if str(X_train[col].dtype) == 'category':
            X_val[col] = X_val[col].astype('category')
            X_test[col] = X_test[col].astype('category')

    print("Train: {} rows, {} features, {} phenotypes | Val: {} rows | Test: {} rows".format(
        X_train.shape[0], X_train.shape[1], len(label_columns), X_val.shape[0], X_test.shape[0]))

    print("\nComputing cohort statistics (patient counts, label prevalence, missingness)...")
    cohort_stats = {}
    for part_name, X_part, meta_part, Y_part in [
        ('train', X_train, meta_train, Y_train), ('val', X_val, meta_val, Y_val), ('test', X_test, meta_test, Y_test)
    ]:
        part_stats = compute_cohort_stats(X_part, meta_part, label_data=Y_part, label_columns=label_columns)
        part_stats['image_matching'] = matching_yield_stats.get(part_name, {})
        cohort_stats[part_name] = part_stats
        print_cohort_stats_summary(part_name, part_stats)

    seeds = [int(s.strip()) for s in args.seeds.split(',') if s.strip()]
    print("\nTraining with {} seed(s) per phenotype: {}".format(len(seeds), seeds))

    all_metrics = []
    for phenotype in label_columns:
        y_train = Y_train[phenotype].values
        y_val = Y_val[phenotype].values
        y_test = Y_test[phenotype].values

        n_pos_train = int(y_train.sum())
        if n_pos_train < args.min_positives or len(np.unique(y_train)) < 2:
            print("Skipping '{}': only {} positive case(s) in train (min_positives={}).".format(
                phenotype, n_pos_train, args.min_positives))
            continue

        per_seed_results = []
        for seed in seeds:
            xgb_kwargs = dict(
                n_estimators=args.n_estimators, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, eval_metric='auc',
                enable_categorical=True, random_state=seed, n_jobs=args.n_jobs,
            )
            if args.early_stopping:
                xgb_kwargs['early_stopping_rounds'] = 30
            model = xgb.XGBClassifier(**xgb_kwargs)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

            test_pred = model.predict_proba(X_test)[:, 1]
            try:
                auroc = roc_auc_score(y_test, test_pred) if len(np.unique(y_test)) > 1 else float('nan')
            except ValueError:
                auroc = float('nan')
            auprc = average_precision_score(y_test, test_pred)

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            importance_df = pd.DataFrame({
                'feature': X_test.columns, 'mean_abs_shap': mean_abs_shap,
            }).sort_values('mean_abs_shap', ascending=False)
            importance_df['modality'] = importance_df['feature'].apply(_modality)
            modality_importance = importance_df.groupby('modality')['mean_abs_shap'].sum().sort_values(ascending=False)
            modality_importance_pct = (modality_importance / modality_importance.sum() * 100).round(2)

            trees_used = int(model.get_booster().num_boosted_rounds())
            per_seed_results.append({
                'seed': seed, 'model': model, 'auroc': auroc, 'auprc': auprc, 'trees_used': trees_used,
                'shap_values': shap_values, 'importance_df': importance_df,
                'modality_importance_pct': modality_importance_pct,
            })

        prefix = "{}_{}".format(args.variant_label, _safe_name(phenotype))

        perseed_records = [{'seed': r['seed'], 'test_auroc': float(r['auroc']), 'test_auprc': float(r['auprc']),
                             'trees_used': r['trees_used']} for r in per_seed_results]
        with open(os.path.join(args.output_dir, '{}_perseed_metrics.json'.format(prefix)), 'w') as f:
            json.dump(perseed_records, f, indent=2)

        valid_aurocs = [r['auroc'] for r in per_seed_results if r['auroc'] == r['auroc']]  # NaN-safe
        auroc_stats = mean_std_ci(valid_aurocs) if valid_aurocs else {'mean': float('nan'), 'std': 0.0,
                                                                        'ci_lower': float('nan'), 'ci_upper': float('nan'), 'n': 0}
        auprc_stats = mean_std_ci([r['auprc'] for r in per_seed_results])

        importance_df_agg = aggregate_feature_importance([r['importance_df'] for r in per_seed_results])
        modality_df_agg = aggregate_modality_importance([r['modality_importance_pct'] for r in per_seed_results])

        importance_df_agg.to_csv(os.path.join(args.output_dir, '{}_feature_importance.csv'.format(prefix)), index=False)
        modality_df_agg.to_csv(
            os.path.join(args.output_dir, '{}_modality_importance_detailed.csv'.format(prefix)), index=False)
        modality_df_agg.set_index('modality')['mean_pct'].to_csv(
            os.path.join(args.output_dir, '{}_modality_importance.csv'.format(prefix)))

        shap_values_avg = np.mean([r['shap_values'] for r in per_seed_results], axis=0)
        np.save(os.path.join(args.output_dir, '{}_shap_values.npy'.format(prefix)), shap_values_avg)
        X_test.to_csv(os.path.join(args.output_dir, '{}_test_features.csv'.format(prefix)), index=False)

        for r in per_seed_results:
            seed_suffix = '' if len(seeds) == 1 else '_seed{}'.format(r['seed'])
            r['model'].save_model(os.path.join(args.output_dir, '{}{}_model.json'.format(prefix, seed_suffix)))

        trees_used = int(round(np.mean([r['trees_used'] for r in per_seed_results])))

        # Per-phenotype metrics.json, same shape as train_mortality.py's, so
        # plot_results.py works unchanged treating each phenotype as its own
        # "variant" (--variant_label {run_label}_{phenotype_safe_name}).
        phenotype_metrics = {
            'variant_label': prefix,
            'phenotype': phenotype,
            'fallback_mode': args.fallback_mode,
            'image_features_included': not args.no_image_features,
            'image_feature_set': (None if args.no_image_features else args.image_feature_set),
            'n_train': int(X_train.shape[0]),
            'n_val': int(X_val.shape[0]),
            'n_test': int(X_test.shape[0]),
            'n_features': int(X_train.shape[1]),
            'n_estimators_requested': args.n_estimators,
            'early_stopping': args.early_stopping,
            'seeds': seeds,
            'n_seeds': len(seeds),
            'trees_used': trees_used,
            'test_auroc': auroc_stats['mean'],
            'test_auprc': auprc_stats['mean'],
            'test_auroc_ci': {k: auroc_stats[k] for k in ('mean', 'std', 'ci_lower', 'ci_upper', 'n')},
            'test_auprc_ci': {k: auprc_stats[k] for k in ('mean', 'std', 'ci_lower', 'ci_upper', 'n')},
            'modality_importance_pct': dict(zip(modality_df_agg['modality'], modality_df_agg['mean_pct'])),
        }
        with open(os.path.join(args.output_dir, '{}_metrics.json'.format(prefix)), 'w') as f:
            json.dump(phenotype_metrics, f, indent=2)

        all_metrics.append({
            'phenotype': phenotype,
            'n_pos_train': n_pos_train,
            'test_auroc': auroc_stats['mean'],
            'test_auprc': auprc_stats['mean'],
            'test_auroc_std': auroc_stats['std'],
            'test_auprc_std': auprc_stats['std'],
            'test_auroc_ci_lower': auroc_stats['ci_lower'],
            'test_auroc_ci_upper': auroc_stats['ci_upper'],
            'test_auprc_ci_lower': auprc_stats['ci_lower'],
            'test_auprc_ci_upper': auprc_stats['ci_upper'],
            'trees_used': trees_used,
            'modality_importance_pct': dict(zip(modality_df_agg['modality'], modality_df_agg['mean_pct'])),
        })
        print("[{}] AUROC={:.4f}+/-{:.4f} AUPRC={:.4f}+/-{:.4f} trees_used={} (n_seeds={})".format(
            phenotype, auroc_stats['mean'], auroc_stats['std'], auprc_stats['mean'], auprc_stats['std'],
            trees_used, len(seeds)))

    metrics_summary = {
        'variant_label': args.variant_label,
        'fallback_mode': args.fallback_mode,
        'image_features_included': not args.no_image_features,
        'image_feature_set': (None if args.no_image_features else args.image_feature_set),
        'n_estimators_requested': args.n_estimators,
        'early_stopping': args.early_stopping,
        'n_train': int(X_train.shape[0]),
        'n_val': int(X_val.shape[0]),
        'n_test': int(X_test.shape[0]),
        'n_features': int(X_train.shape[1]),
        'n_phenotypes_total': len(label_columns),
        'n_phenotypes_trained': len(all_metrics),
        'per_phenotype': all_metrics,
    }
    with open(os.path.join(args.output_dir, '{}_all_metrics.json'.format(args.variant_label)), 'w') as f:
        json.dump(metrics_summary, f, indent=2)
    with open(os.path.join(args.output_dir, '{}_cohort_stats.json'.format(args.variant_label)), 'w') as f:
        json.dump(cohort_stats, f, indent=2)

    print("\nSaved results for {}/{} phenotypes to {}".format(
        len(all_metrics), len(label_columns), args.output_dir))


if __name__ == '__main__':
    main()
