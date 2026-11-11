# mimic-iv-preprocess
Preprocessing MIMIC-IV requires many different choices. For our experiments, we investigate the impact of some of them. The code of this repository adapts and extends the mimic4extract pipeline to investigate differences in preprocessing choices of time series and tabular data, with the experiments being closely inspired by HAIM. The [text-section leakage experiment](experiments/text_section_leakage/README.md) studies retrospective information in admission sections reconstructed from MIMIC-IV discharge summaries. It has its own hospital-admission cohort, patient-level splits, and configuration separate from the ICU preprocessing below.

## Preprocessing (for experiments 4.1 and 4.2)
This section will detail our preprocessing for Experiments 4.1 and 4.2
### Changes from mimic4preprocess
Some of the minor changes we made simply ensured that the code works with the 3.1 version of MIMIC-IV.
Noteworthy changes are: 
- the option of choosing storetime as anchor timestamp, or choosing a timestamp per feature family (hybrid)
- the option to encode missingness of hourly measurements as binary variables
- addition of more laboratory values 
- restricting laboratory measurements to only those matching in unit and specimen fluid. However, it is possible that there are lab_ids currently not being considered. 
-stricter checking of in-house mortality.
- removed the Glasgow overall score and capillary refill rate as the IDs used in the script could not find any valid measurements in MIMIC-IV 3.1 and they were empty variables.

Note: the code we adapted contained a spelling error, turning "Glasgow" into "Glascow", which so far remains unfixed 
### Installation
You can clone the code and install all required packages by running the following lines:
```
git clone https://github.com/m2l-lab/mimic-iv-preprocess && cd mimic-iv-preprocess
conda env create -f environment.yml
conda activate mimic-pipeline
```
Start by running the following commands, changing the paths as needed. This will handle the general preprocessing, extracting all episodes with charttime and storetime timestamps ("_s" file ending for storetime) and splitting patients into train and test.
```
python -m mimic3benchmark.scripts.extract_subjects_iv your_path/physionet/files/mimiciv data/root/
python -m mimic3benchmark.scripts.validate_events data/root/
python -m mimic3benchmark.scripts.extract_episodes_from_subjects data/root/
python -m mimic3benchmark.scripts.split_train_and_test data/root/
```

The next step creates task-specific episodes. Data is also binned into 1-hour bins. The choice between charttime and storetime needs to be made. Optionally, binary variables encoding hourly missingness can be created for time series variables. For following our experiments they are not needed as we later add an aggregated missingness-count feature. If our preprocessing is used for other purposes, they might be helpful to keep the missingness information. 

```
#with missingness variables
#python -m mimic3benchmark.scripts.create_in_hospital_mortality data/root data/mort_store --variant storetime --mortality_column Mortality --include_mask 
#without missingness variables
python -m mimic3benchmark.scripts.create_in_hospital_mortality data/root data/mort_store --variant storetime --mortality_column Mortality --no-include_mask 
python -m mimic3benchmark.scripts.create_in_hospital_mortality data/root data/mort_chart --variant charttime --mortality_column Mortality --no-include_mask
python -m mimic3benchmark.scripts.create_in_hospital_mortality data/root data/mort_hybrid --variant hybrid    --mortality_column Mortality --no-include_mask
python -m mimic3benchmark.scripts.create_phenotyping data/root data/pheno_hybrid --variant hybrid --no-include_mask
python -m mimic3benchmark.split_train_val data/mort_store
python -m mimic3benchmark.split_train_val data/mort_chart
python -m mimic3benchmark.split_train_val data/mort_hybrid
python -m mimic3benchmark.split_train_val data/pheno_hybrid
```

For the phenotype experiment, we also preprocess the CXR images (adjust the path in resize_images accordingly):
```
python resize_images.py
python -m haim_experiments.image_embeddings --cxr_metadata data/mimic-cxr-2.0.0-metadata.csv --all_stays data/root/all_stays.csv --image_dir data/resized --output_csv data/image_embeddings_full.csv --weights densenet121-res224-chex --include_predictions

```
## Experiments 4.1 and 4.2
This details the HAIM-derived experiments regarding mortality and phenotype classification we included in our paper, along with the visualization of results.

### Mortality Prediction Experiment 4.1
Training the different mortality models (timestamp x imputation x missingness (12 runs), injected pitfalls (2 runs)):
```
S="42,43,44,45,46"     # training seeds
R=results              # results directory
for V in hybrid store chart; do
  python -m haim_experiments.train_mortality --root_path data/root --binned_dir data/mort_$V \
    --output_dir $R/mortality_${V}_static       --fallback_mode clinical   --no-include_measured_count \
    --variant_label ${V}_static       --seeds $S
  python -m haim_experiments.train_mortality --root_path data/root --binned_dir data/mort_$V \
    --output_dir $R/mortality_${V}_static_miss  --fallback_mode clinical \
    --variant_label ${V}_static_miss  --seeds $S
  python -m haim_experiments.train_mortality --root_path data/root --binned_dir data/mort_$V \
    --output_dir $R/mortality_${V}_statistical         --fallback_mode train_mean --no-include_measured_count \
    --variant_label ${V}_statistical         --seeds $S
  python -m haim_experiments.train_mortality --root_path data/root --binned_dir data/mort_$V \
    --output_dir $R/mortality_${V}_statistical_miss    --fallback_mode train_mean \
    --variant_label ${V}_statistical_miss    --seeds $S
done
B="--root_path data/root --binned_dir data/mort_hybrid --fallback_mode clinical --seeds $S"
python -m haim_experiments.train_mortality $B --output_dir $R/inj_icd \
  --include_icd_phenotypes --variant_label inj_icd
python -m haim_experiments.train_mortality --root_path data/root_staysplit \
  --binned_dir data/mort_hybrid_staysplit --fallback_mode clinical --seeds $S \
  --output_dir $R/inj_staysplit --variant_label inj_staysplit
```
### Phenotype Experiment 4.2
Training with, without image modality, and 20 pca:
```
P="--root_path data/root --binned_dir data/pheno_hybrid --cxr_metadata data/mimic-cxr-2.0.0-metadata.csv --fallback_mode clinical --seeds $S"
I="--image_embeddings data/image_embeddings_full.csv --image_feature_set dense"
python -m haim_experiments.train_phenotyping $P    --output_dir $R/pheno_noimg --no_image_features --variant_label no_cxr
python -m haim_experiments.train_phenotyping $P $I --output_dir $R/pheno       --variant_label cxr_dense
python -m haim_experiments.train_phenotyping $P $I --output_dir $R/pheno_pca20 \
  --image_transform pca --image_pca_dim 20 --variant_label cxr_pca20
```
### Bootstrapping
```
#Mortality
files=""
for c in static statistical static_miss statistical_miss; do
  for V in hybrid store chart; do
    files="$files $R/mortality_${V}_$c/${V}_${c}_test_predictions.csv"
  done
done
python -m haim_experiments.bootstrap_ci $files --output $R/ci_table4.csv
 
python -m haim_experiments.bootstrap_ci \
    $R/inj_icd/inj_icd_test_predictions.csv \
    $R/inj_staysplit/inj_staysplit_test_predictions.csv \
    --output $R/ci_ablations.csv
#Phenotype
python -m haim_experiments.bootstrap_compare --n_boot 2000 \
  --groups haim_experiments/cxr_phenotype_groups.json --output $R/boot_pheno.csv \
  --pair $R/pheno/cxr_dense_test_predictions.csv       $R/pheno_noimg/no_cxr_test_predictions.csv \
  --pair $R/pheno_pca20/cxr_pca20_test_predictions.csv $R/pheno_noimg/no_cxr_test_predictions.csv \
  --pair $R/pheno/cxr_dense_test_predictions.csv       $R/pheno_pca20/cxr_pca20_test_predictions.csv
#SHAP
runs="--run $R/inj_icd inj_icd"
for c in statistical static statistical_miss static_miss; do for V in store chart; do
  runs="$runs --run $R/mortality_${V}_$c ${V}_$c"
done; done
python -m haim_experiments.bootstrap_shap mortality $runs --output $R/boot_shap_mort.csv
python -m haim_experiments.plot_results single --results_dir $R/inj_icd --variant_label inj_icd \
  --bootstrap_shap_csv $R/boot_shap_mort.csv --output_dir $R/figures
 
```
### Feature importance shift
```
# The 11 timestamp / imputation / missingness variants
python -m haim_experiments.explanation_shift \
  --baseline $R/mortality_hybrid_static_miss hybrid_static_miss \
  --run $R/mortality_hybrid_static hybrid_static \
  --run $R/mortality_hybrid_statistical hybrid_statistical \
  --run $R/mortality_hybrid_statistical_miss hybrid_statistical_miss \
  --run $R/mortality_store_static store_static \
  --run $R/mortality_store_static_miss store_static_miss \
  --run $R/mortality_store_statistical store_statistical \
  --run $R/mortality_store_statistical_miss store_statistical_miss \
  --run $R/mortality_chart_static chart_static \
  --run $R/mortality_chart_static_miss chart_static_miss \
  --run $R/mortality_chart_statistical chart_statistical \
  --run $R/mortality_chart_statistical_miss chart_statistical_miss \
  --output $R/explanation_shift.csv --latex $R/explanation_shift.tex

# ICD injection against baseline
python -m haim_experiments.explanation_shift \
  --baseline $R/mortality_hybrid_static_miss hybrid_static_miss \
  --run $R/inj_icd inj_icd \
  --output $R/icd_feature_shift.csv --latex $R/icd_feature_shift.tex
```
### Visualizations

the visualization from our paper can be created with:
```
#feature shift of one case
python -m haim_experiments.plot_case \
  --a results/mortality_chart_preferred_miss chart_static_miss \
  --b results/mortality_store_preferred_miss store_static_miss \
  --stay_id 00000000 --top_n 6 \ #put stay id that should be displayed
  --output_path results/fig_case_example_prediction.pdf

python -m haim_experiments.plot_results shap_vs_delta \
  --results_dir_with $R/pheno --variant_label_with cxr_dense \
  --results_dir_with_b $R/pheno_pca20 --variant_label_with_b cxr_pca20 \
  --results_dir_without $R/pheno_noimg --variant_label_without no_cxr \
  --bootstrap_csv $R/boot_pheno.csv --bootstrap_shap_csv $R/boot_shap_pheno.csv \
  --output_path $R/figures/fig_shap_vs_delta_dense_pca20.pdf
```


## Thanks
We wish to thank the contributors of the repository [MedFuse](https://github.com/nyuad-cai/MedFuse?tab=readme-ov-file), from which we adapted the mimic4process scripts, which are in turn based upon the repository [mimic3-benchmarks](https://github.com/YerevaNN/mimic3-benchmarks/), which we also wish to acknowledge. We also wish to tank the contributors of the repository [HAIM](https://github.com/lrsoenksen/HAIM), from which we derived our experiments.
Our text-based experiment is built upon the [COP-IV extraction](https://github.com/DATEXIS/ClinicalOutcomePrediction-IV).
