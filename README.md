# mimic-iv-preprocess
Preprocessing MIMIC-IV requires many different choices. This adapts and extends the mimic4extract pipeline to investigate differences in preprocessing choices, with the experiments being closely inspired by HAIM.

## Preprocessing
This section will detail our preprocessing
### Changes from mimic4preprocess
Some of the minor changes we made simply ensured that the code works with the 3.1 version of MIMIC-IV.
Noteworthy changes are: 
- the option of choosing storetime as anchor timestamp
- the option to encode missingness of hourly measurements as binary variables
- addition of more laboratory values 
- restricting laboratory measurements to only those matching in unit and specimen fluid. However, it is possible that there are lab_ids currently not being considered. 
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
python -m mimic3benchmark.scripts.create_phenotyping data/root data/pheno_store --variant storetime --no-include_mask
python -m mimic3benchmark.split_train_val data/mort_store
python -m mimic3benchmark.split_train_val data/mort_chart
python -m mimic3benchmark.split_train_val data/pheno_store
```

For the phenotype experiment, we also preprocess the CXR images (adjust the path in resize_images accordingly):
```
python resize_images.py
python -m haim_experiments.image_embeddings --cxr_metadata data/mimic-cxr-2.0.0-metadata.csv --all_stays data/root/all_stays.csv --image_dir data/resized --output_csv data/image_embeddings_full.csv --weights densenet121-res224-chex --include_predictions

```
## Experiments
This details the HAIM-derived experiments regarding mortality and phenotype classification we included in our paper, along with the visualization of results.

### Mortality Experiment
Training the different mortality models:
```
python -m haim_experiments.train_mortality \
    --root_path data/root --binned_dir data/mort_store --output_dir results/mortality_store_preferred_missing \
    --fallback_mode clinical --variant_label store_preferred_miss --seeds "42,43,44,45,46"
python -m haim_experiments.train_mortality \
    --root_path data/root --binned_dir data/mort_store --output_dir results/mortality_store_default_missing \
    --fallback_mode train_mean --variant_label store_default_miss --seeds "42,43,44,45,46"
python -m haim_experiments.train_mortality \
    --root_path data/root --binned_dir data/mort_store --output_dir results/mortality_store_preferred \
     --fallback_mode clinical --variant_label store_preferred --no-include_measured_count --seeds "42,43,44,45,46"
python -m haim_experiments.train_mortality \
    --root_path data/root --binned_dir data/mort_store --output_dir results/mortality_store_default \
    --fallback_mode train_mean --variant_label store_default --no-include_measured_count --seeds "42,43,44,45,46"
python -m haim_experiments.train_mortality \
    --root_path data/root --binned_dir data/mort_chart --output_dir results/mortality_chart_default \
    --fallback_mode train_mean --variant_label chart_default --no-include_measured_count --seeds "42,43,44,45,46"
python -m haim_experiments.train_mortality \
    --root_path data/root --binned_dir data/mort_chart --output_dir results/mortality_chart_preferred \
    --fallback_mode clinical --variant_label chart_preferred --no-include_measured_count --seeds "42,43,44,45,46"
python -m haim_experiments.train_mortality \
    --root_path data/root --binned_dir data/mort_chart --output_dir results/mortality_chart_preferred_missing \
    --fallback_mode clinical --variant_label chart_preferred_miss --seeds "42,43,44,45,46"
python -m haim_experiments.train_mortality \
    --root_path data/root --binned_dir data/mort_chart --output_dir results/mortality_chart_default_missing \
    --fallback_mode train_mean --variant_label chart_default_miss --seeds "42,43,44,45,46"
```
each experiments results can be analyzed by themselves with:
```
python -m haim_experiments.plot_results single \
  --results_dir results/mortality_store_preferred_missing --variant_label store_preferred_miss
python -m haim_experiments.plot_results single \
  --results_dir results/mortality_chart_default_missing --variant_label chart_default_miss
python -m haim_experiments.plot_results single \
  --results_dir results/mortality_store_default --variant_label store_default
python -m haim_experiments.plot_results single \
  --results_dir results/mortality_store_preferred --variant_label store_preferred
python -m haim_experiments.plot_results single \
  --results_dir results/mortality_chart_preferred_missing --variant_label chart_preferred_miss
python -m haim_experiments.plot_results single \
  --results_dir results/mortality_chart_default_missing --variant_label chart_default_miss
python -m haim_experiments.plot_results single \
  --results_dir results/mortality_chart_default --variant_label chart_default
python -m haim_experiments.plot_results single \
  --results_dir results/mortality_chart_preferred --variant_label chart_preferred
```
the visualization from our paper can be created with:
```
python -m haim_experiments.plot_results dumbbell \
    --results_dir_a1 results/mortality_chart_default --variant_label_a1 chart_default \
    --results_dir_a2 results/mortality_store_default --variant_label_a2 store_default \
    --results_dir_b1 results/mortality_chart_preferred --variant_label_b1 chart_preferred \
    --results_dir_b2 results/mortality_store_preferred --variant_label_b2 store_preferred \
    --results_dir_c1 results/mortality_chart_default_missing --variant_label_c1 chart_default_miss \
    --results_dir_c2 results/mortality_store_default_missing --variant_label_c2 store_default_miss \
    --results_dir_d1 results/mortality_chart_preferred_missing --variant_label_d1 chart_preferred_miss \
    --results_dir_d2 results/mortality_store_preferred_missing --variant_label_d2 store_preferred_miss \
    --legend_name_1 charttime --legend_name_2 storetime \
    --panel_a_caption a --panel_b_caption b --panel_c_caption c --panel_d_caption d \
    --output_path results/fig_shap_dumbbell_full.pdf \
    --top_n 10
```
### Phenotype Experiment
Training with and without image modality:
```
python -m haim_experiments.train_phenotyping \
    --root_path data/root --binned_dir data/pheno_store --cxr_metadata data/mimic-cxr-2.0.0-metadata.csv \
    --image_embeddings data/image_embeddings_full.csv --output_dir results/pheno_store_preferred \
    --image_feature_set dense --fallback_mode clinical --variant_label cxr_dense --seeds "42,43,44,45,46"
python -m haim_experiments.train_phenotyping \
    --root_path data/root --binned_dir data/pheno_store --cxr_metadata data/mimic-cxr-2.0.0-metadata.csv \
    --output_dir results/pheno_store_preferred_noimg  --fallback_mode clinical --variant_label no_cxr \
    --no_image_features --seeds "42,43,44,45,46"
```
Visualizing differences in feature importance and results:
```
python -m haim_experiments.plot_results shap_vs_delta \
    --results_dir_with results/pheno_store_preferred --variant_label_with cxr_dense \
    --results_dir_without results/pheno_store_preferred_noimg --variant_label_without no_cxr \
    --include_auprc --combine_metrics --output_path results/fig_shap_vs_delta_pheno.pdf
```
## Thanks
We wish to thank the contributors of the repository [MedFuse](https://github.com/nyuad-cai/MedFuse?tab=readme-ov-file), from which we adapted the mimic4process scripts, which are in turn based upon the repository [mimic3-benchmarks](https://github.com/YerevaNN/mimic3-benchmarks/), which we also wish to acknowledge. We also wish to tank the contributors of the repository [HAIM](https://github.com/lrsoenksen/HAIM), from which we derived our experiments.
