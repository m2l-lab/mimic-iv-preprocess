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
python -m mimic3models.split_train_val data/mort_store
python -m mimic3models.split_train_val data/mort_chart
python -m mimic3models.split_train_val data/pheno_store
```

For the phenotype experiment, we also preprocess the CXR images:
```
python resize_images.py
python -m haim_experiments.image_embeddings --cxr_metadata data/mimic-cxr-2.0.0-metadata.csv --all_stays data/root/all_stays.csv --image_dir data/resized --output_csv data/image_embeddings_full.csv --weights densenet121-res224-chex --include_predictions

```


## Thanks
We wish to thank the contributors of the repository [MedFuse](https://github.com/nyuad-cai/MedFuse?tab=readme-ov-file), from which we adapted the mimic4process scripts, which are in turn based upon the repository [mimic3-benchmarks](https://github.com/YerevaNN/mimic3-benchmarks/), which we also wish to acknowledge. We also wish to tank the contributors of the repository [HAIM](https://github.com/lrsoenksen/HAIM), from which we derived our experiments.
