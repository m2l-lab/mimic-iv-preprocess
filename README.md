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
Start by running the following commands, changing the paths as needed:
```
python -m mimic3benchmark.scripts.extract_subjects_iv /physionet/files/mimiciv data/root/
python -m mimic3benchmark.scripts.validate_events data/root/
python -m mimic3benchmark.scripts.extract_episodes_from_subjects data/root/
python -m mimic3benchmark.scripts.split_train_and_test data/root/
```

## Thanks
We wish to thank the contributors of the repository [MedFuse](https://github.com/nyuad-cai/MedFuse?tab=readme-ov-file), from which we adapted the mimic4process scripts, which are in turn based upon the repository [mimic3-benchmarks](https://github.com/YerevaNN/mimic3-benchmarks/), which we also wish to acknowledge. We also wish to tank the contributors of the repository [HAIM](https://github.com/lrsoenksen/HAIM), from which we derived our experiments.
