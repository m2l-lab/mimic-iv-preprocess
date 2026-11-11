# Text-section leakage in reconstructed admission notes

This experiment measures retrospective information in admission sections
reconstructed from MIMIC-IV discharge summaries and its effect on in-hospital
mortality prediction. It includes section extraction, sentence screening,
TF-IDF/logistic-regression baselines, frozen CORe inference, patient-clustered
paired evaluation, and automated screening summaries.

The population is hospital admissions with a usable discharge summary. It is
independent of this repository's ICU cohort: do not run the ICU extraction or
reuse its split files as prerequisites. The original experiment uses MIMIC-IV
and MIMIC-IV-Note 2.2 and patient-disjoint 70/15/15 splits (seed 42).

## Setup and configuration

Use a separate Python 3.12 environment; `requirements.txt` pins this experiment's
dependencies independently of the root `environment.yml`.

```bash
cd experiments/text_section_leakage
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pytest -q tests
```

Tests use synthetic fixtures and do not require MIMIC data or model downloads.

Set `data.mimic_root` in `config.yaml`, or set `TSL_MIMIC_ROOT` to the directory
containing `hosp/` and `note/`. Required files are `hosp/admissions.csv`,
`hosp/patients.csv`, and `note/discharge.csv`. Optional analyses need additional
tables, as described below. The inputs are uncompressed CSV files.

All scripts use `config.yaml` by default. `TSL_CONFIG=/absolute/path/config.yaml`
selects another configuration; relative output paths resolve against that file's
directory. Set `core.device` to `cpu`, `mps`, or `cuda` for your machine. Frozen
CORe inference retrieves the model revision pinned in the configuration and
checks its weight hash.

By default, row-level outputs go to the git-ignored
`artifacts_private/text_section_leakage/` directory at the repository root
(directory permissions 700). Aggregate outputs go to
`reports/text_section_leakage/results/`; generated figures and tables are also
ignored. Never print note text or send it to external services. The checkpoint
download does not require uploading notes.

## Conditions

| Condition | Text |
|---|---|
| `T0_COPIV` | Historical COP-IV regex extraction, preserved as the baseline |
| `T1_ROBUST` | Frozen robust parser over the same eight admission-related sections |
| `T2_SCREENED` | T1 minus sentences flagged by frozen screening rules |
| `T3_RETROSPECTIVE` | T1 plus Discharge Condition, Discharge Diagnosis, and Brief Hospital Course from the same note |
| `T4_OTHERPATIENT` | T1 plus those sections from a different patient in the same split |
| `T1B_ROBUST`, `T2B_SCREENED` | Improved parser and its screened version; T1 remains unchanged |
| `T1_NO_PE`, `T1_NO_HPI`, `T1_PE_LABELLED` | Section ablations for the TF-IDF baseline |
| `T1_ICD` | T1 plus current-admission diagnosis-code tokens |
| `T1_RANDOM_SPLIT` | T1 with an admission-level split; evaluated separately because its test set differs |
| `T1_OVL_A`, `T1_OVL_B` | Patient-overlap controls with a fixed evaluation set |

The paired analysis population is the intersection of admissions retained by
T0 and having a nonempty core section under T1 (`in_analysis`). Every text
condition uses the same selected discharge summary. Rendering uses the shared
COP-IV/CORe section template. CORe reserves 382 admission-text tokens and 128
injection tokens; TF-IDF uses the complete text.

## Main automated workflow

Run these commands from this directory, after configuring the inputs:

```bash
.venv/bin/python build_cohort.py
.venv/bin/python build_variants.py
.venv/bin/python build_t1b.py
.venv/bin/python train_tfidf_baseline.py
.venv/bin/python train_tfidf_baseline.py T1B_ROBUST T2B_SCREENED
.venv/bin/python run_core_inference.py
.venv/bin/python bootstrap_evaluation.py tfidf
.venv/bin/python bootstrap_evaluation.py core
.venv/bin/python summarize_screening.py
.venv/bin/python exposure_analysis.py
```

`cohort.parquet` and `split_manifest.csv` preserve cohort identifiers and split
membership. `variants.parquet`, `variants_t1b.parquet`, and `screening_log.parquet`
contain text variants and their provenance. Prediction files remain private;
evaluation writes aggregate AUROC, AUPRC, Brier scores, and paired differences
with patient-clustered bootstrap confidence intervals. Run memory-intensive
training and inference steps sequentially.

## Optional controls and reference representations

```bash
.venv/bin/python train_tfidf_baseline.py T1_NO_PE T1_NO_HPI T1_PE_LABELLED T1_ICD T1_RANDOM_SPLIT
.venv/bin/python train_tfidf_baseline.py T1_OVL_A T1_OVL_B
.venv/bin/python build_injection_parts.py
.venv/bin/python train_tfidf_baseline.py T3A_DCOND T3B_DDIAG T3C_BHC T3D_BHC_NOOUTCOME T4M_OUTCOMEMATCHED
.venv/bin/python microbiology_leakage.py
.venv/bin/python icd_feature_weights.py
.venv/bin/python icd_decomposition.py
.venv/bin/python bootstrap_evaluation.py tfidf
```

The ICD analyses require `hosp/diagnoses_icd.csv` and
`hosp/d_icd_diagnoses.csv`. Microbiology requires `hosp/microbiologyevents.csv`
and uses its own cohort of admissions alive and in hospital 48 hours after
hospital admission; it compares results available by `storetime` with results
selected by collection time.

`ed_reference.py` is optional and needs a pre-existing Parquet cache under
`<mimic_root>/parquet_cache/<module>/<table>/*.parquet`, with datetime columns
already typed as timestamps. It reads `ed/edstays`, `ed/triage`, `ed/vitalsign`,
`ed/medrecon`, `ed/pyxis`, and `hosp/labevents`. Cache construction is not
included here. Once that cache is available, run:

```bash
.venv/bin/python ed_reference.py
```

## Figures

`make_figures.py` generates aggregate plots and `TABLES.md` after the main
workflow, including `summarize_screening.py`. Some panels and tables require
the optional analyses above. Screening summaries count rule matches; they do
not establish the prevalence of all retrospective content.

## Source and attribution

Imported from the `text_section_leakage` experiment in `medllm-multiagents`.
The experiment keeps the historical COP-IV implementation and attribution in
`copiv_original.py` (ClinicalOutcomePrediction-IV commit
`a1b6708b23149a3ef29c3656ffdf7f7906cf770c`). Model identifiers, revisions, and
experimental settings remain pinned in `config.yaml`. Reports and aggregate
findings from the original study are not imported. See [PROVENANCE.md](PROVENANCE.md)
for the scope of this import.
