"""Ablation: the admission's 25 benchmark phenotype groups (HCUP CCS from the ICD discharge codes, same definition
as the labels in create_phenotyping.py) as icd_* input features."""

import os

import pandas as pd
import yaml


DEFINITIONS = os.path.join(os.path.dirname(__file__), '..', 'mimic3benchmark', 'resources',
                           'icd_9_10_definitions_2.yaml')


def benchmark_phenotypes(definitions=DEFINITIONS):
    with open(definitions) as f:
        d = yaml.safe_load(f)
    return sorted(k for k, v in d.items() if v['use_in_benchmark'])


def add_phenotype_features(X, meta, root_physical_dir, phenotypes=None):
    phenotypes = phenotypes or benchmark_phenotypes()
    rows, cached_patient, dx = [], None, None
    for stay, stay_id in zip(meta['stay'], meta['stay_id']):
        patient = stay.split('_')[0]
        if patient != cached_patient:
            dx = pd.read_csv(os.path.join(root_physical_dir, patient, 'diagnoses.csv'), dtype={'icd_code': str})
            cached_patient = patient
        present = set(dx.loc[(dx.stay_id == stay_id) & (dx.USE_IN_BENCHMARK == 1), 'HCUP_CCS_2015'])
        rows.append([int(p in present) for p in phenotypes])
    feats = pd.DataFrame(rows, columns=['icd_' + p for p in phenotypes])
    return pd.concat([X.reset_index(drop=True), feats], axis=1)
