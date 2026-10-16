"""Feature families and the timestamp each family is anchored on.

Used by the 'hybrid' variant of extract_episodes_from_subjects.py. The idea:
the timestamp should match the information state of the clinician.

- Bedside measurements (vitals, GCS, FiO2, height/weight) are visible at the
  bedside when they are taken, so charttime is the right anchor. Their
  storetime only records when a nurse validated/charted the value.
- Laboratory results only become available once the result is reported, so
  storetime is the better proxy for when the value was known.

Families are assigned per ITEMID (not per variable), because some variables
mix sources: 'Oxygen saturation' combines pulse oximetry (bedside) with
blood-gas SaO2 (lab), and 'pH' combines labevents with arterial pH charted
in chartevents.

To change the assignment, edit FAMILY_TIMESTAMP (family -> timestamp) or
ITEMID_FAMILY_OVERRIDES (single itemids). print_family_table() shows the
resulting assignment per variable.
"""

from __future__ import absolute_import
from __future__ import print_function

import pandas as pd


# Which timestamp each family uses in the hybrid variant.
FAMILY_TIMESTAMP = {
    'lab': 'storetime',          # labevents
    'charted_lab': 'storetime',  # lab results charted into chartevents (CATEGORY 'Labs' / ABG)
    'bedside': 'charttime',      # all other chartevents (vitals, GCS, FiO2, height, weight, ...)
}

# chartevents CATEGORY values that mark a lab result rather than a bedside
# measurement. In MIMIC-IV (metavision) these are e.g. 220227 'Arterial O2
# Saturation' and 223830 'PH (Arterial)'. The carevue IDs ("ABG", "ABG's")
# are MIMIC-III items and are kept here only for completeness.
CHARTED_LAB_CATEGORIES = {'Labs', 'ABG', "ABG's"}

# Explicit per-itemid overrides, applied after the rules above,
# e.g. {220227: 'bedside'}.
ITEMID_FAMILY_OVERRIDES = {}


def _family_for_row(linksto, category):
    linksto = str(linksto).strip().lower()
    category = str(category).strip()
    if linksto == 'labevents':
        return 'lab'
    if linksto == 'chartevents':
        return 'charted_lab' if category in CHARTED_LAB_CATEGORIES else 'bedside'
    raise ValueError("No timestamp family rule for source table '{}'. Add a rule in "
                     "timestamp_families.py or an entry in ITEMID_FAMILY_OVERRIDES.".format(linksto))


def load_itemid_families(variable_map_file, variable_column='LEVEL2'):
    """Return a DataFrame (itemid, variable, mimic_label, linksto, category,
    family, timestamp) for all itemids the pipeline uses.

    Applies the same row filter as preprocessing.read_itemid_to_variable_map
    (non-empty variable, COUNT > 0, STATUS == 'ready'), so the itemid set is
    identical to the one that ends up in the timeseries.
    """
    var_map = pd.read_csv(variable_map_file).fillna('').astype(str)
    var_map.COUNT = var_map.COUNT.astype(int)
    var_map = var_map[(var_map[variable_column] != '') & (var_map.COUNT > 0)]
    var_map = var_map[var_map.STATUS == 'ready']

    fam = pd.DataFrame({
        'itemid': var_map.ITEMID.astype(int).values,
        'variable': var_map[variable_column].values,
        'mimic_label': var_map['MIMIC LABEL'].values,
        'linksto': var_map.LINKSTO.values,
        'category': var_map.CATEGORY.values,
    })
    fam['family'] = [_family_for_row(l, c) for l, c in zip(fam.linksto, fam.category)]
    for itemid, family in ITEMID_FAMILY_OVERRIDES.items():
        if family not in FAMILY_TIMESTAMP:
            raise ValueError("Unknown family '{}' for itemid {}".format(family, itemid))
        fam.loc[fam.itemid == int(itemid), 'family'] = family
    fam['timestamp'] = fam.family.map(FAMILY_TIMESTAMP)
    return fam.drop_duplicates(subset='itemid').reset_index(drop=True)


def itemids_with_timestamp(families, timestamp):
    """Set of itemids whose family is anchored on `timestamp`."""
    return set(families.loc[families.timestamp == timestamp, 'itemid'].astype(int))


def print_family_table(families):
    """Print the family/timestamp assignment per variable."""
    print('Timestamp families (hybrid variant):')
    for var, g in families.groupby('variable'):
        parts = ['{}: {} ({})'.format(ts, fam, len(gg))
                 for (fam, ts), gg in g.groupby(['family', 'timestamp'])]
        print('  {:40s} {}'.format(var, '; '.join(parts)))
