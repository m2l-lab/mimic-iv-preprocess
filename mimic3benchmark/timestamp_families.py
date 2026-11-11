"""Timestamp per feature family for the hybrid variant: storetime for labs, charttime for bedside items."""

from __future__ import absolute_import
from __future__ import print_function

import pandas as pd


FAMILY_TIMESTAMP = {
    'lab': 'storetime',
    'charted_lab': 'storetime',
    'bedside': 'charttime',
}

CHARTED_LAB_CATEGORIES = {'Labs', 'ABG', "ABG's"}

ITEMID_FAMILY_OVERRIDES = {}  


def _family_for_row(linksto, category):
    linksto = str(linksto).strip().lower()
    category = str(category).strip()
    if linksto == 'labevents':
        return 'lab'
    if linksto == 'chartevents':
        return 'charted_lab' if category in CHARTED_LAB_CATEGORIES else 'bedside'
    raise ValueError("No timestamp family rule for source table '{}'.".format(linksto))


def load_itemid_families(variable_map_file, variable_column='LEVEL2'):
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
    return set(families.loc[families.timestamp == timestamp, 'itemid'].astype(int))


def print_family_table(families):
    print('Timestamp families (hybrid variant):')
    for var, g in families.groupby('variable'):
        parts = ['{}: {} ({})'.format(ts, fam, len(gg))
                 for (fam, ts), gg in g.groupby(['family', 'timestamp'])]
        print('  {:40s} {}'.format(var, '; '.join(parts)))
