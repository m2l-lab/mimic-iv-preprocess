
from __future__ import absolute_import
from __future__ import print_function

import argparse
import os

import numpy as np
import pandas as pd


N_HOURS = 48
EPS = 1e-6


def load_base_cohort(root_path):
    stays = pd.read_csv(os.path.join(root_path, 'all_stays.csv'))
    for col in ('intime', 'outtime', 'deathtime', 'dod'):
        stays[col] = pd.to_datetime(stays[col])
    stays['los_hours'] = 24.0 * stays['los']
    death = stays['deathtime'].fillna(stays['dod'])
    stays['hours_to_death'] = (death - stays['intime']).dt.total_seconds() / 3600.0
    return stays


def load_task_stays(task_dir):
    frames = []
    for part in ('train', 'test'):
        fn = os.path.join(task_dir, part, 'listfile.csv')
        if not os.path.exists(fn):
            raise FileNotFoundError('{} not found (run the create_* script first)'.format(fn))
        df = pd.read_csv(fn)
        df['partition'] = part
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _summary(step, df, base_deaths):
    n_deaths = int(df['mortality'].sum())
    return {
        'step': step,
        'stays': len(df),
        'patients': df['subject_id'].nunique(),
        'in_hospital_deaths': n_deaths,
        'mortality_pct': round(100.0 * n_deaths / len(df), 2) if len(df) else np.nan,
        'pct_of_all_deaths': round(100.0 * n_deaths / base_deaths, 2) if base_deaths else np.nan,
    }


def report(stays, task_dir):
    task = load_task_stays(task_dir)
    included_ids = set(task['stay_id'])

    unknown = included_ids - set(stays['stay_id'])
    if unknown:
        print('  WARNING: {} listfile stays are not in all_stays.csv'.format(len(unknown)))

    base_deaths = int(stays['mortality'].sum())
    los_missing = stays['los_hours'].isnull()
    short = ~los_missing & (stays['los_hours'] < N_HOURS - EPS)
    included = stays['stay_id'].isin(included_ids)
    no_data = ~los_missing & ~short & ~included

    if (included & (los_missing | short)).any():
        print('  WARNING: {} stays in the listfile have LOS < 48 h or no LOS'.format(
            int((included & (los_missing | short)).sum())))

    rows = [
        _summary('base cohort (all_stays.csv)', stays, base_deaths),
        _summary('excluded: LOS missing', stays[los_missing], base_deaths),
        _summary('excluded: LOS < 48 h', stays[short], base_deaths),
        _summary('excluded: no events in first 48 h', stays[no_data], base_deaths),
        _summary('analysis cohort (listfile)', stays[included], base_deaths),
    ]

    short_dead = stays[short & (stays['mortality'] == 1)]
    rows.append(_summary('  LOS < 48 h, died in ICU', short_dead[short_dead['mortality_inunit'] == 1], base_deaths))
    rows.append(_summary('  LOS < 48 h, died within 48 h of ICU admission',
                         short_dead[short_dead['hours_to_death'] < N_HOURS], base_deaths))

    pat_included = set(stays.loc[included, 'subject_id'])
    pat_all = set(stays['subject_id'])
    pat_lost = pat_all - pat_included
    lost = stays[stays['subject_id'].isin(pat_lost)]
    lost_dead = lost.groupby('subject_id')['mortality'].max()
    extra = {
        'patients_without_any_stay_in_analysis_cohort': len(pat_lost),
        'of_these_died_in_hospital': int(lost_dead.sum()),
        'mortality_base_cohort_pct': rows[0]['mortality_pct'],
        'mortality_analysis_cohort_pct': rows[4]['mortality_pct'],
    }

    if 'y_true' in task.columns:
        merged = task.merge(stays[['stay_id', 'mortality']], on='stay_id', how='inner')
        n_mismatch = int((merged['y_true'] != merged['mortality']).sum())
        if n_mismatch:
            print('  NOTE: {} stays have y_true != in-hospital mortality '
                  '(expected if the task used --mortality_column Mortality_1yr)'.format(n_mismatch))

    out = pd.DataFrame(rows)
    out.insert(0, 'task_dir', task_dir)
    return out, extra


def main():
    parser = argparse.ArgumentParser(description='Report how 48 h eligibility criterion changes the cohort.')
    parser.add_argument('root_path', help='data/root (contains all_stays.csv)')
    parser.add_argument('task_dirs', nargs='+', help='task datasets, e.g. data/mort_chart data/pheno_store')
    parser.add_argument('--output', default=None, help='optional CSV with all tables')
    args = parser.parse_args()

    stays = load_base_cohort(args.root_path)
    tables = []
    for task_dir in args.task_dirs:
        print('\n=== {} ==='.format(task_dir))
        table, extra = report(stays, task_dir)
        print(table.drop(columns='task_dir').to_string(index=False))
        print('Patients without any stay in the analysis cohort: {} (of whom {} died in hospital)'.format(
            extra['patients_without_any_stay_in_analysis_cohort'], extra['of_these_died_in_hospital']))
        print('In-hospital mortality: base cohort {}% -> analysis cohort {}%'.format(
            extra['mortality_base_cohort_pct'], extra['mortality_analysis_cohort_pct']))
        tables.append(table)

    if args.output:
        out_dir = os.path.dirname(args.output)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir)
        pd.concat(tables, ignore_index=True).to_csv(args.output, index=False)
        print('\nSaved {}'.format(args.output))


if __name__ == '__main__':
    main()
