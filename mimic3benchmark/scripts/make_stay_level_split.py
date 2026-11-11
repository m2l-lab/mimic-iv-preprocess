"""Ablation: random stay-level split instead of the patient-level split (stays of one patient can end up in
train and test). Writes a parallel task folder and root folder of symlinks with the same partition sizes;
train with --binned_dir <out_binned> --root_path <out_root>.

python -m mimic3benchmark.scripts.make_stay_level_split data/root data/mort_chart data/root_staysplit data/mort_chart_staysplit
"""

from __future__ import absolute_import
from __future__ import print_function

import argparse
import os

import numpy as np
import pandas as pd


PARTS = ('train', 'val', 'test')


def _link(src, dst):
    if os.path.lexists(dst):
        os.remove(dst)
    os.symlink(os.path.abspath(src), dst)


def main():
    parser = argparse.ArgumentParser(description='Random stay-level split (ablation).')
    parser.add_argument('root_path', help='data/root with train/ and test/ patient folders')
    parser.add_argument('binned_dir', help='task folder after split_train_val, e.g. data/mort_chart')
    parser.add_argument('out_root')
    parser.add_argument('out_binned')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    frames = []
    for part in PARTS:
        df = pd.read_csv(os.path.join(args.binned_dir, '{}_listfile.csv'.format(part)))
        df['_src'] = os.path.join(args.binned_dir, 'test' if part == 'test' else 'train')
        frames.append(df)
    sizes = [len(f) for f in frames]
    pooled = pd.concat(frames, ignore_index=True)
    order = np.random.default_rng(args.seed).permutation(len(pooled))
    bounds = np.cumsum([0] + sizes)
    new = {part: pooled.iloc[order[bounds[i]:bounds[i + 1]]] for i, part in enumerate(PARTS)}

    patient_dir = {}
    for part in ('train', 'test'):
        d = os.path.join(args.root_path, part)
        for p in filter(str.isdigit, os.listdir(d)):
            patient_dir[p] = os.path.join(d, p)

    for sub in ('train', 'test'):
        os.makedirs(os.path.join(args.out_binned, sub), exist_ok=True)
        os.makedirs(os.path.join(args.out_root, sub), exist_ok=True)
    for name in ('all_stays.csv', 'all_diagnoses.csv', 'phenotype_labels.csv'):
        if os.path.exists(os.path.join(args.root_path, name)):
            _link(os.path.join(args.root_path, name), os.path.join(args.out_root, name))

    for part, df in new.items():
        sub = 'test' if part == 'test' else 'train'
        for stay, src in zip(df.stay, df._src):
            _link(os.path.join(src, stay), os.path.join(args.out_binned, sub, stay))
            patient = stay.split('_')[0]
            _link(patient_dir[patient], os.path.join(args.out_root, sub, patient))
        out = df.drop(columns='_src')
        out.to_csv(os.path.join(args.out_binned, '{}_listfile.csv'.format(part)), index=False)
    pd.concat([new['train'], new['val']]).drop(columns='_src').to_csv(
        os.path.join(args.out_binned, 'train', 'listfile.csv'), index=False)
    new['test'].drop(columns='_src').to_csv(os.path.join(args.out_binned, 'test', 'listfile.csv'), index=False)

    pat = {part: set(df.stay.str.split('_').str[0]) for part, df in new.items()}
    test_pat = new['test'].stay.str.split('_').str[0]
    shared = test_pat.isin(pat['train'] | pat['val'])
    print('stays: ' + ', '.join('{} {}'.format(p, len(new[p])) for p in PARTS))
    print('test stays whose patient also has a stay in train/val: {} ({:.1%}); patients shared: {}'.format(
        int(shared.sum()), shared.mean(), len(pat['test'] & (pat['train'] | pat['val']))))


if __name__ == '__main__':
    main()
