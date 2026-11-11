from __future__ import absolute_import
from __future__ import print_function

import shutil
import argparse
import os
import pandas as pd

from mimic3benchmark.split_assignment import assign, VAL_BUCKETS

def main():
    parser = argparse.ArgumentParser(description="Split train data into train and validation sets.")
    parser.add_argument('dataset_dir', type=str, help='Path to the directory which contains the dataset')
    args, _ = parser.parse_known_args()

    with open(os.path.join(args.dataset_dir, 'train/listfile.csv')) as listfile:
        lines = listfile.readlines()
        header = lines[0]
        lines = lines[1:]

    patients = {x[:x.find("_")] for x in lines}
    val_patients, n_file, n_hash = assign(patients, 'valset_iv.csv', VAL_BUCKETS)
    print('patients: {} train, {} val | assigned by valset_iv.csv: {}, by subject_id hash: {}'.format(
        len(patients) - len(val_patients), len(val_patients), n_file, n_hash))
    train_lines = [x for x in lines if x[:x.find("_")] not in val_patients]
    val_lines = [x for x in lines if x[:x.find("_")] in val_patients]
    assert len(train_lines) + len(val_lines) == len(lines)

    with open(os.path.join(args.dataset_dir, 'train_listfile.csv'), 'w') as train_listfile:
        train_listfile.write(header)
        for line in train_lines:
            train_listfile.write(line)

    with open(os.path.join(args.dataset_dir, 'val_listfile.csv'), 'w') as val_listfile:
        val_listfile.write(header)
        for line in val_lines:
            val_listfile.write(line)

    shutil.copy(os.path.join(args.dataset_dir, 'test/listfile.csv'),
                os.path.join(args.dataset_dir, 'test_listfile.csv'))
    


if __name__ == '__main__':
    
    main()

    
