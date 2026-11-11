from __future__ import absolute_import
from __future__ import print_function

import os
import shutil
import argparse

from mimic3benchmark.split_assignment import assign, TEST_BUCKETS


def move_to_partition(args, patients, partition):
    if not os.path.exists(os.path.join(args.subjects_root_path, partition)):
        os.mkdir(os.path.join(args.subjects_root_path, partition))
    for patient in patients:
        src = os.path.join(args.subjects_root_path, patient)
        dest = os.path.join(args.subjects_root_path, partition, patient)
        # replace instead of nesting into an existing folder on rerun
        if os.path.exists(dest):
            shutil.rmtree(dest)
        shutil.move(src, dest)


def remove_stale_patients(args, keep, partition):
    part_dir = os.path.join(args.subjects_root_path, partition)
    if not os.path.isdir(part_dir):
        return 0
    stale = [p for p in filter(str.isdigit, os.listdir(part_dir)) if p not in keep]
    for patient in stale:
        shutil.rmtree(os.path.join(part_dir, patient))
    return len(stale)


def main():
    parser = argparse.ArgumentParser(description='Split data into train and test sets.')
    parser.add_argument('subjects_root_path', type=str, help='Directory containing subject sub-directories.')
    args, _ = parser.parse_known_args()

    folders = list(filter(str.isdigit, os.listdir(args.subjects_root_path)))
    test_set, n_file, n_hash = assign(folders, 'testset_iv.csv', TEST_BUCKETS)
    train_patients = [x for x in folders if x not in test_set]
    test_patients = [x for x in folders if x in test_set]
    print('patients: {} train, {} test | assigned by testset_iv.csv: {}, by subject_id hash: {}'.format(
        len(train_patients), len(test_patients), n_file, n_hash))

    # a fresh extraction is the complete cohort: drop patients left over from earlier runs
    if folders:
        n_train = remove_stale_patients(args, set(train_patients), "train")
        n_test = remove_stale_patients(args, set(test_patients), "test")
        print('removed patients not in the new extraction: {} from train, {} from test'.format(n_train, n_test))

    move_to_partition(args, train_patients, "train")
    move_to_partition(args, test_patients, "test")


if __name__ == '__main__':
    main()

