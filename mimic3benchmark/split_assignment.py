from __future__ import absolute_import
from __future__ import print_function

import hashlib
import os

RESOURCES = os.path.join(os.path.dirname(__file__), 'resources')
TEST_BUCKETS = range(0, 20)
VAL_BUCKETS = range(20, 28)


def hash_bucket(subject_id):
    return int(hashlib.sha256('mimic-iv-split:{}'.format(subject_id).encode()).hexdigest(), 16) % 100


def read_split_file(name):
    listed, flagged = set(), set()
    with open(os.path.join(RESOURCES, name)) as f:
        for line in f:
            x, y = line.strip().split(',')
            listed.add(x)
            if int(y) == 1:
                flagged.add(x)
    return listed, flagged


def assign(patients, split_file, buckets):
    listed, flagged = read_split_file(split_file)
    selected = set()
    n_file = n_hash = 0
    for p in patients:
        if p in listed:
            n_file += 1
            if p in flagged:
                selected.add(p)
        else:
            n_hash += 1
            if hash_bucket(p) in buckets:
                selected.add(p)
    return selected, n_file, n_hash
