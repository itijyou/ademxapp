# pylint: skip-file
"""
file iterator for image semantic segmentation
"""
import os

import numpy as np


def parse_split_file(dataset, split, data_root=''):
    split_filename = 'issegm/data/{}/{}.lst'.format(dataset, split)
    image_list = []
    label_list = []
    with open(split_filename) as f:
        for item in f.readlines():
            fields = item.strip().split('\t')
            image_list.append(os.path.join(data_root, fields[0]))
            label_list.append(os.path.join(data_root, fields[1]))
    return image_list, label_list

def make_divisible(v, divider):
    return int(np.ceil(float(v) / divider) * divider)

