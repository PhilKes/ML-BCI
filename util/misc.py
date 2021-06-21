"""
Miscellaneous Utility Methods
"""
import errno
import math
import os

import numpy as np
import torch  # noqa
import torch.nn.functional as F  # noqa
import torch.optim as optim  # noqa
from torch import nn, Tensor  # noqa
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler, Subset  # noqa
from torch.utils.data.dataset import ConcatDataset as _ConcatDataset  # noqa

from config import chs_names_txt


def print_subjects_ranges(train, test):
    if (train[0] < test[0]) & (train[-1] < test[0]):
        print(f"Subjects for Training:\t[{train[0]}-{train[-1]}]")
    elif (train[0] < test[0]) & (train[-1] > test[0]):
        print(f"Subjects for Training:\t[{train[0]}-{test[0] - 1}],[{test[-1] + 1}-{train[-1]}]")
    elif (train[0] > test[0]):
        print(f"Subjects for Training:\t[{train[0]}-{train[-1]}]")
    print(f"Subjects for Testing:\t[{test[0]}-{test[-1]}]")
    return


# Loads list of ch_names from training results folder
def load_chs_of_model(model_path):
    try:
        path = os.path.join(model_path, chs_names_txt)
        chs = np.genfromtxt(path, dtype='str')
    except OSError:
        raise FileNotFoundError("Please specify a valid model path (folder with training results, ch_names.txt,...)")
    return chs


def datetime_to_folder_str(datetime):
    return datetime.strftime("%Y-%m-%d_%H_%M_%S")


str_n_classes = ["", "", "Left/Right Fist", "Left/Right-Fist / Rest", "Left/Right-Fist / Rest / Both-Feet"]


def get_str_n_classes(n_classes):
    return f'Classes: {[str_n_classes[i] for i in n_classes]}'


# Split python list into chunks with equal size (last chunk can have smaller size)
# source: https://stackoverflow.com/questions/24483182/python-split-list-into-n-chunks
def split_list_into_chunks(p_list, chunk_size):
    m = int(len(p_list) / int(math.ceil(len(p_list) / chunk_size))) + 1
    return [p_list[i:i + m] for i in range(0, len(p_list), m)]


def split_np_into_chunks(arr, chunk_size):
    chunks = math.ceil(arr.shape[0] / chunk_size)
    arr2 = np.zeros((chunks - 1, chunk_size, arr.shape[1]), dtype=np.float)
    splits = np.split(arr, np.arange(chunk_size, len(arr), chunk_size))
    # Last chunk is of smaller size, so it is skipped
    for i in range(chunks - 1):
        arr2[i] = splits[i]
    return arr2


def list_to_str(list):
    return ','.join([str(i) for i in list])


# Shuffle 2 arrays unified
# Source: https://stackoverflow.com/a/4602224/9748566
def unified_shuffle_arr(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


# Prints counts of all present values in arr
def print_numpy_counts(arr):
    unique, counts = np.unique(arr, return_counts=True)
    print(dict(zip(unique, counts)))


def makedir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise e
        pass


# Returns an array with groups of length (size/groups)
# All elements in one group have same value
def groups_labels(size, groups):
    groups_vals = np.zeros(0, dtype=np.int)
    group_size = math.ceil(size / groups)
    for i in range(groups):
        groups_vals = np.concatenate((groups_vals, np.full(group_size, i)))
    group = 0
    while groups_vals.shape[0] > size:
        groups_vals = np.delete(groups_vals, np.where(groups_vals == group)[0][0])
        group = (group + 1) % groups
    # print(collections.Counter(groups_vals))
    return groups_vals


# Returns amount of Trials per class
# and Accuracies per class
def get_class_prediction_stats(n_class, class_hits):
    class_trials, class_accs = np.zeros(n_class), np.zeros(n_class)
    for cl in range(n_class):
        class_trials[cl] = len(class_hits[cl])
        class_accs[cl] = (100 * (sum(class_hits[cl]) / class_trials[cl]))
    return class_trials, class_accs

# Calculate average accuracies per class
def get_class_avgs(n_class, class_accuracies):
    avg_class_accs = np.zeros(n_class)
    for cl in range(n_class):
        avg_class_accs[cl] = np.average(
            [float(class_accuracies[fold][cl]) for fold in range(class_accuracies.shape[0])])
    return avg_class_accs