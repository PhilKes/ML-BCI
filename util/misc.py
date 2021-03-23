"""
Miscellaneous Utility
"""
import math

import numpy as np
import torch  # noqa
import torch.nn.functional as F  # noqa
import torch.optim as optim  # noqa
from torch import nn, Tensor  # noqa
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler, Subset  # noqa
from torch.utils.data.dataset import ConcatDataset as _ConcatDataset  # noqa

from config import training_results_folder, chs_names_txt


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
    return np.genfromtxt(f"{model_path}/{training_results_folder}/{chs_names_txt}", dtype='str')


def datetime_to_folder_str(datetime):
    return datetime.strftime("%Y-%m-%d %H_%M_%S")


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


def list_to_string(list):
    return ','.join([str(i) for i in list])

# Shuffle 2 arrays unified
def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]