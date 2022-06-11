"""
Miscellaneous Utility Methods
"""
import errno
import logging
import math
import os
import platform
from typing import List

import numpy as np
import pandas
import pandas as pd
import torch
from scipy import io
from tabulate import tabulate
import functools

from app.paths import chs_names_txt


def print_subjects_ranges(train, test):
    if (train[0] < test[0]) & (train[-1] < test[0]):
        logging.info(f"Subjects for Training:\t[{train[0]}-{train[-1]}]")
    elif (train[0] < test[0]) & (train[-1] > test[0]):
        logging.info(f"Subjects for Training:\t[{train[0]}-{test[0] - 1}],[{test[-1] + 1}-{train[-1]}]")
    elif (train[0] > test[0]):
        logging.info(f"Subjects for Training:\t[{train[0]}-{train[-1]}]")
    logging.info(f"Subjects for Testing:\t[{test[0]}-{test[-1]}]")
    return


def load_chs_of_model(model_path: str):
    """
    Loads list of ch_names from training results folder
    """
    try:
        path = os.path.join(model_path, chs_names_txt)
        chs = np.genfromtxt(path, dtype='str')
    except OSError:
        raise FileNotFoundError("Please specify a valid model path (folder with training results, ch_names.txt,...)")
    return chs


def datetime_to_folder_str(datetime):
    return datetime.strftime("%Y-%m-%d_%H_%M_%S")


str_n_classes = ["", "", "Left/Right", "Left/Right/Down ", "Left/Right/Up/Down"]


def get_str_n_classes(n_classes):
    return f'Classes: {[str_n_classes[i] for i in n_classes]}'


def split_list_into_chunks(p_list: List, chunk_size: int) -> List:
    """
    Split python list into chunks with equal size (last chunk can have smaller size)
    Source: https://stackoverflow.com/questions/24483182/python-split-list-into-n-chunks
    """
    m = int(len(p_list) / int(math.ceil(len(p_list) / chunk_size))) + 1
    return [p_list[i:i + m] for i in range(0, len(p_list), m)]


def split_np_into_chunks(arr: np.ndarray, chunk_size) -> np.ndarray:
    """
    Splits numpy array into chunks with equal size
    """
    chunks = math.ceil(arr.shape[0] / chunk_size)
    arr2 = np.zeros((chunks - 1, chunk_size, arr.shape[1]), dtype=np.float)
    splits = np.split(arr, np.arange(chunk_size, len(arr), chunk_size))
    # Last chunk is of smaller size, so it is skipped
    for i in range(chunks - 1):
        arr2[i] = splits[i]
    return arr2


def list_to_str(list: List) -> str:
    return ','.join([str(i) for i in list])


def unified_shuffle_arr(a: np.ndarray, b: np.ndarray):
    """
    Shuffle 2 arrays unified
    Source: https://stackoverflow.com/a/4602224/9748566
    """
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def print_numpy_counts(arr: np.ndarray):
    """
    Prints counts of all present values in arr
    """
    unique, counts = np.unique(arr, return_counts=True)
    logging.info(dict(zip(unique, counts)))


def makedir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise e
        pass


def groups_labels(size: int, groups: int) -> np.ndarray:
    """
    Returns an array with groups of length (size/groups)
    All elements in one group have the same value
    """
    groups_vals = np.zeros(0, dtype=np.int)
    group_size = math.ceil(size / groups)
    for i in range(groups):
        groups_vals = np.concatenate((groups_vals, np.full(group_size, i)))
    group = 0
    while groups_vals.shape[0] > size:
        groups_vals = np.delete(groups_vals, np.where(groups_vals == group)[0][0])
        group = (group + 1) % groups
    # logging.info(collections.Counter(groups_vals))
    return groups_vals


def get_class_prediction_stats(n_class: int, class_hits):
    """
    Returns amount of Trials per class
    and Accuracies per class
    """
    class_trials, class_accs = np.zeros(n_class), np.zeros(n_class)
    for cl in range(n_class):
        class_trials[cl] = len(class_hits[cl])
        class_accs[cl] = (100 * (sum(class_hits[cl]) / class_trials[cl]))
    return class_trials, class_accs


def get_class_avgs(n_class: int, class_accuracies: np.ndarray):
    """
    Calculate average accuracies per class
    """
    avg_class_accs = np.zeros(n_class)
    for cl in range(n_class):
        avg_class_accs[cl] = np.average(
            [float(class_accuracies[fold][cl]) for fold in range(class_accuracies.shape[0])])
    return avg_class_accs


def file_write(path, data):
    file_result = open(path, "w+")
    file_result.write(data)
    file_result.close()


def get_subdirs(dir: str):
    """
    Get Subdirectories of dir sorted by name
    """
    return sorted([path for path in os.listdir(dir) if os.path.isdir(os.path.join(dir, path))])


def to_el_list(ndarr):
    """
    Converts numpy array to list
    unpacks all inner arrays to single elements
    if shape=1
    :param ndarr: numpy array
    :return: Python List
    """
    l = []
    for el in ndarr:
        if all([s == 1 for s in el.shape]):
            l.append(el.item())
        else:
            l.append(el)
    return l


def copy_attrs(obj_to, arr_from):
    """
    Copies all values of arr_from to obj_to
    Names of attribute name are given in
    arr_from.dtype.names
    Unpacks unnecessary inner arrays if shape=1
    """
    for i in range(len(arr_from.dtype.names)):
        data = arr_from[i]
        if all([s == 1 for s in data.shape]):
            data = data.item()
        elif data.shape[0] == 1:
            data = data[0]
        label = arr_from.dtype.names[i]
        setattr(obj_to, str.lower(label), data)


def print_counts(list):
    """
    Counts element occurences in list and prints as Table
    """
    print_pretty_table(counts_of_list(list))
    logging.info("\n")


def counts_of_list(list):
    """
    Returns a list of element occurences in list
    """
    df = pd.DataFrame(list, columns=['data'])
    return df['data'].value_counts(dropna=False)


def save_dataframe(df, save_path):
    """
    Stores pandas.DataFrame df to save_path
    """
    with open(save_path, 'w') as outfile:
        df.to_string(outfile)


def to_idxs_of_list_str(elements: List[str], list: List[str]):
    """
    Returns list of elements' indexes in string List
    """
    list_upper = [e.upper() for e in list]
    return [list_upper.index(el.upper()) for el in elements]


def to_idxs_of_list(elements: List[any], list: List[any]):
    """
    Returns list of elements' indexes in list
    """
    return [list.index(el) for el in elements]


def print_pretty_table(dataframe: pandas.DataFrame):
    """
    Prints pretty formatted Table (DataFrame) in the Console
    """
    logging.info(tabulate(dataframe, headers='keys', tablefmt='fancy_grid'))


def load_matlab(filename):
    return io.loadmat(filename)['BCI']


def calc_n_samples(tmin: float, tmax: float, samplerate: float):
    return int((tmax - tmin) * samplerate)


def combine_dims(a, i=0, n=1):
    """
    Combines dimensions of numpy array `a`,
    starting at index `i`,
    and combining `n` dimensions
    Source: https://stackoverflow.com/a/46924437/9748566
    """
    s = list(a.shape)
    combined = functools.reduce(lambda x, y: x * y, s[i:i + n + 1])
    return np.reshape(a, s[:i] + [combined] + s[i + n + 1:])


def get_device_name(device: torch.types.Device):
    """
    Get Name from torch if CUDA Device
    """
    if 'cuda' in str(device):
        return f"CUDA ({torch.cuda.get_device_name(device)})"
    return f"CPU ({platform.processor()})"

def str_replace(s, newstring, index, nofail=False):
    # raise an error if index is outside of the string
    if not nofail and index not in range(len(s)):
        raise ValueError("index outside given string")

    # if not erroring, but the index is still not in the correct range..
    if index < 0:  # add it to the beginning
        return newstring + s
    if index > len(s):  # add it to the end
        return s + newstring

    # insert the new string between "slices" of the original
    return s[:index] + newstring + s[index + 1:]