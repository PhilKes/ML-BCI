"""
IGNORE
Python script for miscellaneous testing of libraries
"""
import math
from typing import List

import mne
import numpy as np

import torch
from scipy import io
from config import ROOT
from data.datasets.lsmr21.lsmr21_data_loading import LSMRSubjectRun
from data.datasets.phys.phys_data_loading import PHYSDataLoader
from util.misc import copy_attrs, to_el_list, print_counts

print(F"Torch version:\t{torch.__version__}")
print(F"Cuda available:\t{torch.cuda.is_available()},\t{torch.cuda.device_count()} Devices found. ")
print(F"Current Device:\t{torch.cuda.get_device_name(0)}\t(Device {torch.cuda.current_device()})")



mne.set_log_level('WARNING')

dev = None
if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device("cpu")


# scale = lambda x: x * 10000
# ds_train = TrialsDataset(ALL_SUBJECTS, 4, device)
# loader_train = DataLoader(ds_train, 32, sampler=SequentialSampler(ds_train), pin_memory=False, )


# data, labels = next(iter(loader_train))
# print(data.shape, labels.shape)
# print("mean", data.mean(), "std", data.std())

def print_data(loader):
    for data, labels in loader:
        if not torch.any(data.isfinite()):
            print("Not finite data", data)
        if torch.any(data.isnan()):
            print("Nan found ")
        if torch.any(data.isinf()):
            print("Nan found ")


def get_mean_std(loader):
    # var[X] = E[X**2] - E[X]**2
    channels_sum, channels_square_sum, num_batches = 0, 0, 0
    for data, _ in loader:
        # print("data", data.shape)
        # [Batch_size,1,Samples,Channels]
        # data.shape = [32,1,1281,64]

        channels_sum += torch.mean(data, dim=[0, 1, 2, 3])
        channels_square_sum += torch.mean(data ** 2, dim=[0, 1, 2, 3])
        num_batches += 1
    mean = channels_sum / num_batches
    std = (channels_square_sum / num_batches - mean ** 2) ** 0.5
    return mean, std


# print_data(loader_train)

# mean, std = get_mean_std(loader_train)
# print("mean", mean.shape, "std", std.shape)
# print("mean", mean, "std", std)
def check_bad_data(subjects, n_classes):
    min, max = math.inf, -math.inf
    for idx, i in enumerate(subjects):
        data, labels = PHYSDataLoader.load_n_classes_tasks(i, n_classes)
        if np.isneginf(data).any():
            print("negative infinite data")
        if np.isnan(data).any():
            print("Nan found ")
        if np.isinf(data).any():
            print("Not finite data")
        print(f"{i:3d}", " X", data.shape, "y", labels.shape)
        loc_min = data.min()
        loc_max = data.max()
        if (loc_min < min):
            min = loc_min
        if (loc_max > max):
            max = loc_max
    print("Min", min, "Max", max)



from config import datasets_folder
import time

from data.datasets.lsmr21.lmsr_21_dataset import LSMR21





import pandas as pd

if __name__ == '__main__':
    start = time.time()
    s1 = LSMRSubjectRun(1, load_subject_run(0, 0))
    s46 = LSMRSubjectRun(46, load_subject_run(45, 10))
    # print(time.time() - start)
    #
    # # metadata=x['metadata'][0, 0][0, 0]
    # # metadata= LSMRMetadata(metadata)
    # # print(metadata)
    #
    # path = f"{datasets_folder}/{LSMR21.short_name}/numpy/S1_Session_1"
    #
    # # start = time.time()
    # ds.to_npz(path)
    # # print(time.time() - start)
    #
    # start = time.time()
    # LSMRSubjectRun.from_npz(path)
    # print(time.time() - start)
    # data = x['TrialData'][0, 0]
    # print(data)
    s1.print_trials_with_min_mi_time()
    s46.print_trials_with_min_mi_time()
