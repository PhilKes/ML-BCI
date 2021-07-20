"""
IGNORE
Python script for miscellaneous testing of libraries
"""
import math
import os
from typing import List

import mne
import numpy as np

import torch
from scipy import io
from sympy import pretty_print
from torch.utils.data import BatchSampler, SequentialSampler, SubsetRandomSampler

from config import ROOT, set_eeg_artifacts_trial_category
from data.datasets.lsmr21.lsmr21_data_loading import LSMRSubjectRun, LSMR21DataLoader, LSMRNumpyRun
from data.datasets.phys.phys_data_loading import PHYSDataLoader
from machine_learning.util import SubjectTrialsRandomSampler, get_valid_trials_per_subject
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


import time
from config import datasets_folder
from data.datasets.lsmr21.lmsr_21_dataset import LSMR21


def yield_stuff(subjects, trials_per_subject):
    trials = np.arange(subjects * trials_per_subject)
    trials = np.split(trials, subjects)
    np.random.seed(42)
    for subject_trials in trials:
        np.random.shuffle(subject_trials)
        for trial in subject_trials:
            yield trial


def matlab_to_numpy(subjects, runs):
    """
    Convert Matlab LSMR Subject Run Files to numpy files
    """
    for i, subject in enumerate(subjects):
        for run in runs:
            npz_path = f"{datasets_folder}/{LSMR21.short_name}/numpy/S{subject}_Session_{run}.npz"
            if os.path.isfile(npz_path):
                continue
            s = LSMRSubjectRun(subject, LSMR21DataLoader.load_subject_run(subject, run))
            s.to_npz(npz_path)


import pandas as pd
from util.misc import print_pretty_table


def print_trials_per_tmin(subjects, runs, mi_tmins=np.arange(4, 11, 1)):
    """
    Prints Table of Trials per Subject Run with minimum Sample Size (Tmin)
    :param mi_tmins: List of Tmins the Trials have to have
    """
    subject_trials = []
    row_labels = []
    for i, subject in enumerate(subjects):
        for run in runs:
            s = LSMR21DataLoader.load_subject_run(subject, run)
            subject_trials.append(s.get_trials_tmin())
            row_labels.append(f"S{subject:03d} R{run:02d}")
    df = pd.DataFrame(subject_trials, columns=[f"{tmin:.2f}s" for tmin in mi_tmins], index=row_labels)
    print(f"--- Available Trials of Subject {subjects} of Runs {runs} ---")
    print_pretty_table(df)


if __name__ == '__main__':
    # start = time.time()
    subjects = [1, 26, 46]
    #
    runs = [1, 11]
    # print_trials_per_tmin(subjects, runs)

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

    # subjects = 2
    # trials_per_subject = 10
    # print(list(SubjectTrialsRandomSampler(subjects, trials_per_subject)))
    # print(list(yield_stuff(subjects,trials_per_subject)))
    # data = np.asarray(
    #     [
    #         [
    #             [0, 1, 2], [3, 4, 5]
    #         ],
    #         [
    #             [6, 7, 8], [9, 10, 11]
    #         ],
    #     ])
    # print(data[:,:,::3])
    # data = np.resize(data, (data.shape[0], data.shape[1], 2))
    # data = np.vstack(data[:, :, :]).astype(np.float)

    # r = LSMRNumpyRun.from_npz(np.load("/opt/datasets/LSMR21/numpy/S1_Session_1.npz",allow_pickle=True))
    # r.print_trials_with_min_mi_time()
    def load_sub():
        x, y = LSMR21DataLoader.load_subject(0, 2, LSMR21.CHANNELS, n_trials_max=2000)
        y = np.expand_dims(y, 0)
        trials = get_valid_trials_per_subject(y, [0], [0], 2000)
        print(trials)


    LSMR21.runs = [7]
    for i in range(2):
        for j in range(3):
            print(f"artifacts= {i} + Trial Cat. {j}")
            set_eeg_artifacts_trial_category(i, j)
            load_sub()
