"""
Handles all EEG-Data loading of Physionet Motor Imagery Dataset via MNE Library
(https://neuro.inf.unibe.ch/AlgorithmsNeuroscience/Tutorial_files/DataLoading.html)
On initial Run MNE downloads the Physionet Dataset into ./data/datasets
(https://physionet.org/content/eegmmidb/1.0.0/)
"""
import collections
import math

import mne
import numpy as np
import torch  # noqa
import torch.nn.functional as F  # noqa
import torch.optim as optim  # noqa
from mne import Epochs
from mne.io import concatenate_raws, read_raw_edf
from torch import nn, Tensor  # noqa
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler, Subset  # noqa
from torch.utils.data.dataset import ConcatDataset as _ConcatDataset, TensorDataset, random_split  # noqa
from tqdm import tqdm

from config import VERBOSE, eeg_config, datasets_folder, DATA_PRELOAD, BATCH_SIZE, \
    global_config
from data.data_utils import dec_label, increase_label, normalize_data, get_trials_size, n_classes_tasks, \
    get_equal_trials_per_class, split_trials, get_runs_of_n_classes
from data.physionet_dataset import runs, mne_dataset, ALL_SUBJECTS, MNE_CHANNELS, TRIALS_PER_SUBJECT_RUN, DEFAULTS
from util.misc import print_subjects_ranges, split_np_into_chunks, unified_shuffle_arr, print_numpy_counts


# Returns Loaders of Training + Test Datasets from index splits
# for n_class classification
# also returns Validtion Loader containing validation_subjects subject for loss calculation
def create_loaders_from_splits(splits, validation_subjects, n_class, device, preloaded_data=None,
                               preloaded_labels=None, bs=BATCH_SIZE, ch_names=MNE_CHANNELS,
                               equal_trials=True, used_subjects=ALL_SUBJECTS):
    subjects_train_idxs, subjects_test_idxs = splits
    subjects_train = [used_subjects[idx] for idx in subjects_train_idxs]
    subjects_test = [used_subjects[idx] for idx in subjects_test_idxs]
    print_subjects_ranges(subjects_train, subjects_test)
    validation_loader = None
    if len(validation_subjects) > 0:
        validation_loader = create_loader_from_subjects(validation_subjects, n_class, device, preloaded_data,
                                                        preloaded_labels, bs, ch_names, equal_trials)
    return create_loader_from_subjects(subjects_train, n_class, device,
                                       preloaded_data, preloaded_labels, bs, ch_names, equal_trials), \
           create_loader_from_subjects(subjects_test, n_class, device,
                                       preloaded_data, preloaded_labels, bs, ch_names, equal_trials), \
           validation_loader


# Creates DataLoader with Random Sampling from subject list
def create_loader_from_subjects(subjects, n_class, device, preloaded_data=None,
                                preloaded_labels=None, bs=BATCH_SIZE, ch_names=MNE_CHANNELS, equal_trials=True):
    ds_train = TrialsDataset(subjects, n_class, device,
                             preloaded_tuple=(preloaded_data, preloaded_labels) if DATA_PRELOAD else None,
                             ch_names=ch_names, equal_trials=equal_trials)
    # Sample the trials in random order
    sampler_train = RandomSampler(ds_train)
    return DataLoader(ds_train, bs, sampler=sampler_train, pin_memory=False)


# Returns Train/Test Loaders containing all n_class Runs of subject
# n_test_runs specifies how many Runs are reserved for Testing
# 2/3class: 3 Runs, 4class: 6 Runs
def create_n_class_loaders_from_subject(subject, n_class, n_test_runs, batch_size, ch_names, device):
    n_class_runs = get_runs_of_n_classes(n_class)
    train_runs = n_class_runs[:-n_test_runs]
    test_runs = n_class_runs[-n_test_runs:]
    loader_train = create_loader_from_subject_runs(subject, n_class, batch_size, ch_names, device,
                                                   ignored_runs=test_runs)
    loader_test = create_loader_from_subject_runs(subject, n_class, batch_size, ch_names, device,
                                                  ignored_runs=train_runs)
    return loader_train, loader_test


# Creates Loader containing all n_class Runs of subject
# ignored_runs[] will not be loaded
def create_loader_from_subject_runs(subject, n_class, batch_size, ch_names, device,
                                    ignored_runs=[]):
    preloaded_data, preloaded_labels = load_subjects_data([subject], n_class, ch_names, ignored_runs=ignored_runs)
    preloaded_data = preloaded_data.reshape((preloaded_data.shape[1], 1, preloaded_data.shape[2],
                                             preloaded_data.shape[3]))
    preloaded_labels = preloaded_labels.reshape(preloaded_labels.shape[1])
    data_set = TensorDataset(torch.as_tensor(preloaded_data, device=device, dtype=torch.float32),
                             torch.as_tensor(preloaded_labels, device=device, dtype=torch.int))
    loader_data = DataLoader(data_set, batch_size, sampler=RandomSampler(data_set), pin_memory=False)
    return loader_data


def create_preloaded_loader(subjects, n_class, ch_names, batch_size, device, equal_trials=True):
    print(f"Preloading Subjects [{subjects[0]}-{subjects[-1]}] Data in memory")
    preloaded_data, preloaded_labels = load_subjects_data(subjects, n_class, ch_names,
                                                          equal_trials=equal_trials)
    return create_loader_from_subjects(subjects, n_class, device, preloaded_data,
                                       preloaded_labels, batch_size, equal_trials=equal_trials)


# Loads all Subjects Data + Labels for n_class Classification
# Very high memory usage for ALL_SUBJECTS (~2GB)
# used_runs can be passed to force to load only these runs
def load_subjects_data(subjects, n_class, ch_names=MNE_CHANNELS, equal_trials=True,
                       normalize=False, ignored_runs=[]):
    subjects.sort()
    trials = get_trials_size(n_class, equal_trials, ignored_runs)
    trials_per_run_class = np.math.floor(trials / n_class)
    trials = trials * eeg_config.TRIALS_SLICES
    if n_class > 2:
        trials -= DEFAULTS.REST_TRIALS_LESS

    preloaded_data = np.zeros((len(subjects), trials, len(ch_names), eeg_config.SAMPLES), dtype=np.float32)
    preloaded_labels = np.zeros((len(subjects), trials,), dtype=np.int)
    print("Preload Shape", preloaded_data.shape)
    for i, subject in tqdm(enumerate(subjects), total=len(subjects)):
        data, labels = load_n_classes_tasks(subject, n_class, ch_names, equal_trials, trials_per_run_class,
                                            ignored_runs)
        # if data.shape[0] > preloaded_data.shape[1]:
        #     data, labels = data[:preloaded_data.shape[1]], labels[:preloaded_labels.shape[1]]
        if eeg_config.TRIALS_SLICES > 1:
            data, labels = split_trials(data, labels, eeg_config.TRIALS_SLICES, eeg_config.SAMPLES)
        preloaded_data[i] = data
        preloaded_labels[i] = labels
    if normalize:
        preloaded_data = normalize_data(preloaded_data)
    print("Trials per class loaded:")
    print_numpy_counts(preloaded_labels)
    # print(collections.Counter(preloaded_labels))
    return preloaded_data, preloaded_labels


# Loads corresponding tasks for n_classes Classification
def load_n_classes_tasks(subject, n_classes, ch_names=MNE_CHANNELS, equal_trials=True,
                         trials_per_run_class=TRIALS_PER_SUBJECT_RUN,
                         ignored_runs=[]):
    tasks = n_classes_tasks[n_classes].copy()
    if (not DEFAULTS.REST_TRIALS_FROM_BASELINE_RUN) & (0 in tasks):
        tasks.remove(0)
    data, labels = load_task_runs(subject, tasks,
                                  exclude_bothfists=(n_classes == 4), exclude_rests=(n_classes == 2),
                                  ch_names=ch_names, ignored_runs=ignored_runs,
                                  equal_trials=equal_trials, trials_per_run_class=trials_per_run_class,
                                  n_class=n_classes)
    if n_classes == 2:
        labels = dec_label(labels)
    # TODO? if REST_TRIALS_FROM_BASELINE_RUN:
    #     np.random.seed(39)
    #     data, labels = unified_shuffle_arr(data, labels)
    return data, labels


event_dict = {'T0': 1, 'T1': 2, 'T2': 3}


# Loads Rest trials from the 1st baseline run of subject
# if baseline run is not long enough for all needed trials
# random Trials are generated from baseline run
def mne_load_rests(subject, trials, ch_names, samples):
    used_trials = trials - DEFAULTS.REST_TRIALS_LESS
    X, y = mne_load_subject(subject, 1, tmin=0, tmax=60, event_id='auto', ch_names=ch_names)
    X = np.swapaxes(X, 2, 1)
    chs = len(ch_names)
    if X.shape[0] > 1:
        X = X[:1, :, :]
    X = np.squeeze(X, axis=0)
    X_cop = np.array(X, copy=True)
    X = split_np_into_chunks(X, samples)

    trials_diff = used_trials - X.shape[0]
    if trials_diff > 0:
        for m in range(trials_diff):
            np.random.seed(m)
            rand_start_idx = np.random.randint(0, X_cop.shape[0] - samples)
            # print("rand_start", rand_start_idx)
            rand_x = np.zeros((1, samples, chs))
            rand_x[0] = X_cop[rand_start_idx: (rand_start_idx + samples)]
            X = np.concatenate((X, rand_x))
    elif trials_diff < 0:
        X = X[:trials_diff]
    y = np.full(X.shape[0], y[0])
    # print("X", X.shape, "Y", y)
    X = np.swapaxes(X, 2, 1)
    return X, y


# Merges runs from different tasks + correcting labels for n_class classification
def load_task_runs(subject, tasks, exclude_bothfists=False, ch_names=MNE_CHANNELS, n_class=3,
                   equal_trials=True, trials_per_run_class=TRIALS_PER_SUBJECT_RUN, exclude_rests=False,
                   ignored_runs=[]):
    load_samples = eeg_config.SAMPLES * eeg_config.TRIALS_SLICES
    all_data = np.zeros((0, len(ch_names), load_samples))
    all_labels = np.zeros((0), dtype=np.int)
    # Load Subject Data of all Tasks
    for task_idx, task in enumerate(tasks):
        # Task = 0 -> Rest Trials "T0"
        if DEFAULTS.REST_TRIALS_FROM_BASELINE_RUN & (task == 0):
            data, labels = mne_load_rests(subject, trials_per_run_class, ch_names, load_samples)
        else:
            # if Rest Trials are loaded from Baseline Run, ignore "TO"s in all other Runs
            # exclude_rests is True for 2class Classification
            if DEFAULTS.REST_TRIALS_FROM_BASELINE_RUN | exclude_rests:
                tasks_event_dict = {'T1': 2, 'T2': 3}
            else:
                tasks_event_dict = {'T0': 1, 'T1': 2, 'T2': 3}
            # for 4class classification exclude both fists event of task 4 ("T1")
            if exclude_bothfists & (task == 4):
                tasks_event_dict = {'T2': 2}
            used_runs = [run for run in runs[task] if run not in ignored_runs]
            data, labels = mne_load_subject(subject, used_runs, event_id=tasks_event_dict, ch_names=ch_names)
            # Ensure equal amount of trials per class
            if equal_trials:
                classes = n_class
                if n_class == 2:
                    classes = 3
                data, labels = get_equal_trials_per_class(data, labels, classes, trials_per_run_class)
            # Correct labels if multiple tasks are loaded
            # e.g. in Task 2: "1": left fist, in Task 4: "1": both fists
            contains_rest_task = (0 in tasks)
            for n in range(task_idx if (not contains_rest_task) else task_idx - 1):
                labels = increase_label(labels)
        all_data = np.concatenate((all_data, data))
        all_labels = np.concatenate((all_labels, labels))
    # all_data, all_labels = unison_shuffled_copies(all_data, all_labels)
    return all_data, all_labels


# Loads single Subject of Physionet Data with MNE
# returns EEG data (X) and corresponding Labels (y)
# event_id specifies which event types should be loaded,
# if some are missing, they are ignored
# event_id= 'auto' loads all event types
# ch_names: List of Channel Names to be used (see config.py MNE_CHANNELS)
# tmin,tmax define what time interval of the events is returned
def mne_load_subject(subject, runs, event_id='auto', ch_names=MNE_CHANNELS, tmin=None,
                     tmax=None):
    if tmax is None:
        tmax = eeg_config.EEG_TMAX
    if tmin is None:
        tmin = eeg_config.EEG_TMIN
    raw = mne_load_subject_raw(subject, runs)

    events, event_ids = mne.events_from_annotations(raw, event_id=event_id)
    picks = mne.pick_channels(raw.info['ch_names'], ch_names)

    epochs = Epochs(raw, events, event_ids, tmin, tmax - (1 / eeg_config.SAMPLERATE), picks=picks,
                    baseline=None, preload=True)
    # [trials, channels, timepoints,]
    subject_data = epochs.get_data().astype('float32')
    # Labels (0-index based)
    subject_labels = epochs.events[:, -1] - 1
    return subject_data, subject_labels


# Loads raw Subject run with specified channels
# Can apply Bandpassfilter + Notch Filter
def mne_load_subject_raw(subject, runs, ch_names=MNE_CHANNELS, notch=False,
                         fmin=global_config.FREQ_FILTER_HIGHPASS, fmax=global_config.FREQ_FILTER_LOWPASS):
    if VERBOSE:
        print(f"MNE loading Subject {subject} Runs {runs}")
    raw_fnames = mne_dataset.load_data(subject, runs, datasets_folder)
    raw_files = [read_raw_edf(f, preload=True) for f in raw_fnames]
    raw = concatenate_raws(raw_files)
    raw.rename_channels(lambda x: x.strip('.'))
    if notch:
        picks = mne.pick_channels(raw.info['ch_names'], ch_names)
        raw.notch_filter(60.0, picks=picks, filter_length='auto',
                         phase='zero')
    if ((fmin is not None) | (fmax is not None)):
        # If method=”iir”, 4th order Butterworth will be used
        raw.filter(fmin, fmax, method='iir')
    return raw


# Dataset for EEG Trials Data (divided by subjects)
class TrialsDataset(Dataset):

    def __init__(self, subjects, n_classes, device, preloaded_tuple=None,
                 ch_names=MNE_CHANNELS, equal_trials=True, used_subjects=ALL_SUBJECTS):
        self.subjects = subjects
        # Buffers for last loaded Subject data+labels
        self.loaded_subject = -1
        self.loaded_subject_data = None
        self.loaded_subject_labels = None
        self.n_classes = n_classes
        self.runs = []
        self.device = device
        self.trials_per_subject = get_trials_size(n_classes, equal_trials) * eeg_config.TRIALS_SLICES - DEFAULTS.REST_TRIALS_LESS
        self.equal_trials = equal_trials
        self.preloaded_data = preloaded_tuple[0] if preloaded_tuple is not None else None
        self.preloaded_labels = preloaded_tuple[1] if preloaded_tuple is not None else None
        self.ch_names = ch_names
        self.used_subjects = used_subjects

    # Length of Dataset (all trials)
    def __len__(self):
        return len(self.subjects) * self.trials_per_subject

    # Determines corresponding Subject of trial and loads subject's data+labels
    # Uses buffer for last loaded subject if DATA_PRELOAD = False
    # trial: trial idx
    # returns trial data (X) and trial label (y)
    def load_trial(self, trial):
        local_trial_idx = trial % self.trials_per_subject

        # determine required subject for trial
        subject = self.subjects[int(trial / self.trials_per_subject)]

        # Immediately return from preloaded data if available
        if self.preloaded_data is not None:
            if self.preloaded_data.shape[0] == len(self.used_subjects):
                idx = self.used_subjects.index(subject)
            else:
                idx = self.subjects.index(subject)
            return self.preloaded_data[idx][local_trial_idx], self.preloaded_labels[idx][local_trial_idx]

        # If Subject is in current buffer, skip MNE Loading
        if self.loaded_subject == subject:
            return self.loaded_subject_data[local_trial_idx], self.loaded_subject_labels[local_trial_idx]

        subject_data, subject_labels = load_n_classes_tasks(subject, self.n_classes, ch_names=self.ch_names,
                                                            equal_trials=self.equal_trials)
        # Buffer newly loaded subject
        self.loaded_subject = subject
        self.loaded_subject_data = subject_data
        # BCELoss excepts one-hot encoded, Cross Entropy (used here) not:
        #   labels (0,1,2) to categorical/one-hot encoded: 0 = [1 0 0], 1 =[0 1 0],...
        #   self.loaded_subject_labels = np.eye(self.n_classes, dtype='uint8')[subject_labels]
        #   (https://discuss.pytorch.org/t/cross-entropy-with-one-hot-targets/13580/2)
        self.loaded_subject_labels = subject_labels
        # Return single trial from all Subject's Trials
        X, y = self.loaded_subject_data[local_trial_idx], self.loaded_subject_labels[local_trial_idx]
        return X, y

    # Returns a single trial as Tensor with Labels
    def __getitem__(self, trial):
        X, y = self.load_trial(trial)
        # Shape of 1 Batch (list of multiple __getitem__() calls):
        # [samples (BATCH_SIZE), 1 , Channels (len(ch_names), Timepoints (641)]
        X = torch.as_tensor(X[None, ...], device=self.device, dtype=torch.float32)
        # X = TRANSFORM(X)
        return X, y
