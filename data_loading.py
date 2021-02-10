"""
Handles all EEG-Data loading of Physionet Motor Imagery Dataset via MNE Library
(https://physionet.org/content/eegmmidb/1.0.0/)
(https://neuro.inf.unibe.ch/AlgorithmsNeuroscience/Tutorial_files/DataLoading.html)
"""
import mne
import numpy as np
import torch  # noqa
import torch.nn.functional as F  # noqa
import torch.optim as optim  # noqa
from mne import Epochs, pick_types
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf
from torch import nn, Tensor  # noqa
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler, Subset  # noqa
from torch.utils.data.dataset import ConcatDataset as _ConcatDataset  # noqa
from tqdm import tqdm

from config import VERBOSE, EEG_TMIN, EEG_TMAX, datasets_folder, DATA_PRELOAD, BATCH_SIZE, SAMPLES, CHANNELS
from utils import print_subjects_ranges

excluded_subjects = [88, 92, 100, 104]
ALL_SUBJECTS = [i for i in range(1, 110) if i not in excluded_subjects]

runs_rest = [1]  # Baseline, eyes open
runs_t1 = [3, 7, 11]  # Task 1 (open and close left or right fist)
runs_t2 = [4, 8, 12]  # Task 2 (imagine opening and closing left or right fist)
runs_t3 = [5, 9, 13]  # Task 3 (open and close both fists or both feet)
runs_t4 = [6, 10, 14]  # Task 4 (imagine opening and closing both fists or both feet)

runs = [runs_rest, runs_t1, runs_t2, runs_t3, runs_t4]

trials_for_classes = {2: 42, 3: 84, 4: 147, }


# Dataset for EEG Trials Data (divided by subjects)
class TrialsDataset(Dataset):

    def __init__(self, subjects, n_classes, device, preloaded_tuple=None):
        self.subjects = subjects
        # Buffers for last loaded Subject data+labels
        self.loaded_subject = -1
        self.loaded_subject_data = None
        self.loaded_subject_labels = None
        self.n_classes = n_classes
        self.runs = []
        self.device = device
        self.trials_per_subject = trials_for_classes[n_classes]
        self.preloaded_data = preloaded_tuple[0] if preloaded_tuple is not None else None
        self.preloaded_labels = preloaded_tuple[1] if preloaded_tuple is not None else None

    # Length of Dataset (84 Trials per Subject)
    def __len__(self):
        return len(self.subjects) * self.trials_per_subject

    # Determines corresponding Subject of trial and loads subject's data+labels
    # Uses buffer for last loaded subject
    # trial: trial idx
    # returns trial data (X) and trial label (y)
    def load_trial(self, trial):
        local_trial_idx = trial % self.trials_per_subject

        # determine required subject for trial
        subject = self.subjects[int(trial / self.trials_per_subject)]

        if self.preloaded_data is not None:
            idx = ALL_SUBJECTS.index(subject)
            return self.preloaded_data[idx][local_trial_idx], self.preloaded_labels[idx][local_trial_idx]

        # If Subject is in current buffer, skip MNE Loading
        if self.loaded_subject == subject:
            return self.loaded_subject_data[local_trial_idx], self.loaded_subject_labels[local_trial_idx]

        subject_data, subject_labels = load_n_classes_tasks(subject, self.n_classes)
        # Buffer newly loaded subject
        self.loaded_subject = subject
        self.loaded_subject_data = subject_data
        # BCELoss excepts one-hot encoded, Cross Entropy not:
        #   labels (0,1,2) to categorical/one-hot encoded: 0 = [1 0 0], 1 =[0 1 0],...
        #   self.loaded_subject_labels = np.eye(self.n_classes, dtype='uint8')[subject_labels]
        self.loaded_subject_labels = subject_labels
        # Return single trial from all Subject's Trials
        X, y = self.loaded_subject_data[local_trial_idx], self.loaded_subject_labels[local_trial_idx]
        return X, y

    # Returns a single trial as Tensor with Labels
    def __getitem__(self, trial):
        X, y = self.load_trial(trial)
        X = torch.as_tensor(X[None, ...], device=self.device)
        # X = TRANSFORM(X)
        return X, y


# Returns Loaders of Training + Test Datasets from index splits
# for n_class classification
def create_loaders_from_splits(splits, n_class, device, preloaded_data=None, preloaded_labels=None):
    subjects_train_idxs, subjects_test_idxs = splits
    subjects_train = [ALL_SUBJECTS[idx] for idx in subjects_train_idxs]
    subjects_test = [ALL_SUBJECTS[idx] for idx in subjects_test_idxs]
    print_subjects_ranges(subjects_train, subjects_test)
    return create_loader_from_subjects(subjects_train, n_class, device, preloaded_data, preloaded_labels), \
           create_loader_from_subjects(subjects_test, n_class, device, preloaded_data, preloaded_labels)


def create_loader_from_subjects(subjects, n_class, device, preloaded_data=None, preloaded_labels=None):
    ds_train = TrialsDataset(subjects, n_class, device,
                             preloaded_tuple=(
                                 preloaded_data, preloaded_labels) if DATA_PRELOAD else None)
    # Sample the trials in random order
    sampler_train = RandomSampler(ds_train)
    return DataLoader(ds_train, BATCH_SIZE, sampler=sampler_train, pin_memory=False)


# Finds indices of label-value occurrences in y
# and deletes them from X,y
def remove_label_occurences(X, y, label):
    label_idxs = np.where(y == label)
    return np.delete(X, label_idxs, axis=0), np.delete(y, label_idxs)


# Loads all Subjects Data + Labels for n_class Classification
# Very high memory usage (~4GB)
def load_all_subjects(n_class):
    preloaded_data = np.zeros((len(ALL_SUBJECTS), trials_for_classes[n_class], SAMPLES, CHANNELS),
                              dtype=np.float32)
    preloaded_labels = np.zeros((len(ALL_SUBJECTS), trials_for_classes[n_class]), dtype=np.float32)
    for i, subject in tqdm(enumerate(ALL_SUBJECTS), total=len(ALL_SUBJECTS)):
        data, labels = load_n_classes_tasks(subject, n_class)
        preloaded_data[i] = data
        preloaded_labels[i] = labels
    return preloaded_data, preloaded_labels


# Loads corresponding tasks for n_classes Classification
def load_n_classes_tasks(subject, n_classes):
    tasks = []
    if (n_classes == 4):
        tasks = [2, 4]
    elif (n_classes == 3):
        tasks = [2]
    elif (n_classes == 2):
        tasks = [2]
    return load_task_runs(subject, tasks, exclude_rest=(n_classes == 2),
                          exclude_bothfists=(n_classes == 4))


# If multiple tasks are used (4classes classification)
# labels need to be adjusted because different events from
# different tasks have the same numbers
inc_label = lambda label: label + 2 if label != 0 else label
increase_label = np.vectorize(inc_label)

# Both fists("1") gets removed, both feet("2") becomes the new "1"
# map_feet_to_fists = lambda label: label - 1 if label == 2 else label
# map_labels = np.vectorize(map_feet_to_fists)

event_dict = {'T0': 1, 'T1': 2, 'T2': 3}


# Merges runs from different tasks + correcting labels for n_class classification
def load_task_runs(subject, tasks, exclude_rest=False, exclude_bothfists=False):
    global map_label
    all_data = np.zeros((0, SAMPLES, CHANNELS))
    all_labels = np.zeros((0), dtype=np.int)
    for i, task in enumerate(tasks):
        tasks_event_dict = event_dict
        # for 2class classification exclude Rest events ("T0")
        # (see Paper "An Accurate EEGNet-based Motor-Imagery Brain–Computer ... ")
        if exclude_rest:
            tasks_event_dict = {'T1': 1, 'T2': 2}
        # for 4class classification exclude both fists event of task 4 ("T1")
        if exclude_bothfists & (task == 4):
            tasks_event_dict = {'T0': 1, 'T2': 2}
        data, labels = mne_load_subject(subject, runs[task], event_id=tasks_event_dict)

        # Correct labels if multiple tasks are loaded
        # e.g. in Task 2: "1": left fist, in Task 4: "1": both fists
        for n in range(i):
            labels = increase_label(labels)
        all_data = np.concatenate((all_data, data))
        all_labels = np.concatenate((all_labels, labels))
    return all_data, all_labels


# Loads single Subject of Physionet Data with MNE
# returns EEG data (X) and corresponding Labels (y)
# event_id specifies which event types should be loaded,
# if some are missing, they are ignored
# event_id= 'auto' loads all event types
def mne_load_subject(subject, runs, event_id='auto'):
    if VERBOSE:
        print(f"MNE loading Subject {subject}")
    # for 4 Class: need to map to 0,1,2,3
    # split reading in run lists (runs_t1,runs_t2,...)
    # give unique labels
    raw_fnames = eegbci.load_data(subject, runs, datasets_folder)
    raw_files = [read_raw_edf(f, preload=True) for f in raw_fnames]
    raw = concatenate_raws(raw_files)
    raw.rename_channels(lambda x: x.strip('.'))
    events, event_ids = mne.events_from_annotations(raw, event_id=event_id)
    picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                       exclude='bads')
    epochs = Epochs(raw, events, event_ids, EEG_TMIN, EEG_TMAX, picks=picks,
                    baseline=None, preload=True)
    # [trials (84), timepoints (1281), channels (64)]
    subject_data = np.swapaxes(epochs.get_data(), 2, 1)
    # Labels (0-index based)
    subject_labels = epochs.events[:, -1] - 1

    return subject_data, subject_labels
