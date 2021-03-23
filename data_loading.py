"""
Handles all EEG-Data loading of Physionet Motor Imagery Dataset via MNE Library
(https://neuro.inf.unibe.ch/AlgorithmsNeuroscience/Tutorial_files/DataLoading.html)
On initial Run MNE downloads the Physionet Dataset into ./datasets
(https://physionet.org/content/eegmmidb/1.0.0/)
"""
import mne
import numpy as np
import torch  # noqa
import torch.nn.functional as F  # noqa
import torch.optim as optim  # noqa
from mne import Epochs
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf
from sklearn.preprocessing import MinMaxScaler
from torch import nn, Tensor  # noqa
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler, Subset  # noqa
from torch.utils.data.dataset import ConcatDataset as _ConcatDataset, TensorDataset, random_split  # noqa
from tqdm import tqdm

from config import VERBOSE, eeg_config, datasets_folder, DATA_PRELOAD, BATCH_SIZE, \
    MNE_CHANNELS, global_config, TRIALS_PER_SUBJECT_RUN

from util.misc import print_subjects_ranges, split_np_into_chunks

# Some Subjects are excluded due to differing numbers of Trials in the recordings
excluded_subjects = [88, 92, 100, 104]
ALL_SUBJECTS = [i for i in range(1, 110) if i not in excluded_subjects]

runs_rest = [1]  # Baseline, eyes open
runs_t1 = [3, 7, 11]  # Task 1 (open and close left or right fist)
runs_t2 = [4, 8, 12]  # Task 2 (imagine opening and closing left or right fist)
runs_t3 = [5, 9, 13]  # Task 3 (open and close both fists or both feet)
runs_t4 = [6, 10, 14]  # Task 4 (imagine opening and closing both fists or both feet)

runs = [runs_rest, runs_t1, runs_t2, runs_t3, runs_t4]

n_classes_tasks = {1: [0], 2: [2], 3: [0, 2], 4: [0, 2, 4]}
# Sample run for n_class live_sim mode
n_classes_live_run = {2: runs_t2[0], 3: runs_t2[0], 4: runs_t2[0]}

# Maximum available trials
trials_for_classes_per_subject_avail = {2: 42, 3: 84, 4: 153}

# All total trials per class per n_class-Classification
classes_trials = {
    "2class": {
        0: 445,  # Left
        1: 437,  # Right
    },
    "3class": {
        0: 882,  # Rest
        1: 445,  # Left
        2: 437,  # Right
    },
    "4class": {
        0: 1748,  # Rest
        1: 479,  # Left
        2: 466,  # Right
        3: 394,  # Both Fists
    },
}


# Dataset for EEG Trials Data (divided by subjects)
class TrialsDataset(Dataset):

    def __init__(self, subjects, n_classes, device, preloaded_tuple=None, ch_names=MNE_CHANNELS, equal_trials=False):
        self.subjects = subjects
        # Buffers for last loaded Subject data+labels
        self.loaded_subject = -1
        self.loaded_subject_data = None
        self.loaded_subject_labels = None
        self.n_classes = n_classes
        self.runs = []
        self.device = device
        self.trials_per_subject = get_trials_size(n_classes, equal_trials)
        self.equal_trials = equal_trials
        self.preloaded_data = preloaded_tuple[0] if preloaded_tuple is not None else None
        self.preloaded_labels = preloaded_tuple[1] if preloaded_tuple is not None else None
        self.ch_names = ch_names

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
            if self.preloaded_data.shape[0] == len(ALL_SUBJECTS):
                idx = ALL_SUBJECTS.index(subject)
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


# Returns Loaders of Training + Test Datasets from index splits
# for n_class classification
# also returns Validtion Loader containing validation_subjects subject for loss calculation
def create_loaders_from_splits(splits, validation_subjects, n_class, device, preloaded_data=None,
                               preloaded_labels=None, bs=BATCH_SIZE, ch_names=MNE_CHANNELS,
                               equal_trials=False, used_subjects=ALL_SUBJECTS):
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
                                preloaded_labels=None, bs=BATCH_SIZE, ch_names=MNE_CHANNELS, equal_trials=False):
    ds_train = TrialsDataset(subjects, n_class, device,
                             preloaded_tuple=(preloaded_data, preloaded_labels) if DATA_PRELOAD else None,
                             ch_names=ch_names, equal_trials=equal_trials)
    # Sample the trials in random order
    sampler_train = RandomSampler(ds_train)
    return DataLoader(ds_train, bs, sampler=sampler_train, pin_memory=False)


def create_loader_from_subject(used_subject, train_share, test_share, n_class, batch_size, ch_names, device):
    preloaded_data, preloaded_labels = load_subjects_data([used_subject], n_class, ch_names)
    preloaded_data = preloaded_data.reshape((preloaded_data.shape[1], 1, preloaded_data.shape[2],
                                             preloaded_data.shape[3]))
    preloaded_labels = preloaded_labels.reshape(preloaded_labels.shape[1])

    print("data", preloaded_data.shape)
    print("labels", preloaded_labels.shape)

    data_set = TensorDataset(torch.as_tensor(preloaded_data, device=device, dtype=torch.float32),
                             torch.as_tensor(preloaded_labels, device=device, dtype=torch.int))
    lengths = [np.math.ceil(preloaded_data.shape[0] * train_share), np.math.floor(preloaded_data.shape[0] * test_share)]
    train_set, test_set = random_split(data_set, lengths)
    print()

    loader_train = DataLoader(train_set, batch_size, sampler=RandomSampler(train_set), pin_memory=False)
    loader_test = DataLoader(test_set, batch_size, sampler=RandomSampler(test_set), pin_memory=False)

    return loader_train, loader_test


def get_trials_size(n_class, equal_trials):
    trials = trials_for_classes_per_subject_avail[n_class]
    if equal_trials:
        r = len(get_runs_of_n_classes(n_class))
        if n_class == 4:
            r -= 3
        if n_class == 2:
            r -= 1
        if n_class == 3:
            r -= 1
        trials = TRIALS_PER_SUBJECT_RUN * r
    return trials


# Normalize Data to [0;1] range
scaler = MinMaxScaler(copy=False)
normalize_data = lambda x: scaler.fit_transform(x.reshape(-1, x.shape[-1])).reshape(x.shape)


# Loads all Subjects Data + Labels for n_class Classification
# Very high memory usage for ALL_SUBJECTS (~2GB)
def load_subjects_data(subjects, n_class, ch_names=MNE_CHANNELS, equal_trials=True,
                       normalize=False):
    subjects.sort()
    trials = get_trials_size(n_class, equal_trials)
    preloaded_data = np.zeros((len(subjects), trials, len(ch_names), eeg_config.SAMPLES), dtype=np.float32)
    preloaded_labels = np.zeros((len(subjects), trials,), dtype=np.float32)
    print("Preload Shape", preloaded_data.shape)
    for i, subject in tqdm(enumerate(subjects), total=len(subjects)):
        data, labels = load_n_classes_tasks(subject, n_class, ch_names, equal_trials)

        if data.shape[0] > preloaded_data.shape[1]:
            data, labels = data[:preloaded_data.shape[1]], labels[:preloaded_labels.shape[1]]
        preloaded_data[i] = data
        preloaded_labels[i] = labels
    if normalize:
        preloaded_data = normalize_data(preloaded_data)
    return preloaded_data, preloaded_labels


def get_runs_of_n_classes(n_classes):
    n_runs = []
    for task in n_classes_tasks[n_classes]:
        n_runs.extend(runs[task])
    return n_runs


# Loads corresponding tasks for n_classes Classification
def load_n_classes_tasks(subject, n_classes, ch_names=MNE_CHANNELS, equal_trials=False):
    tasks = n_classes_tasks[n_classes]
    data, labels = load_task_runs(subject, tasks,
                                  exclude_bothfists=(n_classes == 4), ch_names=ch_names,
                                  equal_trials=equal_trials, n_class=n_classes)
    if n_classes == 2:
        labels = dec_label(labels)
    return data, labels


# If multiple tasks are used (4classes classification)
# labels need to be adjusted because different events from
# different tasks have the same numbers
inc_label = lambda label: label + 2 if label != 0 else label
dec_label = lambda label: label - 1
increase_label = np.vectorize(inc_label)
decrease_label = np.vectorize(dec_label)

event_dict = {'T0': 1, 'T1': 2, 'T2': 3}


# Loads Rest trials from the 1st baseline run of subject
# if baseline run is not long enough for all needed trials
# random 3s Trials are generated from baseline run
def mne_load_rests(subject, trials, ch_names):
    X, y = mne_load_subject(subject, 1, tmin=0, tmax=60, event_id='auto', ch_names=ch_names)
    X = np.swapaxes(X, 2, 1)
    chs = len(ch_names)
    if X.shape[0] > 1:
        X = X[:1, :, :]
    X = np.squeeze(X, axis=0)
    X_cop = np.array(X, copy=True)
    X = split_np_into_chunks(X, eeg_config.SAMPLES)

    missing_trials = trials - X.shape[0]
    if missing_trials > 0:
        for m in range(missing_trials):
            np.random.seed(m)
            rand_start_idx = np.random.randint(0, X_cop.shape[0] - eeg_config.SAMPLES)
            # print("rand_start", rand_start_idx)
            rand_x = np.zeros((1, eeg_config.SAMPLES, chs))
            rand_x[0] = X_cop[rand_start_idx: (rand_start_idx + eeg_config.SAMPLES)]
            X = np.concatenate((X, rand_x))
    y = np.full(X.shape[0], y[0])
    # print("X", X.shape, "Y", y)
    X = np.swapaxes(X, 2, 1)
    return X, y


# Merges runs from different tasks + correcting labels for n_class classification
def load_task_runs(subject, tasks, exclude_bothfists=False, ch_names=MNE_CHANNELS, n_class=3,
                   equal_trials=False):
    all_data = np.zeros((0, len(ch_names), eeg_config.SAMPLES))
    all_labels = np.zeros((0), dtype=np.int)
    contains_rest_task = (0 in tasks)
    # Load Subject Data of all Tasks
    for task_idx, task in enumerate(tasks):
        tasks_event_dict = {'T1': 2, 'T2': 3}
        # for 4class classification exclude both fists event of task 4 ("T1")
        if exclude_bothfists & (task == 4):
            tasks_event_dict = {'T2': 2}
        if task == 0:
            data, labels = mne_load_rests(subject, TRIALS_PER_SUBJECT_RUN, ch_names)
        else:
            data, labels = mne_load_subject(subject, runs[task], event_id=tasks_event_dict, ch_names=ch_names)
            # Ensure equal amount of trials per class
            if equal_trials:
                trials_per_subject = TRIALS_PER_SUBJECT_RUN
                trials_idxs = np.zeros(0, dtype=np.int)
                classes = n_class
                if n_class == 2:
                    classes = 3
                for cl in range(classes):
                    if n_class == 0:
                        continue
                    cl_idxs = np.where(labels == cl)[0]

                    cl_idxs = cl_idxs[:trials_per_subject]
                    trials_idxs = np.concatenate((trials_idxs, cl_idxs))
                trials_idxs = np.sort(trials_idxs)
                data, labels = data[trials_idxs], labels[trials_idxs]

            # Correct labels if multiple tasks are loaded
            # e.g. in Task 2: "1": left fist, in Task 4: "1": both fists
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
def mne_load_subject(subject, runs, event_id='auto', ch_names=MNE_CHANNELS, tmin=eeg_config.EEG_TMIN,
                     tmax=eeg_config.EEG_TMAX):
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
    raw_fnames = eegbci.load_data(subject, runs, datasets_folder)
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


# Methods for live_sim MODE
tdelta = eeg_config.EEG_TMAX - eeg_config.EEG_TMIN


def crop_time_and_label(raw, time, ch_names=MNE_CHANNELS):
    if (time - tdelta) < 0:
        raise Exception(f"Cant load {tdelta}s before timepoint={time}s")
    raw1 = raw.copy()
    raw1.pick_channels(ch_names)
    raw1.crop(time - tdelta, time)
    data, times = raw1[:, :]
    return data, times, raw1.annotations


def get_data_from_raw(raw, ch_names=MNE_CHANNELS):
    # raw1 = raw.copy()
    raw.pick_channels(ch_names)
    data, times = raw[:, :]
    return data


def get_label_at_idx(times, annot, sample):
    now_time = times[sample]
    if sample < eeg_config.SAMPLES:
        return None, now_time
    middle_sample_of_window = int(sample - (eeg_config.SAMPLES / 2))
    time = times[middle_sample_of_window]
    onsets = annot.onset
    # boolean_array = np.logical_and(onsets >= time, onsets <= time + tdelta)
    # find index where time would be inserted
    # -> index of label is sorted_idx-1
    sorted_idx = np.searchsorted(onsets, [time])[0]
    # Determine if majority of samples lies in
    # get label of sample_idx in the middle of the window
    label = annot.description[sorted_idx - 1]
    return label, now_time


def get_label_at_time(raw, times, time):
    idx = raw.time_as_index(time)
    return get_label_at_idx(times, raw.annotations, idx)
