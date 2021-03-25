import numpy as np
import torch  # noqa
import torch.nn.functional as F  # noqa
import torch.optim as optim  # noqa
from sklearn.preprocessing import MinMaxScaler
from torch import nn, Tensor  # noqa
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler, Subset  # noqa
from torch.utils.data.dataset import ConcatDataset as _ConcatDataset, TensorDataset, random_split  # noqa

from config import eeg_config
from data.physionet_dataset import trials_for_classes_per_subject_avail, n_classes_tasks, runs, \
    MNE_CHANNELS, TRIALS_PER_SUBJECT_RUN


def crop_time_and_label(raw, time, ch_names=MNE_CHANNELS):
    tdelta = eeg_config.EEG_TMAX - eeg_config.EEG_TMIN
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


# Map times in raw to corresponding samples
def map_times_to_samples(raw, times):
    samples = np.zeros(times.shape[0], dtype=np.int)
    for i in range(times.shape[0]):
        samples[i] = raw.time_as_index(times[i])
    return samples


# Map from Trials labels to classes
# e.g. 'T1' -> 1
def map_trial_labels_to_classes(labels):
    return [int(trial[-1]) for trial in labels]


# Normalize Data to [0;1] range
scaler = MinMaxScaler(copy=False)

normalize_data = lambda x: scaler.fit_transform(x.reshape(-1, x.shape[-1])).reshape(x.shape)



def get_runs_of_n_classes(n_classes):
    n_runs = []
    for task in n_classes_tasks[n_classes]:
        n_runs.extend(runs[task])
    return n_runs


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

# If multiple tasks are used (4classes classification)
# labels need to be adjusted because different events from
# different tasks have the same numbers
inc_label = lambda label: label + 2 if label != 0 else label
dec_label = lambda label: label - 1
increase_label = np.vectorize(inc_label)

decrease_label = np.vectorize(dec_label)