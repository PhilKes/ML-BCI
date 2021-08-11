"""
File: data_utils.py

Contains several utility functions needed during dataset loading.

History:
  2021-05-15: butterworth bandpass filter from scipy included - ms
"""
import os

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfilt
from sklearn.preprocessing import MinMaxScaler

from config import CONFIG
from data.datasets.phys.phys_dataset import PHYS
from util.misc import print_pretty_table, save_dataframe

'''
Subroutine: butter_bandpass_definition(lowcut=0.0, highcut=80.0, fs=160, order=3)
  Defintion of a nth order Butterworth bandpass filter. Code is based on::
  https://warrenweckesser.github.io/papers/weckesser-scipy-linear-filters.pdf
'''


def butter_bandpass_definition(lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    if (lowcut != None):
        low = lowcut / nyq
        if (highcut != None):
            high = highcut / nyq
            return butter(order, [low, high], btype='bandpass', output='sos')
        else:
            return butter(order, low, btype='highpass', output='sos')
    else:
        high = highcut / nyq
        return butter(order, high, btype='lowpass', output='sos')


'''
Subroutine: butter_bandpass_filt(indata, lowcut, highcut, fs, order)
  Applies the nth order Butterworth filter defined in butter_bandpass_definition()
  to 'indata'
'''


def butter_bandpass_filt(indata, lowcut, highcut, fs, order):
    sos = butter_bandpass_definition(lowcut, highcut, fs, order)
    outdata = sosfilt(sos, indata)
    return outdata


def crop_time_and_label(raw, time, ch_names=PHYS.CHANNELS):
    tdelta = CONFIG.EEG.TMAX - CONFIG.EEG.TMIN
    if (time - tdelta) < 0:
        raise Exception(f"Cant load {tdelta}s before timepoint={time}s")
    raw1 = raw.copy()
    raw1.pick_channels(ch_names)
    raw1.crop(time - tdelta, time)
    data, times = raw1[:, :]
    return data, times, raw1.annotations


def get_data_from_raw(raw, ch_names=PHYS.CHANNELS):
    # raw1 = raw.copy()
    raw.pick_channels(ch_names)
    data, times = raw[:, :]

    return data


def get_label_at_idx(times, annot, sample):
    now_time = times[sample]
    if sample < CONFIG.EEG.SAMPLES:
        return None, now_time
    middle_sample_of_window = int(sample - (CONFIG.EEG.SAMPLES / 2))
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


# omit_bl: Omit Baseline Runs (Rest Trials)
def get_runs_of_n_classes(n_classes, omit_bl=False):
    n_runs = []
    for task in PHYS.n_classes_tasks[n_classes]:
        if omit_bl & (task == 0):
            continue
        n_runs.extend(PHYS.runs[task])
    return n_runs


# Returns amount of Trials per Subject (all Runs, all Classes)
trials_per_class_for_1_runs = 7
trials_per_class_for_2_runs = 14
trials_per_class_for_3_runs = 21


def get_trials_size(n_class, equal_trials=True, ignored_runs=[]):
    if not equal_trials:
        return PHYS.trials_for_classes_per_subject_avail[n_class]
    n_class_runs = [run for run in get_runs_of_n_classes(n_class, True) if run not in ignored_runs]
    r = len(n_class_runs)
    # 4class uses Task 4 only for T1 events
    if n_class == 4:
        r -= 3
    return (trials_per_class_for_1_runs * n_class) * r


# Ensure that same amount of Trials for each class is present
def get_equal_trials_per_class(data, labels, classes, trials):
    trials_idxs = np.zeros(0, dtype=np.int)
    for cl in range(classes):
        cl_idxs = np.where(labels == cl)[0]
        # Get random Rest Trials from Run
        if cl == 0:
            np.random.seed(39)
            np.random.shuffle(cl_idxs)
        cl_idxs = cl_idxs[:trials]
        if (cl == 0) & (not PHYS.CONFIG.REST_TRIALS_FROM_BASELINE_RUN) & (PHYS.CONFIG.REST_TRIALS_LESS > 0):
            cl_idxs = cl_idxs[:-PHYS.CONFIG.REST_TRIALS_LESS]
        trials_idxs = np.concatenate((trials_idxs, cl_idxs))
    trials_idxs = np.sort(trials_idxs)
    return data[trials_idxs], labels[trials_idxs]


def split_trials(data, labels, splits, samples):
    split_size = np.math.floor(data.shape[0] / splits)
    # data = np.array_split(data, splits, axis=2)
    data_split = np.zeros((data.shape[0] * splits, data.shape[1], samples))
    labels_split = np.zeros((data.shape[0] * splits), dtype=np.int)
    for t_idx in range(data.shape[0]):
        for split in range(splits):
            data_split[t_idx * splits + split] = data[t_idx, :, (samples * split):(samples * (split + 1))]
            labels_split[t_idx * splits + split] = labels[t_idx]
    # print(collections.Counter(labels))
    # print(collections.Counter(labels_split))
    return data_split, labels_split


# Calculates relative area of correct Prediction per Trial
def get_correctly_predicted_areas(n_class, sample_predictions, trials_classes, trials_start_samples, max_samples):
    print()
    trials_correct_areas_relative = np.zeros((len(trials_classes)))
    trials_correct_areas = np.zeros((len(trials_classes)))
    # Entire Area of a Trial (Amount of Samples * 100 (percent))
    trials_areas = np.zeros((len(trials_classes)))
    for idx, trial_class in enumerate(trials_classes):
        first_sample = trials_start_samples[idx]
        # If 1st Trial, Prediction only starts if now_sample> eeg_config.SAMPLES
        if idx == 0:
            first_sample += CONFIG.EEG.SAMPLES
        if idx == len(trials_classes) - 1:
            last_sample = max_samples
        else:
            last_sample = trials_start_samples[idx + 1]
        trials_correct_areas[idx] = np.sum(sample_predictions[trial_class, first_sample:last_sample])
        trials_areas[idx] = (last_sample - first_sample) * 100
        trials_correct_areas_relative[idx] = trials_correct_areas[idx] / trials_areas[idx]
        # print(f"Trial {idx:02d}: {trials_correct_areas_relative[idx]:.3f}")
    return trials_correct_areas_relative


# If multiple tasks are used (4classes classification)
# labels need to be adjusted because different events from
# different tasks have the same numbers
inc_label = lambda label: label + 2 if label != 0 else label
dec_label = lambda label: label - 1
increase_label = np.vectorize(inc_label)

decrease_label = np.vectorize(dec_label)


# Subtracts Accuracies of first config ('defaults') from all other configs
# Returns Array of Accuracy differences
# runs_classes_accs shape: [run,n_class,acc]
def subtract_first_config_accs(runs_classes_accs, amt_configs):
    # Get idxs of 'all' (default) accuracies
    first_conf_accs_idxs = np.arange(0, runs_classes_accs.size, amt_configs)
    first_conf_accs = runs_classes_accs[first_conf_accs_idxs, 0]
    # remove 'all' accuracies
    acc_diffs = np.delete(runs_classes_accs, first_conf_accs_idxs, axis=0)

    # Calculate differences between 'all' and f1/f2/f3 acc
    for run in range(acc_diffs.shape[0]):
        first_conf_acc_idx = run // (amt_configs - 1)
        all_acc = first_conf_accs[first_conf_acc_idx]
        acc_diffs[run] = all_acc - acc_diffs[run]
    return acc_diffs


def save_accs_panda(name, folderName, accs, columns, names, tag=None):
    df = pd.DataFrame(data=accs, index=names, columns=columns)
    # if tag is not None:
    #     folderName += f'/{tag}'
    # Write results into .csv and .txt
    df.to_csv(f"{folderName}/{tag if tag is not None else 'training'}_{name}.csv")
    save_dataframe(df, os.path.join(f"{folderName}",
                                    f"{tag if tag is not None else 'training'}_{name}.txt"))
    print_pretty_table(df)
