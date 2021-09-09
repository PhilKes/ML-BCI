"""
Handles all EEG-Data loading of Physionet Motor Imagery Dataset via MNE Library
(https://neuro.inf.unibe.ch/AlgorithmsNeuroscience/Tutorial_files/DataLoading.html)
On initial Run MNE downloads the Physionet Dataset into datasets_folder
(https://physionet.org/content/eegmmidb/1.0.0/)

Edition History:
2021-05-31: mne_load_subject_raw(): fmin, fmax explicitely set - ms
"""
import math
from typing import List

import mne
import numpy as np
import torch  # noqa
from mne import Epochs
from mne.io import concatenate_raws, read_raw_edf
from torch.utils.data import Dataset, DataLoader, RandomSampler  # noqa
from torch.utils.data.dataset import TensorDataset  # noqa
from tqdm import tqdm

from config import VERBOSE, CONFIG, RESAMPLE
from data.MIDataLoader import MIDataLoader
from data.data_utils import dec_label, increase_label, get_trials_size, \
    get_equal_trials_per_class, slice_trials, get_runs_of_n_classes, get_data_from_raw, map_times_to_samples, \
    map_trial_labels_to_classes
from data.datasets.TrialsDataset import TrialsDataset
from data.datasets.phys.phys_dataset import PHYS, PHYSConstants
from machine_learning.util import calc_slice_start_samples
from paths import datasets_folder, results_folder
from util.misc import split_np_into_chunks, print_numpy_counts
from util.plot import matplot

# Dont print MNE loading logs
mne.set_log_level('WARNING')


class PHYSTrialsDataset(TrialsDataset):
    """
     TrialsDataset class Implementation for Physionet Dataset
    """

    def __init__(self, subjects, used_subjects, n_class, preloaded_tuple,
                 ch_names=PHYS.CHANNELS, equal_trials=True):
        super().__init__(subjects, used_subjects, n_class, preloaded_tuple, ch_names, equal_trials)

        self.trials_per_subject = get_trials_size(n_class, equal_trials) \
                                  * CONFIG.EEG.TRIALS_SLICES - PHYS.CONFIG.REST_TRIALS_LESS


class PHYSDataLoader(MIDataLoader):
    """
    MIDataLoader implementation for Physionet Dataset
    """

    CONSTANTS: PHYSConstants = PHYS
    ds_class = PHYSTrialsDataset

    @classmethod
    def create_n_class_loaders_from_subject(cls, used_subject: int, n_class: int, n_test_runs: int,
                                            batch_size: int, ch_names: List[str]):
        """
        Returns Train/Test Loaders containing all n_class Runs of subject
        n_test_runs specifies how many Runs are reserved for Testing
        2/3class: 3 Runs, 4class: 6 Runs
        :return: loader_train: DataLoader, load_test: DataLoader
        """
        # For n_class = 3/4 the Test Dataset needs at least 1 Run of Task 2 and 1 Run of Task 4
        if n_class > 2:
            n_class_runs = get_runs_of_n_classes(n_class)
            test_runs = [PHYS.runs[2][-1], PHYS.runs[4][-1]]
        # For n_class = 2 the Test Dataset only needs 1 Run of Task 2
        else:
            n_class_runs = get_runs_of_n_classes(n_class)
            test_runs = n_class_runs[-1]

        train_runs = [run for run in n_class_runs if run not in test_runs]
        loader_train = cls.create_loader_from_subject_runs(used_subject, n_class, batch_size, ch_names,
                                                           ignored_runs=test_runs)
        loader_test = cls.create_loader_from_subject_runs(used_subject, n_class, batch_size, ch_names,
                                                          ignored_runs=train_runs)
        return loader_train, loader_test

    @classmethod
    def load_subjects_data(cls, subjects: List[int], n_class: int, ch_names: List[str] = PHYS.CHANNELS,
                           equal_trials: bool = True, ignored_runs: List[int] = []):
        subjects.sort()
        trials = get_trials_size(n_class, equal_trials, ignored_runs)
        # trials_per_run_class = np.math.floor(trials / n_class)
        # trials_per_run_class = 21
        # n-times the amount of Trials for TRIALS_SLICES = n
        trials = trials * CONFIG.EEG.TRIALS_SLICES
        if n_class > 2:
            trials -= PHYS.CONFIG.REST_TRIALS_LESS

        # print(CONFIG)
        preloaded_data = np.zeros((len(subjects), trials, len(ch_names), CONFIG.EEG.SAMPLES), dtype=np.float32)
        preloaded_labels = np.zeros((len(subjects), trials,), dtype=np.int)
        if RESAMPLE & (cls.CONSTANTS.CONFIG.SAMPLERATE != CONFIG.SYSTEM_SAMPLE_RATE):
            print(f"RESAMPLING from {cls.CONSTANTS.CONFIG.SAMPLERATE}Hz to {CONFIG.SYSTEM_SAMPLE_RATE}Hz")

        print("Preload Shape", preloaded_data.shape)
        for i, subject in tqdm(enumerate(subjects), total=len(subjects)):
            data, labels = cls.load_n_classes_tasks(subject, n_class, ch_names, equal_trials,
                                                    ignored_runs=ignored_runs)
            # if data.shape[0] > preloaded_data.shape[1]:
            #     data, labels = data[:preloaded_data.shape[1]], labels[:preloaded_labels.shape[1]]
            preloaded_data[i] = data
            preloaded_labels[i] = labels
        # print_numpy_counts(preloaded_labels)
        return preloaded_data, preloaded_labels

    @classmethod
    def load_live_sim_data(cls, subject, n_class, ch_names):
        """
        Load all necessary Data for the Live Simulation Run of subject
        X: ndarray (channels,Samples) of single Subject's Run data
        max_sample: Maximum sample number of the Run
        slices: Trial Slices
        trials_classes: ndarray with label nr. of every Trial in the Run
        trials_start_times: ndarray with Start Times of every Trial in the Run
        trials_start_samples: ndarray with Start Samples of every Trial in the Run
        slice_start_samples: ndarray with Start Sample of every Slice in the Run
        """
        # Load Raw Subject Run for n_class
        raw = cls.mne_load_subject_raw(subject, PHYS.n_classes_live_run[n_class], ch_names=ch_names)
        # Get Data from raw Run
        X = get_data_from_raw(raw)
        trials_classes = map_trial_labels_to_classes(raw.annotations.description)
        X, _ = cls.prepare_data_labels(X, trials_classes)

        max_sample = X.shape[-1]
        slices = CONFIG.EEG.TRIALS_SLICES
        # times = raw.times[:max_sample]
        trials_start_times = raw.annotations.onset

        # Get samples of Trials Start Times
        trials_start_samples = map_times_to_samples(raw, trials_start_times)
        trials_start_samples = np.zeros(trials_start_times.shape)
        for i, trial_start_time in enumerate(trials_start_times):
            trials_start_samples[i] = int(trial_start_time * CONFIG.EEG.SAMPLERATE)

        trial_time_length = CONFIG.EEG.TMAX - CONFIG.EEG.TMIN
        trials_samples_length = np.full(trials_start_times.shape, int(trial_time_length * CONFIG.EEG.SAMPLERATE))
        slice_start_samples = calc_slice_start_samples(trials_start_times, trials_samples_length, slices)

        return X, max_sample, slices, trials_classes, trials_start_times, trials_start_samples, slice_start_samples

    @classmethod
    def create_loader_from_subject_runs(cls, subject, n_class, batch_size, ch_names,
                                        ignored_runs=[]):
        """
        Creates Loader containing all Trials of n_class Runs of subject
        :param ignored_runs: List of Run Nrs. that should not be loaded
        :return: DataLoader with Data of given Subject's Runs
        """
        preloaded_data, preloaded_labels = cls.load_subjects_data([subject], n_class, ch_names,
                                                                  ignored_runs=ignored_runs)
        preloaded_data = preloaded_data.reshape((preloaded_data.shape[1], 1, preloaded_data.shape[2],
                                                 preloaded_data.shape[3]))
        preloaded_labels = preloaded_labels.reshape(preloaded_labels.shape[1])
        return cls.create_loader(preloaded_data, preloaded_labels, batch_size)

    @classmethod
    def load_n_classes_tasks(cls, subject, n_class, ch_names=PHYS.CHANNELS, equal_trials=True,
                             ignored_runs=[]):
        """
        Loads corresponding tasks for n_class Classification
        :param ignored_runs: List of Run Nrs. that should not be loaded
        :return: data, labels: ndarrays with Data and Labels of n_class Task Runs of Subject
        """
        tasks = PHYS.n_classes_tasks[n_class].copy()
        if (not PHYS.CONFIG.REST_TRIALS_FROM_BASELINE_RUN) & (0 in tasks):
            tasks.remove(0)
        data, labels = cls.load_task_runs(subject, tasks,
                                          # if 3/4-class should contain 'rest' trials:
                                          # exclude_bothfists=(n_class == 4),
                                          # exclude_rests=(n_class == 2),

                                          # if 3/4-class should not contain 'rest' trials:
                                          exclude_bothfists=(n_class == 3),
                                          exclude_rests=True,
                                          ch_names=ch_names, ignored_runs=ignored_runs,
                                          equal_trials=equal_trials,
                                          n_class=n_class)
        # if 3/4-class contain 'rest' Trials:
        # if n_class == 2:
        #     labels = dec_label(labels)
        # if 3/4-class do not contain 'rest' Trials:
        labels = dec_label(labels)
        # print_numpy_counts(labels)
        return data, labels

    event_dict = {'T0': 1, 'T1': 2, 'T2': 3}

    @classmethod
    def mne_load_rests(cls, subject: int, trials: int, ch_names: List[str], samples: int):
        """
        Loads Rest trials from the 1st baseline run of subject
        if baseline run is not long enough for all needed trials
        random Trials are generated from baseline run
        :param samples: Amount of Samples per Rest Trial
        :return: X,y: ndarrays with Data + Labels of generated Rest Trials
        """
        used_trials = trials - PHYS.CONFIG.REST_TRIALS_LESS
        X, y = cls.mne_load_subject(subject, 1, tmin=0, tmax=60, event_id='auto', ch_names=ch_names)
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

    @classmethod
    def load_task_runs(cls, subject: int, tasks: List[int], exclude_bothfists=False, ch_names=PHYS.CHANNELS, n_class=3,
                       equal_trials=True, exclude_rests=False,
                       ignored_runs=[]):
        """
        Merges runs from different tasks + correcting labels for n_class classification
        :param tasks: Task Nrs. to load
        :param exclude_bothfists: For n_class=4 exclude bothfists from Task 4
        :param equal_trials: Specifies if an equal amount of Trials per Class should be loaded
        :param trials_per_run_class: Amount of Trials per Subject Run
        :param exclude_rests: Specifies if Rest Trials should be excluded
        :param ignored_runs: List of Run Nrs. that should not be loaded
        :return: all_data,all_labels: ndarrays with Data + Labels of Trials of given tasks
        """
        load_samples = CONFIG.EEG.SAMPLES
        all_data = np.zeros((0, len(ch_names), load_samples))
        all_labels = np.zeros((0), dtype=np.int)

        # Load Subject Data of all Tasks
        for task_idx, task in enumerate(tasks):
            used_runs = [run for run in PHYS.runs[task] if run not in ignored_runs]
            if len(used_runs) == 0:
                continue
            trials_per_run_class = len(used_runs) * PHYS.TRIALS_PER_CLASS_PER_RUN * CONFIG.EEG.TRIALS_SLICES
            # Task = 0 -> Rest Trials "T0"
            if PHYS.CONFIG.REST_TRIALS_FROM_BASELINE_RUN & (task == 0):
                data, labels = cls.mne_load_rests(subject, trials_per_run_class, ch_names, load_samples)
            else:
                # if Rest Trials are loaded from Baseline Run, ignore "TO"s in all other Runs
                # exclude_rests is True for 2class Classification
                if PHYS.CONFIG.REST_TRIALS_FROM_BASELINE_RUN | exclude_rests:
                    tasks_event_dict = {'T1': 2, 'T2': 3}
                else:
                    tasks_event_dict = {'T0': 1, 'T1': 2, 'T2': 3}
                # for 4class classification exclude both fists event of task 4 ("T1")
                if exclude_bothfists & (task == 4):
                    tasks_event_dict = {'T2': 2}
                data, labels = cls.mne_load_subject(subject, used_runs, event_id=tasks_event_dict,
                                                    ch_names=ch_names)
                # Ensure equal amount of trials per class
                if equal_trials:
                    # classes = n_class
                    # if n_class == 2:
                    #     classes = 3
                    data, labels = get_equal_trials_per_class(data, labels, trials_per_run_class)
                # Correct labels if multiple tasks are loaded
                # e.g. in Task 2: "1": left fist, in Task 4: "1": both fists
                contains_rest_task = (0 in tasks)
                for n in range(task_idx if (not contains_rest_task) else task_idx - 1):
                    labels = increase_label(labels)
            all_data = np.concatenate((all_data, data))
            all_labels = np.concatenate((all_labels, labels))
        # all_data, all_labels = unison_shuffled_copies(all_data, all_labels)
        # print_numpy_counts(all_labels)
        return all_data, all_labels

    @classmethod
    def mne_load_subject(cls, subject, runs, event_id='auto', ch_names=PHYS.CHANNELS, tmin=None,
                         tmax=None):
        """
        Loads single Subject of Physionet Data with MNE

        :param runs: List of Run Nrs. to be loaded
        :param event_id: Specifies which event types should be loaded, if some are missing, they are ignored
        'auto' loads all event types
        :param tmin: Defines start time of the Trial EEG Data
        :param tmax: Defines end time of the Trial EEG Data
        :return: X,y: ndarrays with Data + Labels of Trials of the given Subject Runs
        """
        if tmax is None:
            tmax = CONFIG.EEG.TMAX
        if tmin is None:
            tmin = CONFIG.EEG.TMIN
        raw = cls.mne_load_subject_raw(subject, runs)

        events, event_ids = mne.events_from_annotations(raw, event_id=event_id)
        picks = mne.pick_channels(raw.info['ch_names'], ch_names)

        epochs = Epochs(raw, events, event_ids, tmin, tmax - (1 / CONFIG.EEG.SAMPLERATE), picks=picks,
                        baseline=None, preload=True)
        # [trials, channels, timepoints,]
        subject_data = epochs.get_data().astype('float32')
        # Labels (0-index based)
        subject_labels = epochs.events[:, -1] - 1
        subject_data, subject_labels = cls.prepare_data_labels(subject_data, subject_labels)
        return subject_data, subject_labels

    @classmethod
    def mne_load_subject_raw(cls, subject, runs, ch_names=PHYS.CHANNELS):
        """
        Loads raw Subject run with specified channels
        :return: raw: mne.io.Raw Object containing all Data of given Subject Runs
        """
        if VERBOSE:
            print(f"MNE loading Subject {subject} Runs {runs}")
        raw_fnames = PHYS.mne_dataset.load_data(subject, runs, datasets_folder)
        raw_files = [read_raw_edf(f, preload=True) for f in raw_fnames]
        raw: mne.io.Raw = concatenate_raws(raw_files)
        raw.rename_channels(lambda x: x.strip('.'))
        raw.pick_channels(ch_names)
        # # Resample/Filter if necessary
        # raw.apply_function(cls.resample_and_filter, channel_wise=False)
        # raw.load_data()
        return raw


def plot_live_sim_subject_run(subject=1, n_class=3, save_path=f"{results_folder}/plots_training",
                              ch_names=PHYS.CHANNELS):
    """
    Plots Subject Run with raw EEG Channel data
    :param save_path: Path to location where plot (.png) should be saved
    :param ch_names: List of EEG Channels to plot (see physionet_dataset.py for available Channels)
    """
    # ch_names = ['F4', 'Oz', 'F7', 'F6']

    # Load Raw Subject Run for n_class
    raw = PHYSDataLoader.mne_load_subject_raw(subject, PHYS.n_classes_live_run[n_class], ch_names=ch_names)
    # Get Data from raw Run
    X = get_data_from_raw(raw)
    X = MIDataLoader.prepare_data_labels(X)

    max_sample = raw.n_times
    slices = 5
    # times = raw.times[:max_sample]
    trials_start_times = raw.annotations.onset

    # Get samples of Trials Start Times
    trials_start_samples = map_times_to_samples(raw, trials_start_times)

    # matplot(sample_predictions,
    #         f"{n_class}class Live Simulation_S{subject:03d}",
    #         'Time in sec.', f'Prediction in %', fig_size=(80.0, 10.0), max_y=100.5,
    #         vspans=vspans, vlines=vlines, ticks=trials_start_samples, x_values=trials_start_times,
    #         labels=[f"T{i}" for i in range(n_class)], save_path=dir_results)
    # Split into multiple plots, otherwise too long
    plot_splits = 8
    trials_split_size = int(trials_start_samples.shape[0] / plot_splits)
    n_class_offset = 0 if n_class > 2 else 1
    for i in range(plot_splits):
        first_trial = i * trials_split_size
        last_trial = (i + 1) * trials_split_size - 1
        first_sample = trials_start_samples[first_trial]
        if i == plot_splits - 1:
            last_sample = max_sample
        else:
            last_sample = trials_start_samples[last_trial + 1]
        matplot(X,
                f"EEG Recording ({len(ch_names)} EEG Channels)",
                'Time in sec.', f'Prediction in %', fig_size=(20.0, 10.0),
                color_offset=n_class_offset, font_size=32.0,
                vlines_label="Trained timepoints", legend_loc='lower right',
                ticks=trials_start_samples[first_trial:last_trial + 1],
                min_x=first_sample, max_x=last_sample,
                x_values=trials_start_times[first_trial:last_trial + 1],
                labels=ch_names, save_path=save_path)
