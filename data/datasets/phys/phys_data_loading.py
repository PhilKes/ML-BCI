"""
Handles all EEG-Data loading of Physionet Motor Imagery Dataset via MNE Library
(https://neuro.inf.unibe.ch/AlgorithmsNeuroscience/Tutorial_files/DataLoading.html)
On initial Run MNE downloads the Physionet Dataset into datasets_folder
(https://physionet.org/content/eegmmidb/1.0.0/)

Edition History:
2021-05-31: mne_load_subject_raw(): fmin, fmax explicitely set - ms
"""

import mne
import numpy as np
import torch  # noqa
from mne import Epochs
from mne.io import concatenate_raws, read_raw_edf
from torch.utils.data import Dataset, DataLoader, RandomSampler  # noqa
from torch.utils.data.dataset import TensorDataset  # noqa
from tqdm import tqdm

from config import VERBOSE, datasets_folder, CONFIG
from config import results_folder
from data.MIDataLoader import MIDataLoader
from data.data_utils import dec_label, increase_label, normalize_data, get_trials_size, \
    get_equal_trials_per_class, split_trials, get_runs_of_n_classes, get_data_from_raw, map_times_to_samples, \
    butter_bandpass_filt
from data.datasets.TrialsDataset import TrialsDataset
from data.datasets.phys.phys_dataset import PHYS
from util.misc import split_np_into_chunks, print_numpy_counts
from util.plot import matplot

# Dont print MNE loading logs
mne.set_log_level('WARNING')


class PHYSTrialsDataset(TrialsDataset):
    """
     TrialsDataset class Implementation for Physionet Dataset
    """

    def __init__(self, subjects, used_subjects, n_class, device, preloaded_tuple,
                 ch_names=PHYS.CHANNELS, equal_trials=True):
        super().__init__(subjects, used_subjects, n_class, device, preloaded_tuple, ch_names, equal_trials)

        self.trials_per_subject = get_trials_size(n_class, equal_trials) \
                                  * CONFIG.EEG.TRIALS_SLICES - PHYS.CONFIG.REST_TRIALS_LESS


class PHYSDataLoader(MIDataLoader):
    """
    MIDataLoader implementation for Physionet Dataset
    """
    name = PHYS.name
    name_short = PHYS.short_name
    available_subjects = PHYS.ALL_SUBJECTS
    folds = PHYS.cv_folds
    eeg_config = PHYS.CONFIG
    channels = PHYS.CHANNELS
    ds_class = PHYSTrialsDataset

    # Returns Train/Test Loaders containing all n_class Runs of subject
    # n_test_runs specifies how many Runs are reserved for Testing
    # 2/3class: 3 Runs, 4class: 6 Runs
    @classmethod
    def create_n_class_loaders_from_subject(cls, subject, n_class, n_test_runs, batch_size, ch_names, device):
        n_class_runs = get_runs_of_n_classes(n_class)
        train_runs = n_class_runs[:-n_test_runs]
        test_runs = n_class_runs[-n_test_runs:]
        loader_train = cls.create_loader_from_subject_runs(subject, n_class, batch_size, ch_names, device,
                                                           ignored_runs=test_runs)
        loader_test = cls.create_loader_from_subject_runs(subject, n_class, batch_size, ch_names, device,
                                                          ignored_runs=train_runs)
        return loader_train, loader_test

    # Creates Loader containing all Trials of n_class Runs of subject
    # ignored_runs[] will not be loaded
    @classmethod
    def create_loader_from_subject_runs(cls, subject, n_class, batch_size, ch_names, device,
                                        ignored_runs=[]):
        preloaded_data, preloaded_labels = cls.load_subjects_data([subject], n_class, ch_names,
                                                                  ignored_runs=ignored_runs)
        preloaded_data = preloaded_data.reshape((preloaded_data.shape[1], 1, preloaded_data.shape[2],
                                                 preloaded_data.shape[3]))
        preloaded_labels = preloaded_labels.reshape(preloaded_labels.shape[1])
        return cls.create_loader(preloaded_data, preloaded_labels, device, batch_size)

    @classmethod
    def load_subjects_data(cls, subjects, n_class, ch_names=PHYS.CHANNELS, equal_trials=True,
                           normalize=False, ignored_runs=[]):
        subjects.sort()
        trials = get_trials_size(n_class, equal_trials, ignored_runs)
        trials_per_run_class = np.math.floor(trials / n_class)
        trials = trials * CONFIG.EEG.TRIALS_SLICES
        if n_class > 2:
            trials -= PHYS.CONFIG.REST_TRIALS_LESS

        # print(CONFIG)
        preloaded_data = np.zeros((len(subjects), trials, len(ch_names), CONFIG.EEG.SAMPLES), dtype=np.float32)
        preloaded_labels = np.zeros((len(subjects), trials,), dtype=np.int)
        print("Preload Shape", preloaded_data.shape)
        for i, subject in tqdm(enumerate(subjects), total=len(subjects)):
            data, labels = cls.load_n_classes_tasks(subject, n_class, ch_names, equal_trials,
                                                    trials_per_run_class,
                                                    ignored_runs)
            # if data.shape[0] > preloaded_data.shape[1]:
            #     data, labels = data[:preloaded_data.shape[1]], labels[:preloaded_labels.shape[1]]
            if CONFIG.EEG.TRIALS_SLICES > 1:
                data, labels = split_trials(data, labels, CONFIG.EEG.TRIALS_SLICES, CONFIG.EEG.SAMPLES)
            preloaded_data[i] = data
            preloaded_labels[i] = labels
        if normalize:
            preloaded_data = normalize_data(preloaded_data)
        print("Trials per class loaded:")
        print_numpy_counts(preloaded_labels)
        # print(collections.Counter(preloaded_labels))
        return preloaded_data, preloaded_labels

    # Loads corresponding tasks for n_class Classification
    @classmethod
    def load_n_classes_tasks(cls, subject, n_class, ch_names=PHYS.CHANNELS, equal_trials=True,
                             trials_per_run_class=PHYS.TRIALS_PER_SUBJECT_RUN,
                             ignored_runs=[]):
        tasks = PHYS.n_classes_tasks[n_class].copy()
        if (not PHYS.CONFIG.REST_TRIALS_FROM_BASELINE_RUN) & (0 in tasks):
            tasks.remove(0)
        data, labels = cls.load_task_runs(subject, tasks,
                                          exclude_bothfists=(n_class == 4),
                                          exclude_rests=(n_class == 2),
                                          ch_names=ch_names, ignored_runs=ignored_runs,
                                          equal_trials=equal_trials,
                                          trials_per_run_class=trials_per_run_class,
                                          n_class=n_class)
        if n_class == 2:
            labels = dec_label(labels)
        return data, labels

    event_dict = {'T0': 1, 'T1': 2, 'T2': 3}

    # Loads Rest trials from the 1st baseline run of subject
    # if baseline run is not long enough for all needed trials
    # random Trials are generated from baseline run
    @classmethod
    def mne_load_rests(cls, subject, trials, ch_names, samples):
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

    # Merges runs from different tasks + correcting labels for n_class classification
    @classmethod
    def load_task_runs(cls, subject, tasks, exclude_bothfists=False, ch_names=PHYS.CHANNELS, n_class=3,
                       equal_trials=True, trials_per_run_class=PHYS.TRIALS_PER_SUBJECT_RUN, exclude_rests=False,
                       ignored_runs=[]):
        load_samples = CONFIG.EEG.SAMPLES * CONFIG.EEG.TRIALS_SLICES
        all_data = np.zeros((0, len(ch_names), load_samples))
        all_labels = np.zeros((0), dtype=np.int)
        # Load Subject Data of all Tasks
        for task_idx, task in enumerate(tasks):
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
                used_runs = [run for run in PHYS.runs[task] if run not in ignored_runs]
                data, labels = cls.mne_load_subject(subject, used_runs, event_id=tasks_event_dict,
                                                    ch_names=ch_names)
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
    @classmethod
    def mne_load_subject(cls, subject, runs, event_id='auto', ch_names=PHYS.CHANNELS, tmin=None,
                         tmax=None):
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
        return subject_data, subject_labels

    # Loads raw Subject run with specified channels
    # Can apply Bandpassfilter + Notch Filter
    @classmethod
    def mne_load_subject_raw(cls, subject, runs, ch_names=PHYS.CHANNELS, notch=False,
                             fmin=CONFIG.FILTER.FREQ_FILTER_HIGHPASS, fmax=CONFIG.FILTER.FREQ_FILTER_LOWPASS):

        fmin = CONFIG.FILTER.FREQ_FILTER_HIGHPASS
        fmax = CONFIG.FILTER.FREQ_FILTER_LOWPASS

        if VERBOSE:
            print(f"MNE loading Subject {subject} Runs {runs}")
        raw_fnames = PHYS.mne_dataset.load_data(subject, runs, datasets_folder)
        raw_files = [read_raw_edf(f, preload=True) for f in raw_fnames]
        raw = concatenate_raws(raw_files)
        raw.rename_channels(lambda x: x.strip('.'))
        raw.pick_channels(ch_names)
        if notch:
            picks = mne.pick_channels(raw.info['ch_names'], ch_names)
            raw.notch_filter(60.0, picks=picks, filter_length='auto',
                             phase='zero')
        if ((fmin is not None) | (fmax is not None)):
            # If method=”iir”, 4th order Butterworth will be used
            # iir_params = dict(order=7, ftype='butter', output='sos')
            # raw.filter(fmin, fmax, method='iir', iir_params=iir_params)
            # Apply butter bandpass filter to all channels
            raw.apply_function(butter_bandpass_filt, channel_wise=False,
                               lowcut=CONFIG.FILTER.FREQ_FILTER_HIGHPASS,
                               highcut=CONFIG.FILTER.FREQ_FILTER_LOWPASS,
                               fs=CONFIG.EEG.SAMPLERATE, order=7)
            raw.load_data()
        return raw


# Plots Subject Run with raw EEG Channel data
def plot_live_sim_subject_run(subject=1, n_class=3, save_path=f"{results_folder}/plots_training",
                              ch_names=PHYS.CHANNELS):
    # ch_names = ['F4', 'Oz', 'F7', 'F6']

    # Load Raw Subject Run for n_class
    raw = PHYSDataLoader.mne_load_subject_raw(subject, PHYS.n_classes_live_run[n_class], ch_names=ch_names)
    # Get Data from raw Run
    X = get_data_from_raw(raw)

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
