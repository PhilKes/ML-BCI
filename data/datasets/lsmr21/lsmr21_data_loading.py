"""
Handles all EEG-Data loading of the 'Human EEG Dataset for Brain-Computer Interface and Meditation' Dataset
"""
import math
import time
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import datasets_folder, VERBOSE, CONFIG
from data.MIDataLoader import MIDataLoader
from data.datasets.TrialsDataset import TrialsDataset
from data.datasets.lsmr21.lmsr21_matlab import LSMRSubjectRun
from data.datasets.lsmr21.lmsr_21_dataset import LSMR21
from data.datasets.phys.phys_dataset import PHYS
from machine_learning.util import get_valid_trials_per_subject
from util.misc import to_idxs_of_list, print_pretty_table, load_matlab, counts_of_list


class LSMRNumpyRun:
    # shape: (trials, channels, samples)
    data: np.ndarray
    # per Trial: (label, tasknr, trial_category, artifact, triallength)
    trial_info: np.ndarray
    subject: int

    def __init__(self):
        super().__init__()

    @staticmethod
    def from_npz(npz):
        if all(attr in npz.files for attr in ['data', 'trial_info']):
            ds = LSMRNumpyRun()
            ds.data = npz['data']
            ds.trial_info = npz['trial_info']
            ds.subject = npz['subject']
            return ds
        raise Exception("Incompatible .npz file provided for LSMRNumpyRun!")

    def get_n_class_trials(self, n_class: int):
        """
        Get Trial idxs corresponding to n_class Classifcation
        :return: List of Trial idxs
        """
        # Get all trials with correct tasknr
        # trial_info[1]= tasknr
        return [i for i, td in enumerate(self.trial_info) if
                (td[1] in LSMR21.n_classes_tasks[n_class])]

    def get_labels(self, trials_idxs: List[int] = None, mi_tmin=CONFIG.EEG.TMAX):
        """
        Return int Labels of all Trials as numpy array
        :param mi_tmin: Return only of Trials with minimum MI Cue time of mi_tmin
        :param trials_idxs: Force to return only specified trials
        """
        trials = self.get_trials(tmin=mi_tmin) if trials_idxs is None else trials_idxs
        # trial_info[0]= label (targetnumber)
        return np.asarray([trial[0] for trial in [self.trial_info[i] for i in trials]], dtype=np.int)

    def get_data(self, trials_idxs: List[int] = None, mi_tmin=None, ch_idxs=range(len(LSMR21.CHANNELS))) -> np.ndarray:
        """
        Return float Data of all Trials as numpy array
        :param ch_idxs: Channel Idxs to be used
        :param mi_tmin: Return only of Trials with minimum MI Cue time of mi_tmin
        :param trials_idxs: Force to return only specified trials
        """
        if mi_tmin is None:
            mi_tmin = CONFIG.EEG.TMAX
        trials = self.get_trials(tmin=mi_tmin) if trials_idxs is None else trials_idxs
        # Take samples from MI CUE Start (after 2s blank + 2s target pres.)
        # until after MI Cue + 1s
        min_sample = math.floor(CONFIG.EEG.TMIN * CONFIG.EEG.SAMPLERATE)
        max_sample = math.floor(CONFIG.EEG.SAMPLERATE * (mi_tmin))
        # use ndarray.resize()
        data = np.zeros((0, len(ch_idxs), max_sample - min_sample), dtype=np.float)
        elapsed = 0.0
        start = time.time()
        # TODO Slicing takes ~ 0.7-1.5 Seconds for each Subject
        # data = np.resize(self.data[trials], (len(trials), len(ch_idxs), max_sample - min_sample))
        # data= np.vstack(data[:, :,:]).astype(np.float)
        for d in self.data[trials]:
            trial_data = d[ch_idxs, min_sample: max_sample]
            data = np.concatenate(
                (data, np.reshape(trial_data, (1, trial_data.shape[0], trial_data.shape[1]))))
        # print("Slicing Time: ", f"{time.time() - start:.2f}")
        return data

    def get_trials(self, n_class=4, tmin=CONFIG.EEG.TMIN, artifact=CONFIG.EEG.ARTIFACTS,
                   trial_category=CONFIG.EEG.TRIAL_CATEGORY):
        """
        Get Trials indexes which have a minimum amount of Samples
        for t-seconds of Feedback Control period (Motorimagery Cue)
        :param tmin: Minimum MI Cue Time (after 2s blank screen + 2s target presentation)
        :return: List of Trials indexes
        """
        # Get Trial idxs of n_class Trials (correct Tasks)
        n_class_trials_idxs = self.get_n_class_trials(n_class)
        # print("n-class Trials: ", len(trials))
        # Filter out Trials that dont have enough samples (min. mi_tmin * Samplerate)
        trials_idxs = [i for i in n_class_trials_idxs if self.data[i].shape[1] >= tmin * CONFIG.EEG.SAMPLERATE]
        # Filter out by trial_category (trialdata.result/forcedresult field)
        trials_idxs = [i for i in trials_idxs if self.trial_info[i, 2] >= trial_category]
        # Filter out by artifacts present or not if artifact = 0
        if (artifact is not None) and (artifact != 1):
            trials_idxs = [i for i in trials_idxs if self.trial_info[i, 3] == artifact]
        return trials_idxs

    def get_trials_tmin(self, mi_tmins=np.arange(4, 11, 1)):
        s_t = []
        for mi_tmin in mi_tmins:
            s_t.append(len(self.get_trials(tmin=mi_tmin)))
        return s_t

    def print_trials_with_min_mi_time(self, mi_tmins=np.arange(4, 11, 1)):
        """
        Print Table  with Trials with min. MI Cue Time
        """
        print(f"-- Subject {self.subject} --"
              f"Trials with at least n seconds of MI Cue Period --")
        df = pd.DataFrame([self.get_trials_tmin(mi_tmins)], columns=mi_tmins)
        print_pretty_table(df)


class LSMR21TrialsDataset(TrialsDataset):
    """
     TrialsDataset class Implementation for LSMR21 Dataset
    """

    def __init__(self, subjects, used_subjects, n_class, device, preloaded_tuple,
                 ch_names=LSMR21.CHANNELS, equal_trials=True):
        super().__init__(subjects, used_subjects, n_class, device, preloaded_tuple, ch_names, equal_trials)
        # 11 Runs, 62 Subjects, 75 Trials per Class per Subject
        self.n_trials_max = len(LSMR21.runs) * (LSMR21.trials_per_class_per_sr * n_class)
        # List containing amount of valid Trials per Subject (invalid Trials = -1)
        self.trials_per_subject = get_valid_trials_per_subject(self.preloaded_labels, self.subjects,
                                                               self.used_subjects, self.n_trials_max)

        self.print_stats()

    def print_stats(self):
        """
        Prints Amount of Trials per Subject, per Class, Totals
        as Table
        """
        trials_per_subject_per_class = []
        # Get Trials per Subject per Class
        for s_idx, subject in enumerate(self.subjects):
            counts_per_class = counts_of_list(self.preloaded_labels[s_idx]).tolist()
            # Disregard invalid Trials (where label=-1, last index of counts_of_list)
            counts_per_class = counts_per_class[:-1]
            trials_per_subject_per_class.append(counts_per_class)
            # Calculate total amount of trials per Subject
            trials_per_subject_per_class[s_idx].append(sum(trials_per_subject_per_class[s_idx]))
        # Get Total Amounts per Class as last Row
        totals_per_class = []
        for n_class in range(self.n_class):
            # Get all Trials of n_class
            # s=trials_per_subject_per_class[:,n_class]
            s = [s_cl[n_class] for s_cl in trials_per_subject_per_class]
            totals_per_class.append(sum(s))
        # Add Total Amount of Trials in last Row
        totals_per_class.append(sum(totals_per_class))
        trials_per_subject_per_class.append(totals_per_class)

        df = pd.DataFrame(trials_per_subject_per_class,
                          columns=[n_class for n_class in range(self.n_class)] + ['Total'],
                          index=[f"S{subject}" for subject in self.subjects] + ['Total'])
        print_pretty_table(df)


class LSMR21DataLoader(MIDataLoader):
    """
    MIDataLoader implementation for LSMR21 Dataset
    """
    name = LSMR21.name
    name_short = LSMR21.short_name
    available_subjects = LSMR21.ALL_SUBJECTS
    folds = LSMR21.cv_folds
    eeg_config = LSMR21.CONFIG
    channels = LSMR21.CHANNELS
    ds_class = LSMR21TrialsDataset

    # sampler = SubjectTrialsRandomSampler

    @classmethod
    def load_subjects_data(cls, subjects, n_class, ch_names=PHYS.CHANNELS, equal_trials=True,
                           normalize=False, ignored_runs=[]):
        # 11 Runs, 62 Subjects, 75 Trials per Class
        n_subject_trials_max = len(LSMR21.runs) * (LSMR21.trials_per_class_per_sr * n_class)
        subjects_data = np.zeros((len(subjects), n_subject_trials_max, len(ch_names), CONFIG.EEG.SAMPLES),
                                 dtype=np.float32)
        subjects_labels = np.zeros((len(subjects), n_subject_trials_max), dtype=np.int)
        for i, subject in enumerate(tqdm(subjects)):
            s_data, s_labels = cls.load_subject(subject, n_class, ch_names, n_subject_trials_max)
            subjects_data[i] = s_data
            subjects_labels[i] = s_labels
        return subjects_data, subjects_labels

    @classmethod
    def load_subject(cls, subject_idx, n_class, ch_names, n_trials_max, runs=None, artifact=-1,
                     trial_category=-1):
        """
        Load all Trials of all Runs of Subject
        :return: subject_data Numpy Array, subject_labels Numpy Array for all Subject's Trials
        """
        # if artifact/trial_category = -1 use default values from config.py
        if artifact == -1:
            artifact = CONFIG.EEG.ARTIFACTS
        if trial_category == -1:
            trial_category = CONFIG.EEG.TRIAL_CATEGORY
        if runs is None:
            runs = LSMR21.runs
        subject_data = np.full((n_trials_max, len(ch_names), CONFIG.EEG.SAMPLES), -1, dtype=np.float32)
        subject_labels = np.full((n_trials_max), -1, dtype=np.int)
        t_idx = 0
        # Load Trials of every available Subject Run
        for run in runs:
            if VERBOSE:
                print("\n", f"Loading Subject {subject_idx + 1} Run {run}")
            start = time.time()
            try:
                sr = LSMR21DataLoader.load_subject_run(subject_idx + 1, run + 1)
            except FileNotFoundError as e:
                if VERBOSE:
                    print(f"Skipped missing Subject {subject_idx + 1} Run {run + 1}")
                continue
            # Get Trials idxs of correct n_class and minimum Sample size
            trials_idxs = sr.get_trials(n_class, CONFIG.EEG.TMAX, artifact=artifact, trial_category=trial_category)
            data = sr.get_data(trials_idxs=trials_idxs,
                               ch_idxs=to_idxs_of_list([ch.upper() for ch in ch_names], LSMR21.CHANNELS))
            max_data_trial = t_idx + data.shape[0]
            subject_data[t_idx:max_data_trial] = data
            subject_labels[t_idx:max_data_trial] = sr.get_labels(trials_idxs=trials_idxs) - 1
            t_idx += data.shape[0]
            elapsed = (time.time() - start)
            if VERBOSE:
                print(f"Loading + Slicing Time {subject_idx + 1}: {elapsed:.2f}")
        return subject_data, subject_labels

    @classmethod
    def load_subject_run(cls, subject, run, from_matlab=False) -> LSMRNumpyRun:
        # TODO Remove matlab
        if from_matlab:
            x = load_matlab(f"{datasets_folder}/{LSMR21.short_name}/matlab/S{subject}_Session_{run}")
            return LSMRSubjectRun(subject, x)
        else:
            path = f"{datasets_folder}/{LSMR21.short_name}/numpy/S{subject}_Session_{run}"
            return LSMRNumpyRun.from_npz(np.load(f"{path}.npz", allow_pickle=True))

    @classmethod
    def create_n_class_loaders_from_subject(cls, used_subject, n_class, n_test_runs, batch_size, ch_names, device):
        # 11 Runs, 62 Subjects, 75 Trials per Class
        n_subject_trials_max = len(LSMR21.runs) * (LSMR21.trials_per_class_per_sr * n_class)
        preloaded_data, preloaded_labels = cls.load_subject(used_subject, n_class, ch_names, n_subject_trials_max)
        preloaded_data = preloaded_data.reshape((preloaded_data.shape[0], 1, preloaded_data.shape[1],
                                                 preloaded_data.shape[2]))
        valid_trials = get_valid_trials_per_subject(np.expand_dims(preloaded_labels, 0), [used_subject],
                                                    [used_subject], n_subject_trials_max)[0]
        # Use 80% of the subject's data as Training Data, 20% as Test Data
        training_trials_size = math.floor(4 * valid_trials / 5)
        loader_train = cls.create_loader(preloaded_data[:training_trials_size],
                                         preloaded_labels[:training_trials_size], device, batch_size)
        loader_test = cls.create_loader(preloaded_data[training_trials_size:valid_trials],
                                        preloaded_labels[training_trials_size:valid_trials], device, batch_size)
        return loader_train, loader_test

    @classmethod
    def mne_load_subject_raw(cls, subject, runs, ch_names=[], notch=False, fmin=CONFIG.FILTER.FREQ_FILTER_HIGHPASS,
                             fmax=CONFIG.FILTER.FREQ_FILTER_LOWPASS):
        # TODO
        raise NotImplementedError('This method is not implemented!')
