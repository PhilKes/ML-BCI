"""
Handles all EEG-Data loading of the 'Human EEG Dataset for Brain-Computer Interface and Meditation' Dataset
"""
import math
import os
import time
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import VERBOSE, CONFIG, RESAMPLE
from data.MIDataLoader import MIDataLoader
from data.data_utils import slice_trials
from data.datasets.TrialsDataset import TrialsDataset
from data.datasets.lsmr21.lmsr21_matlab import LSMRSubjectRun
from data.datasets.lsmr21.lmsr_21_dataset import LSMR21, LSMR21Constants
from machine_learning.util import get_valid_trials_per_subject
from paths import datasets_folder
from util.misc import to_idxs_of_list, print_pretty_table, load_matlab, counts_of_list, calc_n_samples, save_dataframe


class LSMRNumpyRun:
    # shape: (trials, channels, samples)
    data: np.ndarray
    # per Trial: (label, tasknr, trial_category, artifact, triallength)
    # label:
    #  1: Left, 2: Right, 3: Up, 4: Down
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
        # trial_info[0]= label, trial_info[1]= tasknr
        n_class_trials = [i for i, td in enumerate(self.trial_info) if
                          (int(td[1]) in LSMR21.n_classes_tasks[n_class])]
        # Omit all Trials with label == 4 ('down') for n_class == 3
        if n_class == 3:
            n_class_trials = [n_class_trials[i] for i, td in enumerate(self.trial_info[n_class_trials])
                              if int(td[0]) != 4]
        return n_class_trials

    def get_labels(self, trials_idxs: List[int] = None, tmin=None):
        """
        Return int Labels of all Trials as numpy array
        :param tmin: Return only of Trials with minimum Time length of 'tmin'
        :param trials_idxs: Force to return only specified trials
        """
        if tmin is None:
            tmin = CONFIG.EEG.TMAX
        trials = self.get_trials(tmin=tmin) if trials_idxs is None else trials_idxs
        # trial_info[0] = label (targetnumber)
        return np.asarray([trial[0] for trial in [self.trial_info[i] for i in trials]], dtype=np.int)

    def get_data(self, trials_idxs: List[int] = None, tmin=None, ch_idxs=range(len(LSMR21.CHANNELS))) -> np.ndarray:
        """
        Return float EEG Data of all Trials as numpy array
        :param ch_idxs: Indexes of Channels to be used
        :param tmin: Return only Data of Trials with Time length of 'tmin' (defaults to CONFIG.EEG.TMAX)
        :param trials_idxs: Force to return only specified trials
        """
        if tmin is None:
            tmin = CONFIG.EEG.TMAX
        trials = self.get_trials(tmin=tmin) if trials_idxs is None else trials_idxs
        # Take samples from MI CUE Start (after 2s blank + 2s target pres.)
        # until after MI Cue + 1s
        min_sample = math.floor(CONFIG.EEG.TMIN * LSMR21.CONFIG.SAMPLERATE)
        max_sample = math.floor(LSMR21.CONFIG.SAMPLERATE * (tmin))
        # use ndarray.resize()
        data = np.zeros((0, len(ch_idxs), max_sample - min_sample), dtype=np.float)
        # elapsed = 0.0
        # start = time.time()
        # TODO Slicing takes ~ 0.7-1.5 Seconds for each Subject
        # data = np.resize(self.data[trials], (len(trials), len(ch_idxs), max_sample - min_sample))
        # data= np.vstack(data[:, :,:]).astype(np.float)
        for d in self.data[trials]:
            trial_data = d[ch_idxs, min_sample: max_sample]
            data = np.concatenate(
                (data, np.reshape(trial_data, (1, trial_data.shape[0], trial_data.shape[1]))))
        # print("Slicing Time: ", f"{time.time() - start:.2f}")
        return data

    def get_data_raw(self, trials=None):
        if trials is None:
            return self.data[:]
        return self.data[trials]

    def get_data_samples(self, n_class: int, ch_idxs=range(len(LSMR21.CHANNELS))):
        """
        Returns Samples of all n_class Trials
        :return: ndarray with shape (channels, samples)
        """
        raw = np.zeros((len(ch_idxs), 0), dtype=np.float32)
        for i in self.get_n_class_trials(n_class):
            x = self.data[i][ch_idxs, :]
            raw = np.append(raw, x, axis=1)
        return raw

    def get_trials(self, n_class=4, tmin=CONFIG.EEG.TMIN, artifact=CONFIG.EEG.ARTIFACTS,
                   trial_category=CONFIG.EEG.TRIAL_CATEGORY):
        """
        Get Trials indexes which have a minimum amount of Samples
        for t-seconds of Feedback Control period (Motorimagery Cue)
        :param tmin: Minimum Trial Time (shorter Trials are omitted)
        :return: List of Trials indexes
        """
        # Get Trial idxs of n_class Trials (correct Tasks)
        n_class_trials_idxs = self.get_n_class_trials(n_class)
        # print("n-class Trials: ", len(trials))
        # Filter out Trials that dont have enough samples (min. tmin * Samplerate)
        trials_idxs = [i for i in n_class_trials_idxs if self.data[i].shape[1] >= tmin * CONFIG.EEG.SAMPLERATE]
        # Filter out by trial_category (trialdata.result/forcedresult field)
        trials_idxs = [i for i in trials_idxs if self.trial_info[i, 2] >= trial_category]
        # Filter out by artifacts present or not if artifact = 0
        if (artifact is not None) and (artifact != 1):
            trials_idxs = [i for i in trials_idxs if self.trial_info[i, 3] == artifact]
        return trials_idxs

    def get_trials_tmin(self, tmins=np.arange(4, 11, 1)):
        s_t = []
        for tmin in tmins:
            s_t.append(len(self.get_trials(tmin=tmin)))
        return s_t

    def print_trials_with_tmins(self, tmins=np.arange(4, 11, 1)):
        """
        Print Table  with Trials with min. MI Cue Time
        """
        print(f"-- Subject {self.subject} --"
              f"Trials with at least n seconds of MI Cue Period --")
        df = pd.DataFrame([self.get_trials_tmin(tmins)], columns=tmins)
        print_pretty_table(df)


class LSMR21TrialsDataset(TrialsDataset):
    """
     TrialsDataset class Implementation for LSMR21 Dataset
    """

    def __init__(self, subjects, used_subjects, n_class, preloaded_tuple,
                 ch_names=LSMR21.CHANNELS, equal_trials=True):
        super().__init__(subjects, used_subjects, n_class, preloaded_tuple, ch_names, equal_trials)
        # 11 Runs, 62 Subjects, 75 Trials per Class per Subject
        self.n_trials_max = LSMR21DataLoader.get_subject_max_trials(n_class)
        # List containing amount of valid Trials per Subject (invalid Trials = -1)
        self.trials_per_subject = get_valid_trials_per_subject(self.preloaded_labels, self.subjects,
                                                               self.used_subjects, self.n_trials_max)
        self.print_stats()

    def print_stats(self, save_path=None):
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
        if save_path is not None:
            save_dataframe(df, save_path)
        print_pretty_table(df)


class LSMR21DataLoader(MIDataLoader):
    """
    MIDataLoader implementation for LSMR21 Dataset
    """

    CONSTANTS: LSMR21Constants = LSMR21
    ds_class = LSMR21TrialsDataset

    @classmethod
    def load_subjects_data(cls, subjects: List[int], n_class: int, ch_names: List[str] = LSMR21.CHANNELS,
                           equal_trials: bool = True, ignored_runs: List[int] = []):
        # 11 Runs, 62 Subjects, 75 Trials per Class
        subject_max_trials = cls.get_subject_max_trials(n_class)
        subjects_data = np.zeros((len(subjects), subject_max_trials, len(ch_names), CONFIG.EEG.SAMPLES),
                                 dtype=np.float32)
        subjects_labels = np.zeros((len(subjects), subject_max_trials), dtype=np.int)
        if RESAMPLE & (cls.CONSTANTS.CONFIG.SAMPLERATE != CONFIG.SYSTEM_SAMPLE_RATE):
            print(f"RESAMPLING from {cls.CONSTANTS.CONFIG.SAMPLERATE}Hz to {CONFIG.SYSTEM_SAMPLE_RATE}Hz")
        for i, subject in enumerate(tqdm(subjects)):
            s_data, s_labels = cls.load_subject(subject, n_class, ch_names)
            subjects_data[i] = s_data
            subjects_labels[i] = s_labels
        return subjects_data, subjects_labels

    @classmethod
    def create_n_class_loaders_from_subject(cls, used_subject: int, n_class: int, n_test_runs: int,
                                            batch_size: int, ch_names: List[str]):
        # 11 Runs, 62 Subjects, 75 Trials per Class
        subject_max_trials = cls.get_subject_max_trials(n_class)
        if RESAMPLE & (cls.CONSTANTS.CONFIG.SAMPLERATE != CONFIG.SYSTEM_SAMPLE_RATE):
            print(f"RESAMPLING from {cls.CONSTANTS.CONFIG.SAMPLERATE}Hz to {CONFIG.SYSTEM_SAMPLE_RATE}Hz")
        preloaded_data, preloaded_labels = cls.load_subject(used_subject, n_class, ch_names)
        preloaded_data = preloaded_data.reshape((preloaded_data.shape[0], 1, preloaded_data.shape[1],
                                                 preloaded_data.shape[2]))
        valid_trials = get_valid_trials_per_subject(np.expand_dims(preloaded_labels, 0), [used_subject],
                                                    [used_subject], subject_max_trials)[0]
        # Use 80% of the subject's data as Training Data, 20% as Test Data
        training_trials_size = math.floor(4 * valid_trials / 5)
        loader_train = cls.create_loader(preloaded_data[:training_trials_size],
                                         preloaded_labels[:training_trials_size], batch_size)
        loader_test = cls.create_loader(preloaded_data[training_trials_size:valid_trials],
                                        preloaded_labels[training_trials_size:valid_trials], batch_size)
        return loader_train, loader_test

    @classmethod
    def load_live_sim_data(cls, subject: int, n_class: int, ch_names: List[str]):
        """
                Load all neccessary Data for the Live Simulation Run of subject
                X: ndarray (channels,Samples) of single Subject Run data
                max_sample: Maximum sample number of the Run
                slices: Trial Slices
                trials_classes: ndarray with label nr. of every Trial in the Run
                trials_start_times: ndarray with Start Times of every Trial in the Run
                trial_sample_deltas: ndarray with Times of every Slice Timepoint in the Run
                """
        # Get Data from raw Run
        data, labels = cls.load_subject_run_raw(subject, LSMR21.runs[0])
        # data, labels = data[:data.shape[0] // 2], labels[:labels.shape[0] // 2]

        slices = CONFIG.EEG.TRIALS_SLICES
        # times = raw.times[:max_sample]
        trials_start_times = []
        trial_sample_deltas = []
        trials_start_samples = []
        samples_before = 0
        for t_idx in range(data.shape[0]):
            trial_sample_length = data[t_idx].shape[-1]
            trials_start_samples.append(samples_before)
            trials_start_times.append((1 / CONFIG.EEG.SAMPLERATE) * samples_before)
            # Get Trial Sample Nr. of each Slice Timepoint in the Trial
            for i in range(1, slices + 1):
                trial_sample_deltas.append(trials_start_samples[-1] + (trial_sample_length / slices) * i)
            samples_before += trial_sample_length

        trials_classes = labels

        X = cls.load_subject_samples_data(subject, LSMR21.runs[0])
        max_sample = X.shape[-1] // 2

        return X, max_sample, slices, trials_classes, np.asarray(trials_start_times), np.asarray(
            trials_start_samples), np.asarray(trial_sample_deltas)

    @classmethod
    def load_subject(cls, subject_idx, n_class, ch_names, runs=None, artifact=-1, trial_category=-1):
        """
        Load all Trials of all Runs of Subject
        :return: subject_data Numpy Array, subject_labels Numpy Array for all Subject's Trials
        """
        subject_max_trials = cls.get_subject_max_trials(n_class)
        # if artifact/trial_category = -1 use default values from config.py
        if artifact == -1:
            artifact = CONFIG.EEG.ARTIFACTS
        if trial_category == -1:
            trial_category = CONFIG.EEG.TRIAL_CATEGORY
        if runs is None:
            runs = LSMR21.runs
        subject_data = np.full((subject_max_trials, len(ch_names),
                                calc_n_samples(CONFIG.EEG.TMIN, CONFIG.EEG.TMAX, cls.CONSTANTS.CONFIG.SAMPLERATE)),
                               -1, dtype=np.float32)
        subject_labels = np.full((subject_max_trials), -1, dtype=np.int)
        t_idx = 0
        # Load Trials of every available Subject Run
        for run in runs:
            if VERBOSE:
                print("\n", f"Loading Subject {subject_idx + 1} Run {run}")
            start = time.time()

            sr = LSMR21DataLoader.load_subject_run(subject_idx + 1, run + 1)
            # Some Subjects have differing number of Runs -> Skip if missing
            if sr is None:
                continue
            # Get Trials idxs of correct n_class and minimum Sample size Trials
            trials_idxs = sr.get_trials(n_class, CONFIG.EEG.TMAX, artifact=artifact, trial_category=trial_category)
            data = sr.get_data(trials_idxs=trials_idxs,
                               ch_idxs=to_idxs_of_list(ch_names, LSMR21.CHANNELS))
            max_data_trial = t_idx + data.shape[0]
            subject_data[t_idx:max_data_trial] = data
            labels = sr.get_labels(trials_idxs=trials_idxs) - 1
            subject_labels[t_idx:max_data_trial] = labels
            t_idx += data.shape[0]
            elapsed = (time.time() - start)
            if VERBOSE:
                print(f"Loading + Slicing Time {subject_idx + 1}: {elapsed:.2f}")
        # Check if resampling or filtering has to be executed
        subject_data, subject_labels = cls.prepare_data_labels(subject_data, subject_labels)
        return subject_data, subject_labels

    @classmethod
    def load_subject_run_raw(cls, subject_idx, run, n_class=4):
        """
        Load all Trials of all Runs of Subject
        :return: subject_data Numpy Array, subject_labels Numpy Array for all Subject's Trials
        """
        sr = LSMR21DataLoader.load_subject_run(subject_idx + 1, run + 1)
        all_trials_idxs = sr.get_trials(tmin=0.0, n_class=n_class)
        subject_run_data = sr.get_data_raw(all_trials_idxs)
        subject_run_labels = sr.get_labels(trials_idxs=all_trials_idxs) - 1
        subject_run_data, subject_run_labels = cls.prepare_data_labels(subject_run_data, subject_run_labels)
        return subject_run_data, subject_run_labels

    @classmethod
    def load_subject_samples_data(cls, subject_idx, run, n_class=4):
        """
        Loads all Samples of the run
        :return: data ndarray with shape (channels,samples)
        """
        sr = LSMR21DataLoader.load_subject_run(subject_idx + 1, run + 1)
        return sr.get_data_samples(n_class)

    @classmethod
    def load_subject_run(cls, subject, run, from_matlab=False) -> LSMRNumpyRun:
        if VERBOSE:
            print("\n", f"Loading Subject {subject} Run {run}")
        try:
            if from_matlab:
                x = load_matlab(f"{datasets_folder}/{LSMR21.short_name}/matlab/S{subject}_Session_{run}")
                return LSMRSubjectRun(subject, x)
            else:
                path = f"{datasets_folder}/{LSMR21.short_name}/numpy/S{subject}_Session_{run}"
                return LSMRNumpyRun.from_npz(np.load(f"{path}.npz", allow_pickle=True))
        except FileNotFoundError as e:
            if VERBOSE:
                print(Exception(f"Missing: Subject {subject} Run {run}"))
            return None

    @staticmethod
    def get_subject_max_trials(n_class: int):
        """
        Returns the maximum Amount of possible Trials for a Subject for n_class Trials in all Runs
        """
        return len(LSMR21.runs) * (LSMR21.trials_per_class_per_sr * n_class) * CONFIG.EEG.TRIALS_SLICES

    @staticmethod
    def print_n_class_stats(save_path=None):
        """
        Prints all available Trials of all Subjects for n_class=2
        :param save_path: If present, save Stats to .txt files in save_path
        """
        CONFIG.EEG.set_config(LSMR21.CONFIG)
        for tmax in np.arange(1, 10, 1):
            print("Minimum MI Cue Time: ", tmax)
            CONFIG.EEG.set_times(tmax=tmax)
            for n_class in [2]:
                # failed_subjects = [1, 7, 8, 9, 14, 16, 18, 27, 28, 30, 40, 45, 49, 50, 53, 54, 57]:
                used_subjects = LSMR21.ALL_SUBJECTS
                preloaded_tuple = LSMR21DataLoader.load_subjects_data(used_subjects, n_class)
                ds = LSMR21TrialsDataset(used_subjects, used_subjects, n_class, preloaded_tuple)
                if save_path is not None:
                    ds.print_stats(save_path=os.path.join(save_path, f"LSMR21_stats-_tmin_{tmax}"))
