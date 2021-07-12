"""
Handles all EEG-Data loading of the 'Human EEG Dataset for Brain-Computer Interface and Meditation' Dataset
"""
import math
import time
from typing import List

import numpy as np
import pandas as pd

from config import eeg_config, datasets_folder, global_config
from data.MIDataLoader import MIDataLoader
from data.datasets.TrialsDataset import TrialsDataset
from data.datasets.lsmr21.lmsr21_matlab import LSMRSubjectRun
from data.datasets.lsmr21.lmsr_21_dataset import LSMR21
from data.datasets.phys.phys_dataset import PHYS
from machine_learning.util import SubjectTrialsRandomSampler
from util.misc import to_idxs_of_list, print_pretty_table, load_matlab


class LSMRNumpyRun:
    data: np.ndarray
    # per Trial: (label,tasknr, trial_category, artifact, triallength)
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
        raise Exception("Incompatible .npz file provided!")

    def get_n_class_trials(self, n_class: int):
        """
        Get Trial idxs corresponding to n_class Classifcation
        :return: List of Trial idxs
        """
        # Get all trials with correct tasknr
        # trial_info[1]= tasknr
        return [i for i, td in enumerate(self.trial_info) if
                (td[1] in LSMR21.n_classes_tasks[n_class])]

    def get_labels(self, trials_idxs: List[int] = None, mi_tmin=eeg_config.TMAX):
        """
        Return int Labels of all Trials as numpy array
        :param mi_tmin: Return only of Trials with minimum MI Cue time of mi_tmin
        :param trials_idxs: Force to return only specified trials
        """
        trials = self.get_trials(tmin=mi_tmin) if trials_idxs is None else trials_idxs
        # trial_info[0]= label (targetnumber)
        return np.asarray([trial[0] for trial in [self.trial_info[i] for i in trials]], dtype=np.int)

    def get_data(self, trials_idxs: List[int] = None, mi_tmin=None, ch_idxs=range(len(LSMR21.CHANNELS))):
        """
        Return float Data of all Trials as numpy array
        :param ch_idxs: Channel Idxs to be used
        :param mi_tmin: Return only of Trials with minimum MI Cue time of mi_tmin
        :param trials_idxs: Force to return only specified trials
        """
        if mi_tmin is None:
            mi_tmin = eeg_config.TMAX
        trials = self.get_trials(tmin=mi_tmin) if trials_idxs is None else trials_idxs
        # Take samples from MI CUE Start (after 2s blank + 2s target pres.)
        # until after MI Cue + 1s
        min_sample = math.floor(eeg_config.TMIN * eeg_config.SAMPLERATE)
        max_sample = math.floor(eeg_config.SAMPLERATE * (mi_tmin))
        # use ndarray.resize()
        data = np.zeros((0, len(ch_idxs), max_sample - min_sample), dtype=np.float)
        elapsed = 0.0
        start = time.time()
        # TODO Slicing takes ~ 1.5 Seconds for each Subject
        # data = np.resize(self.data[trials], (len(trials), len(ch_idxs), max_sample - min_sample))
        # data= np.vstack(data[:, :,:]).astype(np.float)
        for d in self.data[trials]:
            trial_data = d[ch_idxs, min_sample: max_sample]
            data = np.concatenate(
                (data, np.reshape(trial_data, (1, trial_data.shape[0], trial_data.shape[1]))))
        print("Slicing Time: ", f"{time.time() - start:.2f}")
        return data

    def get_trials(self, n_class=4, tmin=eeg_config.TMIN, artifact=0, trial_category=0):
        """
        Get Trials indexes which have a minimum amount of Samples
        for t-seconds of Feedback Control period (Motorimagery Cue)
        :param tmin: Minimum MI Cue Time (after 2s blank screen + 2s target presentation)
        :return: List of Trials indexes
        """
        # Get Trial idxs of n_class Trials (correct Tasks)
        trials = self.get_n_class_trials(n_class)
        print("n-class Trials: ", len(trials))
        # TODO Filter out with artifact + trial_category
        # Filter out Trials that dont have enough samples (min. mi_tmin * Samplerate)
        return [i for i in trials if self.data[i].shape[1] >= tmin * eeg_config.SAMPLERATE]

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

    def __init__(self, subjects, n_classes, device, preloaded_tuple,
                 ch_names=LSMR21.CHANNELS, equal_trials=True):
        super().__init__(subjects, n_classes, device, preloaded_tuple, ch_names, equal_trials)

        self.trials_per_subject = LSMR21.trials_per_subject * eeg_config.TRIALS_SLICES
        self.loaded_subject = None
        self.loaded_subject_data, self.loaded_subject_labels = None, None

    def load_trial(self, trial):
        local_trial_idx = trial % self.trials_per_subject

        # determine required subject for trial
        subject_idx = int(trial / self.trials_per_subject)

        # print(f"S{subject_idx+1} T{local_trial_idx} global T{trial}")
        # TODO RandomSampler switches between subjects all the time
        #  -> e.g. Subject 1 is loaded, Datasets only wants 1 Trial of that Subject
        #  -> immediately needs to load another subject -> very inefficient
        #   introduced util.SubjectTrialsRandomSampler
        # Load subject in buffer if not already
        if self.loaded_subject != subject_idx:
            self.loaded_subject_data, self.loaded_subject_labels = self.load_subject(subject_idx)
            self.loaded_subject = subject_idx

        return self.loaded_subject_data[local_trial_idx], self.loaded_subject_labels[local_trial_idx]

    def load_subject(self, subject_idx, runs=LSMR21.runs):
        """
        Load all Trials of all Runs of Subject
        :return: subject_data Numpy Array, subject_labels Numpy Array for all Subject's Trials
        """
        subject_data = np.zeros((0, len(self.ch_names), eeg_config.SAMPLES), dtype=np.float32)
        subject_labels = np.zeros((0), dtype=np.int)
        elapsed = 0.0
        for run in runs:
            print("\n", f"Loading Subject {subject_idx + 1} Run {run}")
            start = time.time()

            sr = LSMR21DataLoader.load_subject_run(subject_idx + 1, run)
            # Get Trials idxs of correct n_class and minimum Sample size
            trials_idxs = sr.get_trials(self.n_class, eeg_config.TMAX)
            data = sr.get_data(trials_idxs=trials_idxs, ch_idxs=to_idxs_of_list(self.ch_names, LSMR21.CHANNELS))
            subject_data = np.concatenate((subject_data, data))
            subject_labels = np.concatenate((subject_labels, sr.get_labels(trials_idxs=trials_idxs) - 1))
            elapsed = (time.time() - start)
            print(f"Loading + Slicing Time {subject_idx + 1}: {elapsed:.2f}")

        # print_counts(subject_labels)
        # TODO Compare Matlab vs Numpy -> subject_data shape should have same amount of Trials
        print(subject_data.shape)
        return subject_data, subject_labels


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
    sampler = SubjectTrialsRandomSampler

    @classmethod
    def load_subjects_data(cls, subjects, n_class, ch_names=PHYS.CHANNELS, equal_trials=True,
                           normalize=False, ignored_runs=[]):
        return None, None

    @classmethod
    def load_subject_run(cls, subject, run, from_matlab=False):
        # TODO numpy/matlab?
        if from_matlab:
            x = load_matlab(f"{datasets_folder}/{LSMR21.short_name}/matlab/S{subject}_Session_{run}")
            return LSMRSubjectRun(subject, x)
        else:
            path = f"{datasets_folder}/{LSMR21.short_name}/numpy/S{subject}_Session_{run}"
            return LSMRNumpyRun.from_npz(np.load(f"{path}.npz", allow_pickle=True))

    @classmethod
    def create_n_class_loaders_from_subject(cls, used_subject, n_class, n_test_runs, batch_size, ch_names, device):
        # TODO
        raise NotImplementedError('This method is not implemented!')

    @classmethod
    def mne_load_subject_raw(cls, subject, runs, ch_names=[], notch=False, fmin=global_config.FREQ_FILTER_HIGHPASS,
                             fmax=global_config.FREQ_FILTER_LOWPASS):
        # TODO
        raise NotImplementedError('This method is not implemented!')
