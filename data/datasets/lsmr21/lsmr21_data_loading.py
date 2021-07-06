"""
Handles all EEG-Data loading of the 'Human EEG Dataset for Brain-Computer Interface and Meditation' Dataset
"""
import math
from typing import List

import numpy as np
import pandas as pd
from scipy import io
from tqdm import tqdm

from config import eeg_config, datasets_folder, global_config
from data.MIDataLoader import MIDataLoader
from data.data_utils import normalize_data, get_trials_size, \
    split_trials
from data.datasets.TrialsDataset import TrialsDataset
from data.datasets.lsmr21.lmsr_21_dataset import LSMR21
from data.datasets.phys.phys_dataset import PHYS
from machine_learning.configs_results import get_global_config_str
from util.misc import print_numpy_counts, print_counts, copy_attrs, to_el_list, to_idxs_of_list


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

        # Load subject in buffer if not already
        if self.loaded_subject != subject_idx:
            self.loaded_subject_data, self.loaded_subject_labels = self.load_subject(subject_idx)
            self.loaded_subject = subject_idx

        return self.loaded_subject_data[local_trial_idx], self.loaded_subject_labels[local_trial_idx]

    def load_subject(self, subject_idx, runs=LSMR21.runs):
        """
        Load all Trials of all Runs of Subject
        :return: subject_data, subject_labels for all Subject's Trials
        """
        subject_data = np.zeros((0, len(self.ch_names), eeg_config.SAMPLES), dtype=np.float32)
        subject_labels = np.zeros((0), dtype=np.int)
        for run in runs:
            sr = LSMRSubjectRun(subject_idx + 1, LSMR21DataLoader.load_subject_run(subject_idx + 1, run))
            data = sr.get_data(eeg_config.TMAX, to_idxs_of_list(self.ch_names, LSMR21.CHANNELS))
            # trials_idxs = sr.get_trials_with_min_mi_time(eeg_config.TMAX)
            subject_data = np.concatenate(
                (subject_data, data))
            subject_labels = np.concatenate((subject_labels, sr.get_labels(eeg_config.TMAX)))
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

    @classmethod
    def load_subjects_data(cls, subjects, n_class, ch_names=PHYS.CHANNELS, equal_trials=True,
                           normalize=False, ignored_runs=[]):
        return None, None
        subjects.sort()
        trials = get_trials_size(n_class, equal_trials, ignored_runs)
        trials_per_run_class = np.math.floor(trials / n_class)
        trials = trials * eeg_config.TRIALS_SLICES

        print(get_global_config_str())
        preloaded_data = np.zeros((len(subjects), trials, len(ch_names), eeg_config.SAMPLES), dtype=np.float32)
        preloaded_labels = np.zeros((len(subjects), trials,), dtype=np.int)
        print("Preload Shape", preloaded_data.shape)
        for i, subject in tqdm(enumerate(subjects), total=len(subjects)):
            data, labels = cls.load_n_classes_tasks(subject, n_class, ch_names, equal_trials,
                                                    trials_per_run_class,
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

    @classmethod
    def load_subject_run(cls, subject, run):
        x = io.loadmat(f"{datasets_folder}/{LSMR21.short_name}/matlab/S{subject}_Session_{run}")['BCI']
        return x

    @classmethod
    def create_n_class_loaders_from_subject(cls, used_subject, n_class, n_test_runs, batch_size, ch_names, device):
        # TODO
        raise NotImplementedError('This method is not implemented!')

    @classmethod
    def mne_load_subject_raw(cls, subject, runs, ch_names=[], notch=False, fmin=global_config.FREQ_FILTER_HIGHPASS,
                             fmax=global_config.FREQ_FILTER_LOWPASS):
        # TODO
        raise NotImplementedError('This method is not implemented!')


class LSMRMetadata:
    mbsrsubject: int
    meditationpractice: str
    handedness: str
    instrument: str
    athlete: str
    handsport: str
    hobby: str
    gender: str
    age: int
    date: int
    day: int
    time: int

    def __init__(self, metadata):
        super().__init__()
        copy_attrs(self, metadata)


class LSMRTrialData:
    """
    Metadata of a single Trial of the LSMR-21 Dataset
    """
    # Corresponding Task number (1='Left/Right', 2='Up/Down',3='2D')
    tasknumber: int
    runnumber: int
    trialnumber: int
    # Presented target
    # 1='right', 2='left', 3='up', 4='down'
    targetnumber: int
    # Actually hit target
    targethitnumber: int
    # Time length of feedback control period (Subject tries to hit the target -> max 6.04s)
    triallength: float
    # TODO Time index for the end of the feedback control portion of the trial
    #  Length(trial) - 1000 -> means after target is hit 1 additional second is recorded?
    resultind: int
    # Result of the trial (1=correct target hit, 2=wrong target hit, NaN=Timeout)
    result: int
    # 1=correct target or cursor was closest to correct target
    # 0=wrong target or cursor was closest to wrong target
    forcedresult: int
    # 1= trial contains artifact, 0= no artifact
    artifact: int

    def __init__(self, trialdata):
        super().__init__()
        copy_attrs(self, trialdata)


class LSMRChanInfo:
    """
    Metadata of the Channel placement (Subject/Run specific!)
    """
    # if false, 'fiducials' and 'shape' are empty
    positionsrecorded: bool
    # Labels of Electrode names (10-10 system)
    label: List[str]
    # List of 'noisy' Channels (containing artifacts)
    noisechan: List[int]
    # 3D positions of electrodes
    electrodes: np.ndarray
    # Locations of the nasion/ preauricular points
    fiducials: np.ndarray
    # Location of the face shape information
    shape: np.ndarray

    def __init__(self, chaninfo):
        super().__init__()
        copy_attrs(self, chaninfo)
        self.label = to_el_list(self.label)
        if type(self.noisechan) == int:
            self.noisechan = [self.noisechan]


class LSMRSubjectRun:
    """
    Entire data of a Subject's single Run Matlab File
    """
    # Subject Nr (0 based index)
    subject: int
    ignore_metadata = False
    metadata: LSMRMetadata
    trialdata: List[LSMRTrialData]
    # Actual EEG Samples Recordings (62 Channels)
    data: np.ndarray
    # Time of every Sample in ms relative to Target Presentation
    # e.g '-2000' means the Sample is taken 2.0 secs before Target Presentation
    time: np.ndarray
    srate: int
    chaninfo: LSMRChanInfo

    def __init__(self, subject: int, matlab):
        super().__init__()
        self.subject = subject
        if matlab is None:
            return
        if not LSMRSubjectRun.ignore_metadata:
            self.metadata = LSMRMetadata(matlab['metadata'][0, 0][0, 0])
        self.trialdata = []
        for trialdata in matlab['TrialData'][0, 0][0]:
            self.trialdata.append(LSMRTrialData(trialdata))
        self.data = matlab['data'][0, 0][0]
        # for i,data in enumerate(self.data):
        #     self.data[i] = self.data[i].astype(np.float16)

        # samples = 8000
        # x = np.zeros((0, 62, samples), dtype=np.object)
        # for i in self.data:
        #     if i.shape[1] >= samples:
        #         x = np.concatenate((x, i[:, :samples].reshape(1, i.shape[0], samples)))
        # self.data = x
        self.time = matlab['time'][0, 0][0]
        self.srate = matlab['SRATE'][0, 0].item()
        self.chaninfo = LSMRChanInfo(matlab['chaninfo'][0, 0][0, 0])
        self.print_stats()
        self.print_trials_with_min_mi_time([eeg_config.TMAX])

    def get_trials_of_tasks(self, tasks):
        # TODO
        return [i for i in range(len(self.trialdata)) if self.trialdata[i].tasknumber in tasks]

    def get_labels(self, mi_tmin=eeg_config.TMAX):
        """
        Return int Labels of all Trials as numpy array
        :param mi_tmin: Return only of Trials with minimum MI Cue time of mi_tmin
        """
        trials = self.get_trials_with_min_mi_time(mi_tmin)
        return np.asarray([trial.targetnumber for trial in  [self.trialdata[i] for i in trials]], dtype=np.int)

    def get_data(self, mi_tmin=eeg_config.TMAX, ch_idxs=range(len(LSMR21.CHANNELS))):
        """
        Return float Data of all Trials as numpy array
        :param ch_idxs: Channel Idxs to be used
        :param mi_tmin: Return only of Trials with minimum MI Cue time of mi_tmin
        """
        trials = self.get_trials_with_min_mi_time(mi_tmin)
        # Take samples from MI CUE Start (after 2s blank + 2s target pres.)
        # until after MI Cue + 1s
        min_sample = math.floor(eeg_config.TMIN * eeg_config.SAMPLERATE)
        max_sample = math.floor(self.srate * (mi_tmin))
        data = np.zeros((0, len(ch_idxs), max_sample - min_sample), dtype=np.float)
        for d in self.data[trials]:
            trial_data = d[ch_idxs, min_sample: max_sample]
            data = np.concatenate(
                (data, np.reshape(trial_data, (1, trial_data.shape[0], trial_data.shape[1]))))
        return data

    # def get_labels(self, tasks):
    #     """
    #     Return int Labels of all Trials as numpy array
    #     :param tasks: Task numbers of trials to get
    #     """
    #     return np.asarray([trial.targetnumber for trial in self.trialdata if trial.tasknumber in tasks], dtype=np.int)
    #
    # def get_data(self, tasks):
    #     """
    #     Return float Data of all Trials as numpy array
    #     :param tasks: Task numbers of trials to get
    #    """
    #     return np.asarray([trial for trial in self.trialdata if trial.tasknumber in tasks], dtype=np.int)

    def get_trials_with_min_mi_time(self, t):
        """
        Get Trials which have a minimum amount of Samples for t-seconds of Feedback Control period (Motorimagery Cue)
        :param t: Minimum MI Cue Time (after 2s blank screen + 2s target presentation)
        :return: List of Trials indexes
        """

        return [i for i, d in enumerate(self.data) if d.shape[1] >= t * self.srate]

    def print_trials_with_min_mi_time(self, mi_tmins=[4, 5, 6, 7, 8, 9, 10, 11]):
        """
        Print Table  with Trials with min. MI Cue Time
        """
        s_t = []
        print(f"-- Subject {self.subject} Trials with at least n seconds of MI Cue Period --")
        for mi_tmin in mi_tmins:
            s_t.append(len(self.get_trials_with_min_mi_time(mi_tmin)))
        df = pd.DataFrame([s_t], columns=mi_tmins)
        print(df)

    def to_npz(self, path):
        # TODO savez or savez_compressed?
        #  Loading Time of S1_Session_1 in Seconds:
        #  scipy (600MB)   |   numpy (970MB)    |   numpy compr. (460MB)
        #      4.3         |      0.90          |      3.3
        #  Downsampling?
        np.savez(f"{path}.npz",
                 data=self.data,
                 time=self.time,
                 trialdata=np.asarray(self.trialdata),
                 metadata=self.metadata,
                 srate=self.srate
                 )

    @staticmethod
    def from_npz(path):
        npz = np.load(f"{path}.npz", allow_pickle=True)
        if all(attr in npz.files for attr in ['data', 'time', 'trialdata']):
            ds = LSMRSubjectRun(None)
            ds.data = npz['data']
            ds.time = npz['time']
            ds.trialdata = npz['trialdata'].tolist()
            ds.metadata = npz['metadata']
            ds.srate = npz['srate']
            return ds
        raise Exception("Incompatible .npz file provided!")

    def print_stats(self):
        # TODO Trials have highly varying nr. of Samples
        #    max. Samples per Trial: 11041 -> means Timeout (trialdata.result= NaN)
        #    if less than 11040 Samples, result either 0 or 1 (hit correct or wrong target)
        #    if a target was hit the Trial is finished -> 1 additional Second is recorded
        #    trialdata.resultind is the Time index of the end of the feedback control period (max 6s)
        #    What to do if less than 11041 Samples? Fill up with last present value?
        #    use trialdata.result or forcedresult?
        #    Trial Time Period: 2s blank Screen + 2s Target Presentation + 0-6s feedback control + 1s additional= 11s

        max_samples = 11040
        results = np.zeros(0, dtype=np.int)
        forcedresults = np.zeros(0, dtype=np.int)
        samples = np.zeros(0, dtype=np.int)
        artifacts = np.zeros(0, dtype=np.int)
        tasknrs = np.zeros(0, dtype=np.int)
        targets = np.zeros(0, dtype=np.int)
        for trial, i in enumerate(self.data):
            t = self.trialdata[trial]
            results = np.append(results, t.result)
            forcedresults = np.append(forcedresults, t.forcedresult)
            samples = np.append(samples, i.shape[1])
            artifacts = np.append(artifacts, t.artifact)
            tasknrs = np.append(tasknrs, t.tasknumber)
            targets = np.append(targets, t.targetnumber)
        # print("NaN results: "+str(np.count_nonzero(np.isnan(results))))
        print("--- Task Nrs (in blocks of 75) ---")
        print_counts(tasknrs)
        print("--- Total Trials ---")
        print(len(results), "\n")
        print("--- Targets---")
        print_counts(targets)
        print("--- Result Counts ---")
        print_counts(results)
        print("--- Forcedresult Counts ---")
        print_counts(forcedresults)
        print("--- Samples Counts ---")
        print(f"min: {np.min(samples)} max: {np.max(samples)}")
        print_counts(samples)
        print("--- Artifacts ---")
        print_counts(artifacts)
