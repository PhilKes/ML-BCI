import math
import time
from typing import List

import numpy as np
import pandas as pd

from config import eeg_config
from data.datasets.lsmr21.lmsr_21_dataset import LSMR21
from util.misc import print_counts, copy_attrs, to_el_list, print_pretty_table


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
        # self.print_stats()
        # self.print_trials_with_min_mi_time([eeg_config.TMAX])

    def get_trials_of_tasks(self, tasks: List[int], ignore_target=None):
        """
        Get Trial idxs corresponding to given Tasks
        :param ignore_target: Omit Trials with specified targetnumber, e.g. 4 = 'Down' (Task 2/3)
        :return: List of Trial idxs
        """
        return [i for i, td in enumerate(self.trialdata) if
                (td.tasknumber in tasks) and (td.targetnumber != ignore_target)]

    def get_labels(self, trials_idxs: List[int] = None, mi_tmin=eeg_config.TMAX):
        """
        Return int Labels of all Trials as numpy array
        :param mi_tmin: Return only of Trials with minimum MI Cue time of mi_tmin
        :param trials_idxs: Force to return only specified trials
        """
        trials = self.get_trials(tmin=mi_tmin) if trials_idxs is None else trials_idxs
        return np.asarray([trial.targetnumber for trial in [self.trialdata[i] for i in trials]], dtype=np.int)

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
        max_sample = math.floor(self.srate * (mi_tmin))
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

    def get_trials(self, n_class=4, tmin=eeg_config.TMIN, ignore_target=None):
        """
        Get Trials indexes which have a minimum amount of Samples
        for t-seconds of Feedback Control period (Motorimagery Cue)
        :param t: Minimum MI Cue Time (after 2s blank screen + 2s target presentation)
        :return: List of Trials indexes
        """
        # TODO Which Trials for 3class?
        #  3class: Left/Right/Up?
        # Get Trial idxs of n_class Trials (correct Tasks)
        trials = self.get_trials_of_tasks(LSMR21.n_classes_tasks[n_class], ignore_target=ignore_target)
        # Filter out Trials that dont have enough samples (min. mi_tmin * Samplerate)
        return [i for i in trials if self.data[i].shape[1] >= tmin * self.srate]

    def get_trials_tmin(self, mi_tmins=np.arange(4, 11, 1)):
        s_t = []
        for mi_tmin in mi_tmins:
            s_t.append(len(self.get_trials(tmin=mi_tmin)))
        return s_t

    def print_trials_with_min_mi_time(self, mi_tmins=[4, 5, 6, 7, 8, 9, 10, 11]):
        """
        Print Table  with Trials with min. MI Cue Time
        """
        print(f"-- Subject {self.subject} Run {self.trialdata[0].runnumber} "
              f"Trials with at least n seconds of MI Cue Period --")
        df = pd.DataFrame([self.get_trials_tmin(mi_tmins)], columns=mi_tmins)
        print_pretty_table(df)

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
