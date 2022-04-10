"""
File: bcic_data_loading.py

Description:
  Handles all EEG-Data loading of BCI competition IV 2a Motor Imagery Dataset.
  Software structure and architecture is taken from file 'data_loading.pc'
  developed by P. Kessler.

Author: Manfred Strahnen (based on the template given by Philipp Kessler
        file: lsmr21_data_loading.py)

History:
  2021-05-10: Getting started - ms (Manfred Strahnen
"""
import logging
import math
from pathlib import Path
from typing import List

import numpy as np
from PyQt5.QtCore import QThread

from app.config import CONFIG, RESAMPLE
from app.data.MIDataLoader import MIDataLoader
from app.data.datasets.TrialsDataset import TrialsDataset
from app.data.datasets.bcic.bcic_dataset import BCIC, BCICConstants
from app.data.datasets.bcic.bcic_iv2a_dataset import BCIC_IV2a_dataset, plot_psds
from app.machine_learning.util import get_valid_trials_per_subject, calc_slice_start_samples
from app.ui.long_operation import is_thread_running


class BCICTrialsDataset(TrialsDataset):
    """
     TrialsDataset class Implementation for BCIC Dataset
    """

    def __init__(self, subjects, used_subjects, n_class, preloaded_tuple,
                 ch_names=BCIC.CHANNELS, equal_trials=True):
        super().__init__(subjects, used_subjects, n_class, preloaded_tuple, ch_names, equal_trials)

        # max number of trials (which is the same for each subject
        # 6 runs with 12 trials per class per subject
        self.n_trials_max = 6 * 12 * self.n_class * CONFIG.EEG.TRIALS_SLICES

        # TODO incorrect trials_per_subject if using trials_sliced_training?
        # number of valid trials per subject is different for each subject, because
        # some trials are marked as artifact
        self.trials_per_subject = get_valid_trials_per_subject(self.preloaded_labels, self.subjects,
                                                               self.used_subjects, self.n_trials_max)

        # Only for testing !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #        for subject_idx in range(len(self.subjects)):
        #            self.trials_per_subject[subject_idx] = self.n_trials_max

        logging.info("trials_per_subject: %s", self.trials_per_subject)


class BCICDataLoader(MIDataLoader):
    """
    MI_DataLoader implementation for BCIC Dataset
    """

    CONSTANTS: BCICConstants = BCIC
    ds_class = BCICTrialsDataset

    @classmethod
    def load_subjects_data(cls, subjects: List[int], n_class: int, ch_names: List[str] = BCIC.CHANNELS,
                           equal_trials: bool = True, ignored_runs: List[int] = [], qthread: QThread = None):
        subjects.sort()
        # preloaded_data, preloaded_labels = ds_w.load_subjects_data(training)
        subject_trials_max = 6 * 12 * n_class * CONFIG.EEG.TRIALS_SLICES
        preloaded_data = np.zeros(
            (len(subjects), subject_trials_max, len(ch_names), CONFIG.EEG.SAMPLES))
        preloaded_labels = np.full((len(subjects), subject_trials_max), -1)
        if RESAMPLE & (cls.CONSTANTS.CONFIG.SAMPLERATE != CONFIG.SYSTEM_SAMPLE_RATE):
            logging.info(f"RESAMPLING from {cls.CONSTANTS.CONFIG.SAMPLERATE}Hz to {CONFIG.SYSTEM_SAMPLE_RATE}Hz")
        for s_idx, subject in enumerate(subjects):
            preloaded_data[s_idx], preloaded_labels[s_idx] = cls.load_subject(subject, n_class, ch_names)
            # Check if thread was stopped
            if is_thread_running(qthread):
                return preloaded_data, preloaded_labels
        cls.print_stats(preloaded_labels)

        return preloaded_data, preloaded_labels

    @classmethod
    def load_subject(cls, subject: int, n_class: int, ch_names: List[str] = BCIC.CHANNELS):
        subject_data, subject_labels = BCIC_IV2a_dataset.get_trials(subject, n_class, ch_names)
        subject_data, subject_labels = cls.prepare_data_labels(subject_data, subject_labels)

        return subject_data, subject_labels

    @classmethod
    def create_n_class_loaders_from_subject(cls, used_subject: int, n_class: int, n_test_runs: List[int],
                                            batch_size: int, ch_names: List[str]):
        preloaded_data, preloaded_labels = BCIC_IV2a_dataset.get_trials(used_subject, n_class, ch_names)
        preloaded_data, preloaded_labels = cls.prepare_data_labels(preloaded_data, preloaded_labels)
        preloaded_data = preloaded_data.reshape((preloaded_data.shape[0], 1, preloaded_data.shape[1],
                                                 preloaded_data.shape[2]))
        n_trials_max = 6 * 12 * n_class  # 6 runs with 12 trials per class per subject
        valid_trials = get_valid_trials_per_subject(np.expand_dims(preloaded_labels, 0), [used_subject],
                                                    [used_subject], n_trials_max)[0]
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
        Load all necessary Data for the Live Simulation Run of subject
        X: ndarray (channels,Samples) of single Subject's Run data
        max_sample: Maximum sample number of the Run
        slices: Trial Slices
        trials_classes: ndarray with label nr. of every Trial in the Run
        trials_start_times: ndarray with Start Times of every Trial in the Run
        trials_start_samples: ndarray with Start Samples of every Trial in the Run
        slice_start_samples: ndarray with Start Samples of every Slice in the Run
        """
        # TODO Are samples and Trials loaded correctly from raw data?
        #  -> see end of last Plot of a BCIC Live Simlation -> missing Trials at the end?
        X, trials_classes, trials_start_times, trials_start_samples, trials_samples_length = BCIC_IV2a_dataset.get_raw_run_data(
            subject,
            n_class,
            ch_names)
        # TODO Resampling not necessary since BCIC and global System SampleRate is the same
        #  if Samplerates are different, have to resample X nad recalc trial_start_samples + slice_start_samples
        # X, _ = cls.prepare_data_labels(X, trials_classes)
        max_sample = X.shape[-1] - 1
        slices = CONFIG.EEG.TRIALS_SLICES
        slice_start_samples = calc_slice_start_samples(trials_start_times, trials_samples_length, slices)
        trials_start_times = trials_start_times.astype(dtype=np.int)
        return X, max_sample, slices, trials_classes, trials_start_times, trials_start_samples, slice_start_samples

    @staticmethod
    def print_stats(labels: np.ndarray):
        """
        Method: print_stats()
          Analysis of pl_labels() and extraction of how many trials of each class we have
          on a per subject basis. Result is printed on the screen.
        """
        logging.info("- Some statistics of BCIC_IV2a dataset:")

        all_subjects_counts = [0, 0, 0, 0, 0, 0]

        logging.info()
        logging.info("  Subject | class1 | class2 | class3 | class4 | artifact | all-legal")
        logging.info("  --------|--------|--------|--------|--------|----------|----------")
        for subject in range(labels.shape[0]):
            class_counts = [0, 0, 0, 0, 0, 0]
            for trial in range(labels.shape[-1]):
                if labels[subject, trial] == 0:
                    class_counts[0] = class_counts[0] + 1
                elif labels[subject, trial] == 1:
                    class_counts[1] = class_counts[1] + 1
                elif labels[subject, trial] == 2:
                    class_counts[2] = class_counts[2] + 1
                elif labels[subject, trial] == 3:
                    class_counts[3] = class_counts[3] + 1
                elif labels[subject, trial] == -1:
                    class_counts[4] = class_counts[4] + 1
                else:
                    logging.info("print_stats(): Illegal class!!! %s", labels[subject, trial])

            class_counts[5] = class_counts[0] + class_counts[1] + class_counts[2] \
                              + class_counts[3]

            for i in range(len(all_subjects_counts)):
                all_subjects_counts[i] = all_subjects_counts[i] + class_counts[i]

            logging.info("    %3d   |   %3d  |   %3d  |   %3d  |   %3d  |    %3d   |    %3d" % \
                         (subject, class_counts[0], class_counts[1], class_counts[2], \
                          class_counts[3], class_counts[4], class_counts[5]))

        logging.info("  --------|--------|--------|--------|--------|----------|----------")
        logging.info("    All   |   %3d  |   %3d  |   %3d  |   %3d  |    %3d   |   %4d" % \
                     (all_subjects_counts[0], all_subjects_counts[1], all_subjects_counts[2], \
                      all_subjects_counts[3], all_subjects_counts[4], all_subjects_counts[5]))
        logging.info()


########################################################################################
if __name__ == '__main__':
    CONFIG.set_eeg_config(BCIC.CONFIG)
    subjects = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    #    subjects = [1]
    n_class = 4
    training = 1

    logging.info(' Generate pl_data and pl_labels and store them in files')
    preloaded_data, preloaded_labels = BCICDataLoader.load_subjects_data(subjects, n_class)
    BCICDataLoader.print_stats(preloaded_labels)
    tmp_folder = f'{Path.home()}/TEMP/'
    BCIC_IV2a_dataset.save_pl_dataLabels(subjects, n_class, preloaded_data, preloaded_labels, fname="test1.npz",
                                         path=tmp_folder)

    logging.info('Load pl_data and pl_labels from file and calculate the psds')
    subjects, n_class, pl_data, pl_labels, n_trials_max = BCIC_IV2a_dataset.load_pl_dataLabels(fname="test1.npz",
                                                                                               path=tmp_folder)
    BCICDataLoader.print_stats(pl_labels)
    BCIC_IV2a_dataset.calc_psds(n_class, subjects, pl_data, pl_labels, path=tmp_folder)

    logging.info(' Plot psds ')
    plot_psds(path=tmp_folder)
    logging.info("The End")
