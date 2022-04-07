"""
2021-03-31: Getting started - ms
"""
import logging
from datetime import datetime

import mne
import numpy as np
import torch  # noqa
from sklearn.model_selection import GroupKFold


import matplotlib.pyplot as plt
from tqdm import tqdm
import sys

from app.config import CONFIG
from app.data.datasets.phys.phys_data_loading import PHYSDataLoader
from app.data.datasets.phys.phys_dataset import PHYS
from app.util.dot_dict import DotDict
from app.util.misc import groups_labels


def psd_calc(in_data):
    fourier_transform = np.fft.rfft(in_data)
    abs_fourier_transform = np.abs(fourier_transform)
    psd = np.square(abs_fourier_transform)
    #    psd = 10 * np.log10(psd)
    #    frequency = np.linspace(0, fs / 2, len(psd))
    #    plt.plot(frequency, psd)
    #    plt.title("Power spectral density (log10)")
    #    plt.show()
    return psd


# Calculate mean psd over all subjects, all trials/subject and channel/trial
# based on the data given by the 'Dataloader's'
def analyze_data(num_epochs=CONFIG.MI.EPOCHS, batch_size=CONFIG.MI.BATCH_SIZE, folds=CONFIG.MI.SPLITS, lr=CONFIG.MI.LR,
                 n_classes=CONFIG.MI.N_CLASSES,
                 save_model=True, device=torch.device("cpu"), name=None, tag=None, ch_names=PHYS.CHANNELS,
                 equal_trials=True, early_stop=False, excluded=[]):
    config = DotDict(num_epochs=num_epochs, batch_size=batch_size, folds=folds, lr=lr, device=device,
                     n_classes=n_classes, ch_names=ch_names, early_stop=early_stop, excluded=excluded)

    # Dont print MNE loading logs
    mne.set_log_level('WARNING')
    CONFIG.set_eeg_config(PHYS.CONFIG)
    CONFIG.EEG.set_trials_slices(1)

    start = datetime.now()
    logging.info("- Data analysis started")

    available_subjects = [i for i in PHYS.ALL_SUBJECTS if i not in excluded]
    logging.info("  - Available subjects: %s", len(available_subjects))

    n_class = 2
    preloaded_data, preloaded_labels = None, None

    logging.info("PRELOADING ALL DATA IN MEMORY")
    preloaded_data, preloaded_labels = PHYSDataLoader.load_subjects_data(available_subjects, n_class,
                                                                         ch_names, equal_trials, normalize=False)

    used_subjects = available_subjects
    validation_subjects = []
    # Group labels (subjects in same group need same group label)
    folds = 2
    groups = groups_labels(len(used_subjects), folds)

    # Split Data into training + util
    cv = GroupKFold(n_splits=folds)

    cv_split = cv.split(X=used_subjects, groups=groups)

    batch_size = 1
    # Next Splits Combination of Train/Test Datasets + Validation Set Loader
    loaders = PHYSDataLoader.create_loaders_from_splits(next(cv_split), validation_subjects, n_class,
                                                        preloaded_data, preloaded_labels, batch_size,
                                                        ch_names, equal_trials,
                                                        used_subjects=used_subjects)
    loader_train, loader_test, loader_valid = loaders

    num_samples = CONFIG.EEG.SAMPLES
    psd_all = np.zeros(int(num_samples / 2) + 1)
    psd_class0 = np.zeros(int(num_samples / 2) + 1)  # Mean psd of Left Hand trials
    psd_class1 = np.zeros(int(num_samples / 2) + 1)  # Mean psd of Right Hand trials
    num_class0_trials = 0
    num_class1_trials = 0

    pbar = tqdm(loader_train, file=sys.stdout)
    # Training in batches from the DataLoader
    for idx_batch, (inputs, labels) in enumerate(pbar):
        # Convert inputs, labels tensors to np array
        #        inputs = inputs.cpu()
        #        labels = labels.cpu()
        inputs = inputs.numpy()
        labels = labels.numpy()
        #        logging.info("inputs.shape = %s", inputs.shape)
        #        logging.info("labels.shape = %s", labels.shape)
        num_channels = inputs.shape[2]
        for ch in range(num_channels):
            data = inputs[0, 0, ch, :]
            psd = psd_calc(data)
            psd_all = psd_all + psd
            if labels[0] == 0:
                num_class0_trials = num_class0_trials + 1
                psd_class0 = psd_class0 + psd
            elif labels[0] == 1:
                num_class1_trials = num_class1_trials + 1
                psd_class1 = psd_class1 + psd
            else:
                logging.info("ERROR: Illegal label")

    pbar = tqdm(loader_test, file=sys.stdout)
    # Training in batches from the DataLoader
    for idx_batch, (inputs, labels) in enumerate(pbar):
        # Convert inputs, labels tensors to np array
        inputs = inputs.cpu()
        labels = labels.cpu()
        inputs = inputs.numpy()
        labels = labels.numpy()
        #        logging.info("inputs.shape = %s", inputs.shape)
        #        logging.info("labels.shape = %s", labels.shape)
        num_channels = inputs.shape[2]
        for ch in range(num_channels):
            data = inputs[0, 0, ch, :]
            psd = psd_calc(data)
            psd_all = psd_all + psd
            if labels[0] == 0:
                num_class0_trials = num_class0_trials + 1
                psd_class0 = psd_class0 + psd
            elif labels[0] == 1:
                num_class1_trials = num_class1_trials + 1
                psd_class1 = psd_class1 + psd
            else:
                logging.info("ERROR: Illegal label")

    logging.info("num_class0_trials = %s", num_class0_trials)
    logging.info("num_class1_trials = %s", num_class1_trials)
    psd_all = psd_all / (num_class0_trials + num_class1_trials)
    psd_class0 = psd_class0 / (num_class0_trials)
    psd_class1 = psd_class1 / (num_class1_trials)

    sampling_rate = 160

    # Plot psd's
    psd_all = 10 * np.log10(psd_all)
    psd_class0 = 10 * np.log10(psd_class0)
    psd_class1 = 10 * np.log10(psd_class1)

    frequency = np.linspace(0, sampling_rate / 2, len(psd))
    plt.plot(frequency, psd_all)
    plt.plot(frequency, psd_class0)
    offset = 0.0
    plt.plot(frequency, psd_class1 + offset)

    legend = []
    legend.append('All trials mean')
    legend.append('Class0 trials mean')
    legend.append('Class1 trials mean')
    plt.legend(legend)

    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Power spectral density dB/Hz')
    plt.title("Power spectral density (log10)")
    plt.grid(True)
    plt.show()

    return


# Calculate mean psd over all subjects, all trials/subject and channel/trial
# based on the 'preloaded' data
def analyze_data1(num_epochs=CONFIG.MI.EPOCHS, batch_size=CONFIG.MI.BATCH_SIZE, folds=CONFIG.MI.SPLITS,
                  lr=CONFIG.MI.LR, n_classes=CONFIG.MI.N_CLASSES, save_model=True, device=torch.device("cpu"),
                  name=None, tag=None, ch_names=PHYS.CHANNELS, equal_trials=True, early_stop=False, excluded=[]):
    config = DotDict(num_epochs=num_epochs, batch_size=batch_size, folds=folds, lr=lr, device=device,
                     n_classes=n_classes, ch_names=ch_names, early_stop=early_stop, excluded=excluded)

    # Dont print MNE loading logs
    mne.set_log_level('WARNING')
    CONFIG.EEG.TRIALS_SLICES = 1

    start = datetime.now()
    logging.info("- Data analysis started")

    available_subjects = [i for i in PHYS.ALL_SUBJECTS if i not in excluded]
    logging.info("  - Available subjects: %s", len(available_subjects))

    n_class = 2
    logging.info("PRELOADING ALL DATA IN MEMORY")
    preloaded_data, preloaded_labels = PHYSDataLoader.load_subjects_data(available_subjects, n_class,
                                                                         ch_names, equal_trials, normalize=False)

    logging.info("  - preloaded_data.shape = %s", preloaded_data.shape)
    logging.info("  - preloaded_labels.shape = %s", preloaded_labels.shape)

    # # plot channel x of trial y of subject z
    # subject = 0     # selected subject
    # trial = 0       # selected trial
    # channel = 1     # selected channel
    # plt.plot(preloaded_data[subject, trial, channel, :])
    # plt.title("subject=%d, trial=%d, ch=%d, trial EEG data" % (subject, trial, channel))
    # plt.show()

    # Assign basic parameters
    sampling_rate = 160.0
    num_samples = preloaded_data.shape[3]
    num_channels = preloaded_data.shape[2]
    num_trials = preloaded_data.shape[1]
    num_subjects = preloaded_data.shape[0]
    logging.info("  - subjects: %d, trials/subject: %d, EEG-channels/trial: %d" ,num_subjects, num_trials, num_channels)

    # Calculate and sum mean power spectral density psd
    psd_all = np.zeros(int(num_samples / 2) + 1)
    psd_class0 = np.zeros(int(num_samples / 2) + 1)  # Mean psd of Left Hand trials
    psd_class1 = np.zeros(int(num_samples / 2) + 1)  # Mean psd of Right Hand trials
    num_class0_trials = 0
    num_class1_trials = 1
    for subject in range(num_subjects):
        # sum up the psd for all trials of a subjects
        for trial in range(num_trials):
            # calculate the psd for each EEG channel and sum up all psd
            for ch in range(num_channels):
                data = preloaded_data[subject, trial, ch, :]
                psd = psd_calc(data)
                psd_all = psd_all + psd
                if preloaded_labels[subject, trial] == 0:
                    num_class0_trials = num_class0_trials + 1
                    psd_class0 = psd_class0 + psd
                elif preloaded_labels[subject, trial] == 1:
                    num_class1_trials = num_class1_trials + 1
                    psd_class1 = psd_class1 + psd
                else:
                    logging.info("ERROR: Illegal label")

    psd_all = psd_all / (num_channels * num_trials * num_subjects)
    psd_class0 = psd_class0 / (num_class0_trials)
    psd_class1 = psd_class1 / (num_class1_trials)

    # Plot psd's
    psd_all = 10 * np.log10(psd_all)
    psd_class0 = 10 * np.log10(psd_class0)
    psd_class1 = 10 * np.log10(psd_class1)

    frequency = np.linspace(0, sampling_rate / 2, len(psd))
    plt.plot(frequency, psd_all)
    plt.plot(frequency, psd_class0)
    plt.plot(frequency, psd_class1)

    legend = []
    legend.append('All trials mean')
    legend.append('Class0 trials mean')
    legend.append('Class1 trials mean')
    plt.legend(legend)

    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Power spectral density dB/Hz')
    plt.title("Power spectral density (log10)")
    plt.grid(True)
    plt.show()

    return


########################################################################################
if __name__ == '__main__':
    analyze_data()
