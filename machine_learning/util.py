from datetime import datetime, timedelta
from typing import Any, Dict, List

import mne
import numpy as np
import torch  # noqa
from sklearn.metrics import confusion_matrix
from torch.utils.data import SequentialSampler

from config import eeg_config, TEST_OVERFITTING
from data.datasets import TrialsDataset
from machine_learning.configs_results import get_trained_model_file
from machine_learning.models.eegnet import EEGNet
from util.misc import get_class_avgs


# Torch to TensorRT for model optimizations
# https://github.com/NVIDIA-AI-IOT/torch2trt
# Comment out if TensorRt is not installed
# if torch.cuda.is_available():
#     import ctypes
# #    from torch2trt import torch2trt
#
#     _cudart = ctypes.CDLL('libcudart.so')


# Source https://stackoverflow.com/questions/39770376/scikit-learn-get-accuracy-scores-for-each-class
def get_class_accuracies(act_labels, pred_labels):
    conf_mat = confusion_matrix(act_labels, pred_labels)
    cm = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
    return 100 * cm.diagonal()


def get_trials_per_class(n_class, act_labels):
    class_trials = np.zeros(n_class)
    for i in range(n_class):
        class_trials[i] = np.count_nonzero(act_labels == i, axis=0)
    return class_trials


def get_confusion_matrix(act_labels, pred_labels):
    return confusion_matrix(act_labels, pred_labels)


# Returns EEGNet model optimized with TensorRT (fp16/32)
def get_tensorrt_model(model, batch_size, chs, device, fp16):
    t = torch.randn((batch_size, 1, chs, eeg_config.SAMPLES)).to(device)
    # add_constant() TypeError: https://github.com/NVIDIA-AI-IOT/torch2trt/issues/440
    # TensorRT either with fp16 ("half") or fp32
    if fp16:
        t = t.half()
        model = model.half()
    model = torch2trt(model, [t], max_batch_size=batch_size, fp16_mode=fp16)
    print(f"Optimized EEGNet model with TensorRT (fp{'16' if fp16 else '32'})")
    return model


# Run some example Inferences to warm up the GPU
def gpu_warmup(device, warm_ups, model, batch_size, chs, fp16):
    print("Warming up GPU")
    for u in range(warm_ups):
        with torch.no_grad():
            data = torch.randn((batch_size, 1, chs, eeg_config.SAMPLES)).to(device)
            y = model(data.half() if fp16 else data)


# Return EEGNet model
# pretrained state will be loaded if present
def get_model(n_class, chs, device, model_path=None):
    model = EEGNet(N=n_class, T=eeg_config.SAMPLES, C=chs)
    # model = DoseNet(C=chs, n_class=n_class, T=eeg_config.SAMPLES)
    if model_path is not None:
        model.load_state_dict(torch.load(get_trained_model_file(model_path, n_class)))
    model.to(device)
    return model


class ML_Run_Data:
    """
    Data Class for a single n_class Training Run
    storing Accuracies, Losses, Best Fold, Elapsed Time,...
    """

    n_class: int
    folds: int
    fold_accuracies: np.ndarray
    accuracies_overfitting: np.ndarray
    best_losses_test: np.ndarray
    best_epochs_test: np.ndarray
    best_fold_act_labels: np.ndarray
    best_fold_pred_labels: np.ndarray
    best_fold: int
    class_accuracies: np.ndarray
    class_trials: np.ndarray
    avg_class_accs: np.ndarray
    epoch_losses_train: np.ndarray
    epoch_losses_test: np.ndarray
    cv_split: Any
    start: datetime
    end: datetime
    elapsed: timedelta
    best_model: List[Dict]

    def __init__(self, folds, n_class, num_epochs, cv_split):
        self.n_class = n_class
        self.folds = folds
        # Avg. Accuracy of each fold
        self.fold_accuracies = np.zeros(folds)
        self.accuracies_overfitting = np.zeros((folds)) if TEST_OVERFITTING else None
        # Best Test Loss of every fold
        self.best_losses_test = np.full((folds), fill_value=np.inf)
        # Epoch with best Loss on Test Set for every fold
        self.best_epochs_test = np.zeros((folds), dtype=np.int)
        # Actual and Predicted labels for the best Fold
        self.best_fold_act_labels = None
        self.best_fold_pred_labels = None
        self.best_fold = -1
        self.class_accuracies, self.class_trials = np.zeros((folds, n_class)), np.zeros(n_class)
        self.avg_class_accs = np.zeros(self.n_class)
        # All Epoch Losses on Train/util for every Fold an epoch
        self.epoch_losses_train = np.zeros((folds, num_epochs))
        self.epoch_losses_test = np.zeros((folds, num_epochs))
        # Subject Splits
        self.cv_split = cv_split
        self.start = None
        self.end = None
        self.elapsed = None
        # Best Model state dict for every fold
        self.best_model = [{} for i in range(folds)]

    def start_run(self):
        self.start = datetime.now()

    def end_run(self):
        self.end = datetime.now()
        self.elapsed = self.end - self.start
        self.avg_class_accs = get_class_avgs(self.n_class, self.class_accuracies)

    def set_train_results(self, fold, fold_results):
        """
        Set Fold's Train results
        :param fold_results: (loss_values_train, loss_values_valid, best_model, best_epoch) see do_train()
        """
        self.epoch_losses_train[fold], self.epoch_losses_test[fold], self.best_model[fold], \
        self.best_epochs_test[fold] = fold_results

    def set_test_results(self, fold, test_accuracy, act_labels, pred_labels):
        """
        Set Fold's Test Results with Accuracy and Labels
        """
        self.fold_accuracies[fold] = test_accuracy
        self.class_trials = get_trials_per_class(self.n_class, act_labels)
        self.class_accuracies[fold] = get_class_accuracies(act_labels, pred_labels)

    def best_epoch_loss_test(self, fold):
        """
        Returns Epoch with lowest Loss on Test Set of fold
        """
        return self.epoch_losses_test[fold][self.best_epochs_test[fold]]

    def set_best_fold(self, fold, act_labels=None, pred_labels=None):
        self.best_fold = fold
        if act_labels is not None:
            self.best_fold_act_labels = act_labels
        if pred_labels is not None:
            self.best_fold_pred_labels = pred_labels


def preferred_device(preferred):
    dev = None
    if (preferred == "gpu") & torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    return torch.device(dev)


class SubjectTrialsRandomSampler(SequentialSampler):
    """
    Samples subject-wise Trials in Random Order
    Shuffles Trials of every Subject individually so that all Trials
    of a Subject only have to be loaded once per Epoch
    """
    subjects: int
    trials: np.ndarray

    def __init__(self, ds: TrialsDataset):
        """
        Initializes the Sampler for the given TrialsDataset
        :param ds: TrialsDataset containing List of Subjects and Trial Per Subject
        """
        self.subjects = len(ds.subjects)
        self.trials = np.arange(self.subjects * ds.trials_per_subject)
        super().__init__(self.trials)

    def __iter__(self):
        """
        Splits all Trials into Trials of each Subject
        Shuffles the Subjects' Trial Lists individually
        :return: yields List of Subjects' Trials
        """
        trials = np.split(self.trials, self.subjects)
        np.random.seed(42)
        for subject_trials in trials:
            np.random.shuffle(subject_trials)
            for trial in subject_trials:
                yield trial


def get_valid_trials_per_subject(labels: np.ndarray, subjects: List[int], used_subjects: List[int],
                                 n_trials_max: int) -> List[int]:
    """
    Get number of valid trials per subject (skip Trials with label=-1)
    :param n_trials_max: Amount of Maximum trials per subject
    :param labels: ndarray shape(subjects,Trial Labels)
    :param subjects: Subjects to calculate valid Trials for
    :param used_subjects: All subjects in preloaded_labels
    :return List of valid Trials per subject
    """
    trials_per_subject = [0] * len(subjects)
    for s_idx, subject in enumerate(subjects):
        global_subject_idx = used_subjects.index(subject)
        for trial in range(n_trials_max):
            if labels[global_subject_idx, trial] != -1:
                trials_per_subject[s_idx] = trials_per_subject[s_idx] + 1
    return trials_per_subject


def resample_eeg_data(data: np.ndarray, or_samplerate: float, dest_samplerate: float, per_subject=False):
    """
    Resamples EEG Data from original Samplerate to destination Samplerate
    :param data: EEG Data as numpy.ndarray, shape: (Subjects (optional), Trials, Channels, Samples)
    :param per_subject: Resample per Subject instead of resampling the whole data array at once
    (data.dtype has to be 'np.float' with shape (Subjects, Trials, Channels, Samples)), used for
    large datasets because resampling uses a lot of memory
    :return: data resampled to dest_samplerate
    """
    print(f"Resampling EEG Data from {or_samplerate}Hz to {dest_samplerate}Hz Samplerate")
    # E.g. Original LSMR21 Dataset has Trials of different Sample size so the dtype of the ndarray is 'object'
    # -> have to resample each Trial Data array individually
    if (data.dtype == object) or per_subject:
        # Need to intialize new ndarray with new Samplesize (last dim of data)
        if per_subject:
            new_shape = (data.shape[0], data.shape[1], data.shape[2],
                         int(data.shape[3] * (dest_samplerate / or_samplerate)))
            res_data = np.zeros(new_shape, dtype=np.float32)
        else:
            res_data = data
        for trial_idx in range(data.shape[0]):
            res_data[trial_idx] = mne.filter.resample(data[trial_idx].astype(np.float64), window='boxcar',
                                                      up=dest_samplerate, down=or_samplerate)
            # TODO mne.filter.resample needs np.float64 -> reconvert to float32 or keep float64(means 2x file size)
            res_data[trial_idx] = res_data[trial_idx].astype(np.float32)
        return res_data
    else:
        # If all Trials have same Sample Size only 1 resample call is needed
        data = mne.filter.resample(data.astype(np.float64), window='boxcar',
                                   up=dest_samplerate, down=or_samplerate)
        data = data.astype(np.float32)
        return data
