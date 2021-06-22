import numpy as np
from sklearn.metrics import confusion_matrix
import torch  # noqa
from config import eeg_config, TEST_OVERFITTING
from machine_learning.configs_results import get_trained_model_file
from machine_learning.models.eegnet import EEGNet
from datetime import datetime
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
        # All Epoch Losses on Train/test for every Fold an epoch
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
        self.best_fold_act_labels = act_labels
        self.best_fold_pred_labels = pred_labels
