"""
Helper functions for printing Statistics
Saving results, etc.
"""

import errno
import os

import numpy as np
import torch  # noqa
import torch.nn.functional as F  # noqa
import torch.optim as optim  # noqa
from torch import nn, Tensor  # noqa
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler, Subset  # noqa
from torch.utils.data.dataset import ConcatDataset as _ConcatDataset  # noqa

from config import TEST_OVERFITTING, training_results_folder, benchmark_results_folder, \
    trained_model_name, chs_names_txt, results_folder, global_config, live_sim_results_folder, \
    training_ss_results_folder, eeg_config, set_eeg_times, set_eeg_trials_slices
from util.misc import datetime_to_folder_str, get_str_n_classes


def create_results_folders(path=None, name=None, datetime=None, type='train'):
    if path is not None:
        if type == 'train':
            folder = f"{results_folder}/{path}{training_results_folder}"
        elif type == 'benchmark':
            folder = f"{path}{benchmark_results_folder}{'' if name is None else f'/{name}'}"
        elif type == 'live_sim':
            folder = f"{path}{live_sim_results_folder}{'' if name is None else f'/{name}'}"
        elif type == 'train_ss':
            folder = f"{path}{training_ss_results_folder}{'' if name is None else f'/{name}'}"

    else:
        now_string = datetime_to_folder_str(datetime)
        folder = f"{results_folder}/{now_string}{training_results_folder}"
    makedir(folder)
    return folder


# Saves config + results.txt in dir_results
def save_benchmark_results(str_conf, n_class, res_str, model, dir_results,
                           tag=None):
    file_result = open(f"{dir_results}/{n_class}class-benchmark{'' if tag is None else f'_{tag}'}.txt", "w+")
    file_result.write(str_conf)
    file_result.write(res_str)
    file_result.close()
    # Save trained EEGNet to results folder

    torch.save(model.state_dict(), f"{dir_results}/{n_class}class_{trained_model_name}")


# Saves config + results.txt in dir_results
def save_training_results(n_class, str_res,
                          dir_results, tag=None):
    file_result = open(f"{dir_results}/{n_class}class-training{'' if tag is None else f'_{tag}'}.txt", "w+")
    file_result.write(str_res)
    file_result.close()


def save_config(str_conf, ch_names, dir_results, tag=None):
    file_result = open(f"{dir_results}/config{'' if tag is None else f'_{tag}'}.txt", "w+")
    file_result.write(str_conf)
    file_result.close()
    np.savetxt(f"{dir_results}/{chs_names_txt}", ch_names, delimiter=" ", fmt="%s")


def save_training_numpy_data(test_accs, class_accuracies, train_losses, test_losses, save_path, n_class,
                             excluded_subjects):
    np.savez(f"{save_path}/{n_class}class-training.npz", test_accs=test_accs, train_losses=train_losses,
             class_accs=class_accuracies, test_losses=test_losses,
             tmin=eeg_config.EEG_TMIN, tmax=eeg_config.EEG_TMAX, slices=eeg_config.TRIALS_SLICES,
             excluded_subjects=np.asarray(excluded_subjects, dtype=np.int))


def training_result_str(accuracies, accuracies_overfitting, class_trials, class_accuracies, elapsed, best_valid_epochs,
                        best_valid_losses, best_fold, early_stop=True):
    folds_str = ""
    for fold in range(len(accuracies)):
        folds_str += f'\tFold {fold + 1} {"[Best]" if fold == best_fold else ""}:\t{accuracies[fold]:.2f}\n'
        if TEST_OVERFITTING:
            folds_str += f"\t\tOverfitting (Test-Training): {accuracies[fold] - accuracies_overfitting[fold]:.2f}\n"

    trials_str = ""
    for cl, trs in enumerate(class_trials):
        trials_str += f"\t[{cl}]: {int(trs)}"
    classes_str = ""
    for l in range(len(class_accuracies)):
        classes_str += f'\t[{l}]: {class_accuracies[l]:.2f}'
    best_epochs_str = ""
    if early_stop:
        best_epochs_str += "Best Validation Loss Epochs of Folds:\n"
        for fold in range(best_valid_epochs.shape[0]):
            best_epochs_str += f'Fold {fold + 1}{" [Best]" if fold == best_fold else ""}: {best_valid_epochs[fold]} (loss: {best_valid_losses[fold]:.5f})\n'

    return f"""#### Results ####
Elapsed Time: {elapsed}
Accuracies of Folds:
{folds_str}
Avg. acc: {np.average(accuracies):.2f}
{f'Avg. Overfitting difference: {np.average(accuracies) - np.average(accuracies_overfitting):.2f}' if TEST_OVERFITTING else ''}
Trials per class:
{trials_str}
Avg. Class Accuracies:
{classes_str}
{best_epochs_str}
###############\n\n"""


def training_ss_result_str(acc, class_trials, class_accs, elapsed):
    trials_str = ""
    for cl, trs in enumerate(class_trials):
        trials_str += f"\t[{cl}]: {int(trs)}"
    classes_str = ""
    for l in range(len(class_accs)):
        classes_str += f'\t[{l}]: {class_accs[l]:.2f}'

    return f"""#### Results ####
Elapsed Time: {elapsed}
Accuracy on Test Subset:
{acc:.2f}
Trials per class:
{trials_str}
Avg. Class Accuracies:
{classes_str}
###############\n\n"""


def benchmark_result_str(config, n_class, batch_lat_avg, trial_inf_time_avg, acc_avg, elapsed):
    return f"""#### Results ####
Elapsed Time: {str(elapsed)}
Avg. Batch Latency:{batch_lat_avg}
Inference time per Trial:{trial_inf_time_avg}
Trials per second:{(1 / trial_inf_time_avg):.2f}
Accuracies:{acc_avg:.2f}
###############\n\n"""


def benchmark_single_result_str(trials, acc, num_batches, batch_lat, trial_inf_time):
    return f"""Accuracy on the {trials} trials: {acc:.2f}
    Batches:{num_batches} Trials:{trials}
    Batch Latency: {batch_lat:.5f}
    Trial Inf. Time: {trial_inf_time}
    Trials per second: {(1 / trial_inf_time): .2f}"""


def training_config_str(config):
    return f"""#### Config ####
{get_default_config_str(config)}
Dataset split in {config.folds} Subject Groups, {config.folds - 1} for Training, {1} for Testing (Cross Validation)
{f'Excluded Subjects:{config["excluded"]}' if len(config["excluded"]) > 0 else ""}
{get_global_config_str()}
Early Stopping: {config.early_stop}
Epochs: {config.num_epochs}
Learning Rate: initial = {config.lr.start}, Epoch milestones = {config.lr.milestones}, gamma = {config.lr.gamma}
###############\n\n"""


def training_ss_config_str(config):
    return f"""#### Config ####
{get_default_config_str(config)}
Subject: {config.subject}
Runs for Testing: {config.n_test_runs}
{get_global_config_str()}
Epochs: {config.num_epochs}
Learning Rate: initial = {config.lr.start}, Epoch milestones = {config.lr.milestones}, gamma = {config.lr.gamma}
###############\n\n"""


def benchmark_config_str(config):
    return f"""#### Config ####
{get_default_config_str(config)}
TensorRT optimized: {config.trt} (fp{16 if bool(config.fp16) else 32})
{get_global_config_str()}
Preload subjects Chunksize: {config.subjects_cs}
Dataset Iterations: {config.iters}
###############\n\n"""


def live_sim_config_str(config, n_class=None):
    return f"""#### Config ####
{get_default_config_str(config)}
{get_global_config_str()}
Subject: {config.subject}
Run: {config.run}
###############\n\n"""


def makedir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise e
        pass


def get_global_config_str():
    return f"""EEG Epoch interval: [{eeg_config.EEG_TMIN};{eeg_config.EEG_TMAX}]s
Bandpass Filter: [{global_config.FREQ_FILTER_HIGHPASS};{global_config.FREQ_FILTER_LOWPASS}]
Notch Filter (60Hz): {global_config.USE_NOTCH_FILTER}"""


def get_default_config_str(config):
    return f"""Device: {config.device}
Nr. of classes: {config.n_classes}
{get_str_n_classes(config.n_classes)}
Channels: {len(config.ch_names)} {config.ch_names}
Batch Size: {config.batch_size}"""


def get_results_file(model, n_class):
    return os.path.join(model, f"{n_class}class-training.npz")


def get_trained_model_file(model, n_class):
    return os.path.join(model, f"{n_class}class_{trained_model_name}")


def load_npz(npz):
    try:
        return np.load(npz)
    except FileNotFoundError:
        raise FileNotFoundError(f'File {npz} does not exist!')


# Load EEG_TMIN, EEG_TMAX, TRIALS_SLICES from .npz result file
def load_global_conf_from_results(results):
    if ('tmin' in results) & ('tmax' in results):
        set_eeg_times(results['tmin'], results['tmax'])
    else:
        raise ValueError(f'There is no "tmin" or "tmax" in {results}')
    if 'slices' in results:
        set_eeg_trials_slices(results['slices'])
    else:
        raise ValueError(f'There is no "slices" in {results}')


# Load excluded from results .npz
# If subject is present in excluded return subject
# else return first subject in excluded
# if excluded is empty, return Subject 1
def get_excluded_if_present(results, subject):
    if subject is not None:
        return subject
    excluded_subjects = results['excluded_subjects']
    if subject is None:
        if excluded_subjects.shape[0] > 0:
            return excluded_subjects[0]
        else:
            raise ValueError(
                f'Training had no excluded Subject, please specify subject to live simulate on with --subject')
    elif subject in excluded_subjects:
        return subject
    else:
        raise ValueError(f'Subject {subject} is not in excluded Subjects of model: {results}')
