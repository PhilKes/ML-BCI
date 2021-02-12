"""
Utility functions for printing Statistics, Plotting,
Saving results, etc.
"""
import math
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import torch  # noqa
import torch.nn.functional as F  # noqa
import torch.optim as optim  # noqa
from torch import nn, Tensor  # noqa
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler, Subset  # noqa
from torch.utils.data.dataset import ConcatDataset as _ConcatDataset  # noqa

from config import TEST_OVERFITTING, training_results_folder, benchmark_results_folder


# Create results folder with current DateTime-PLATFORM as name
def print_subjects_ranges(train, test):
    if (train[0] < test[0]) & (train[-1] < test[0]):
        print(f"Subjects for Training:\t[{train[0]}-{train[-1]}]")
    elif (train[0] < test[0]) & (train[-1] > test[0]):
        print(f"Subjects for Training:\t[{train[0]}-{test[0] - 1}],[{test[-1] + 1}-{train[-1]}]")
    elif (train[0] > test[0]):
        print(f"Subjects for Training:\t[{train[0]}-{train[-1]}]")
    print(f"Subjects for Testing:\t[{test[0]}-{test[-1]}]")
    return


# Plots data with Matplot
# data: either 1d or 2d datasets
# labels: if 2d data, provide labels for legend
# save_path: if plot + data array should be saved, declare save location
# bar_plot: Plot as bars with average line (for Accuracies)
def matplot(data, title='', xlabel='', ylabel='', labels=[], max_y=None, save_path=None, bar_plot=False):
    fig, ax = plt.subplots()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
    plt.ylabel(ylabel)
    if max_y is not None:
        plt.ylim(top=max_y)
    # Avoid X-Labels overlapping
    if data.shape[-1] > 30:
        multiple = 5 if data.shape[-1] % 5 == 0 else 4
        plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(multiple))
        plt.xticks(rotation=90)
    # Plot multiple lines
    if data.ndim == 2:
        for i in range(len(data)):
            plt.plot(data[i], label=labels[i] if len(labels) >= i else "")
            plt.legend()
        plt.grid()
    else:
        if bar_plot:
            ax.bar(np.arange(len(data)), data, 0.35, )
            ax.axhline(np.average(data), color='red', linestyle='--')
        else:
            plt.plot(data, label=labels[0] if len(labels) > 0 else "")
            plt.grid()
    if save_path is not None:
        fig = plt.gcf()
        fig.savefig(f"{save_path}/{title}.png")
        np.save(f"{save_path}/{title}.npy", data)
    # fig.tight_layout()
    plt.show()


# Create Plot from numpy file
# if save = True save plot as .png
def plot_numpy(np_file_path, xlabel, ylabel, save):
    data = np.load(np_file_path)
    labels = []
    if data.ndim > 1:
        labels = [f"Run {i}" for i in range(data.shape[0])]
    filename = os.path.splitext(os.path.basename(np_file_path))[0]
    save_path = os.path.dirname(np_file_path) if save else None
    matplot(data, filename, xlabel, ylabel, labels=labels, save_path=save_path)
    return data


# Saves config + results.txt in dir_results
def save_training_results(str_conf, n_class, accuracies, epoch_losses, elapsed, dir_results,
                          accuracies_overfitting=None):
    # TODO save numpy array in one file with dict names
    str_elapsed = str(elapsed)
    file_result = open(f"{dir_results}/{n_class}class-results.txt", "w+")
    file_result.write(str_conf)
    file_result.write(f"Elapsed Time: {str_elapsed}\n")
    file_result.write(f"Accuracies of Splits:\n")
    for i in range(len(accuracies)):
        file_result.write(f"\tRun {i}: {accuracies[i]:.2f}\n")
        if TEST_OVERFITTING:
            file_result.write(f"\t\tOverfitting (Test-Training): {accuracies[i] - accuracies_overfitting[i]:.2f}\n")
    file_result.write(f"Avg. acc: {np.average(accuracies):.2f}\n")
    if TEST_OVERFITTING:
        file_result.write(
            f"Avg. Overfitting difference: {np.average(accuracies) - np.average(accuracies_overfitting):.2f}")
    file_result.close()


# Saves config + results.txt in dir_results
def save_benchmark_results(str_conf, n_class, batch_lat, trial_inf_time, elapsed, model, dir_results):
    file_result = open(f"{dir_results}/{n_class}class-results.txt", "w+")
    file_result.write(str_conf)
    file_result.write(f"Elapsed Time: {str(elapsed)}\n")
    file_result.write(f"Avg. Batch Latency:{batch_lat}\n")
    file_result.write(f"Inference time per Trial:{trial_inf_time}\n")
    file_result.write(f"Trials per second:{(1 / trial_inf_time):.2f}\n")
    file_result.close()
    # Save trained EEGNet to results folder
    torch.save(model.state_dict(), f"{dir_results}/trained_model.pt")


def create_results_folders(datetime, platform="PC", type='train'):
    now_string = datetime.strftime("%Y-%m-%d %H_%M_%S")
    results = f"{training_results_folder if type == 'train' else benchmark_results_folder}/{now_string}-{platform}"
    try:
        os.makedirs(results)
    except OSError as err:
        raise err
        pass
    return results


str_n_classes = ["", "", "Left/Right Fist", "Left/Right-Fist / Rest", "Left/Right-Fist / Rest / Both-Feet"]

# Split python list into chunks with equal size (last chunk can have smaller size)
# source: https://stackoverflow.com/questions/24483182/python-split-list-into-n-chunks
def split_list_into_chunks(list, chunk_size):
    m = int(len(list) / int(math.ceil(len(list) / chunk_size))) + 1
    return [list[i:i + m] for i in range(0, len(list), m)]


def get_str_n_classes(n_classes):
    return f'Classes: {[str_n_classes[i] for i in n_classes]}'


def training_config_str(config, n_class=None):
    return f"""#### Config ####
CUDA: {config['cuda']}
Nr. of classes: {config['n_classes'] if n_class is None else n_class}
{get_str_n_classes(config['n_classes'] if n_class is None else [n_class])}
Dataset split in {config['splits']} Subject Groups, {config['splits'] - 1} for Training, {1} for Testing (Cross Validation)
Batch Size: {config['batch_size']}
Epochs: {config['num_epochs']}
Learning Rate: initial = {config['lr']['start']}, Epoch milestones = {config['lr']['milestones']}, gamma = {config['lr']['gamma']}
###############\n\n"""


def benchmark_config_str(config, n_class=None):
    return f"""#### Config ####
CUDA: {config['cuda']}
TensorRT optimized: {config['trt']} (fp{16 if config['fp16'] else 32})
Nr. of classes: {config['n_classes'] if n_class is None else n_class}
{get_str_n_classes(config['n_classes'] if n_class is None else [n_class])}
Preload subjects Chunksize: {config['subjects_cs']}
Batch Size: {config['batch_size']}
Dataset Iterations: {config['iters']}
###############\n\n"""
