"""
Utility functions for printing Statistics, Plotting,
Saving results, etc.
"""
import errno
import glob
import math
import os
import shutil

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import torch  # noqa
import torch.nn.functional as F  # noqa
import torch.optim as optim  # noqa
from torch import nn, Tensor  # noqa
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler, Subset  # noqa
from torch.utils.data.dataset import ConcatDataset as _ConcatDataset  # noqa

from config import TEST_OVERFITTING, training_results_folder, benchmark_results_folder, EEG_TMIN, EEG_TMAX, \
    trained_model_name, chs_names_txt, results_folder


# Create results folder with current DateTime as name
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
        # np.save(f"{save_path}/{title}.npy", data)
    # fig.tight_layout()
    plt.show()


colors = ['tab:orange', 'tab:blue', 'tab:green', 'tab:red', 'tab:purple']


# Plots Benchmarking (Batch Latencies) for given configurations data (config_idx,batch_size_idx)
def matplot_grouped_configs(configs_data, batch_sizes, class_idx, title="", ylabel="", save_path=None):
    x = np.arange(len(configs_data))  # the label locations
    width = (1.0 / len(batch_sizes)) - 0.1  # the width of the bars

    fig, ax = plt.subplots()
    bs_rects = []
    bs_labels = []
    for bs_idx, bs in enumerate(batch_sizes):
        bs_rects.append([])
        bs_labels.append(f"BS {bs}")
        for conf_idx in range(len(configs_data)):
            conf_data = configs_data[conf_idx]
            print(conf_data)
            bs_rects[bs_idx].append(ax.bar((conf_idx) - width / len(batch_sizes) + bs_idx * width,
                                           conf_data[bs_idx][class_idx], width, color=colors[bs_idx]))

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(f"Conf_{i}" for i in range(len(configs_data)))
    ax.legend(bs_rects, bs_labels)

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.6f}",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    for rects in bs_rects:
        for rect in rects:
            autolabel(rect)

    fig.tight_layout()

    plt.show()
    if save_path is not None:
        fig = plt.gcf()
        fig.savefig(f"{save_path}/{title}.png")


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


def save_training_numpy_data(accs, class_accuracies, losses, save_path, n_class):
    np.savez(f"{save_path}/{n_class}class-training.npz", accs=accs, losses=losses, class_accs=class_accuracies)


# Loads list of ch_names from training results folder
def load_chs_from_txt(txt_folder):
    return np.genfromtxt(f"{txt_folder}/{training_results_folder}/{chs_names_txt}", dtype='str')


# Saves config + results.txt in dir_results
def save_benchmark_results(str_conf, n_class, res_str, model, dir_results,
                           tag=None):
    file_result = open(f"{dir_results}/{n_class}class-benchmark{'' if tag is None else f'_{tag}'}.txt", "w+")
    file_result.write(str_conf)
    file_result.write(res_str)
    file_result.close()
    # Save trained EEGNet to results folder

    torch.save(model.state_dict(), f"{dir_results}/{n_class}class_{trained_model_name}")


def datetime_to_folder_str(datetime):
    return datetime.strftime("%Y-%m-%d %H_%M_%S")


def create_results_folders(path=None, name=None, datetime=None, type='train'):
    if path is not None:
        if type == 'train':
            folder = f"{results_folder}/{path}{training_results_folder}"
        else:
            folder = f"{path}{benchmark_results_folder}{'' if name is None else f'/{name}'}"
    else:
        now_string = datetime_to_folder_str(datetime)
        folder = f"{results_folder}/{now_string}{training_results_folder}"
    makedir(folder)
    return folder


# Copy all Training results.txt from ./benchmarking_model folder to new benchmark results folder
# def copy_config_txts(path):
#     full_path = f"{benchmark_results_folder}/{path}"
#     makedir(full_path)
#     for file in glob.glob(os.path.join(trained_model_path, "*")):
#         shutil.copy2(file, full_path)


str_n_classes = ["", "", "Left/Right Fist", "Left/Right-Fist / Rest", "Left/Right-Fist / Rest / Both-Feet"]


# Split python list into chunks with equal size (last chunk can have smaller size)
# source: https://stackoverflow.com/questions/24483182/python-split-list-into-n-chunks
def split_list_into_chunks(p_list, chunk_size):
    m = int(len(p_list) / int(math.ceil(len(p_list) / chunk_size))) + 1
    return [p_list[i:i + m] for i in range(0, len(p_list), m)]


def split_np_into_chunks(arr, chunk_size):
    chunks = math.ceil(arr.shape[0] / chunk_size)
    arr2 = np.zeros((chunks - 1, chunk_size, arr.shape[1]), dtype=np.float)
    splits = np.split(arr, np.arange(chunk_size, len(arr), chunk_size))
    # Last chunk is of smaller size, so it is skipped
    for i in range(chunks - 1):
        arr2[i] = splits[i]
    return arr2


def get_str_n_classes(n_classes):
    return f'Classes: {[str_n_classes[i] for i in n_classes]}'


def training_config_str(config, n_class=None):
    return f"""#### Config ####
Device: {config['device']}
Nr. of classes: {config['n_classes'] if n_class is None else n_class}
{get_str_n_classes(config['n_classes'] if n_class is None else [n_class])}
Dataset split in {config['splits']} Subject Groups, {config['splits'] - 1} for Training, {1} for Testing (Cross Validation)
Channels: {len(config['ch_names'])} {config['ch_names']}
EEG Epoch interval: [{EEG_TMIN};{EEG_TMAX}]s
Batch Size: {config['batch_size']}
Epochs: {config['num_epochs']}
Learning Rate: initial = {config['lr']['start']}, Epoch milestones = {config['lr']['milestones']}, gamma = {config['lr']['gamma']}
###############\n\n"""


def training_result_str(accuracies, accuracies_overfitting, class_trials, class_accuracies, elapsed):
    runs_str = ""
    for i in range(len(accuracies)):
        runs_str += f'\tRun {i}: {accuracies[i]:.2f}\n'
        if TEST_OVERFITTING:
            runs_str += f"\t\tOverfitting (Test-Training): {accuracies[i] - accuracies_overfitting[i]:.2f}\n"

    trials_str = ""
    for cl, trs in enumerate(class_trials):
        trials_str += f"\t[{cl}]: {int(trs)}"
    classes_str = ""
    for l in range(len(class_accuracies)):
        classes_str += f'\t[{l}]: {class_accuracies[l]:.2f}'

    return f"""#### Results ####
Elapsed Time: {elapsed}
Accuracies of Splits:
{runs_str}
Avg. acc: {np.average(accuracies):.2f}
{f'Avg. Overfitting difference: {np.average(accuracies) - np.average(accuracies_overfitting):.2f}' if TEST_OVERFITTING else ''}
Trials per class:
{trials_str}
Avg. Class Accuracies:
{classes_str}
###############\n\n"""


def benchmark_config_str(config, n_class=None):
    return f"""#### Config ####
Device: {config['device']}
TensorRT optimized: {config['trt']} (fp{16 if bool(config['fp16']) else 32})
Nr. of classes: {config['n_classes'] if n_class is None else n_class}
{get_str_n_classes(config['n_classes'] if n_class is None else [n_class])}
Channels: {len(config['ch_names'])} {config['ch_names']}
EEG Epoch interval: [{EEG_TMIN};{EEG_TMAX}]s
Preload subjects Chunksize: {config['subjects_cs']}
Batch Size: {config['batch_size']}
Dataset Iterations: {config['iters']}
###############\n\n"""


def benchmark_result_str(config, n_class, batch_lat_avg, trial_inf_time_avg, acc_avg, elapsed):
    return f"""#### Results ####
Elapsed Time: {str(elapsed)}
Avg. Batch Latency:{batch_lat_avg}
Inference time per Trial:{trial_inf_time_avg}
Trials per second:{(1 / trial_inf_time_avg):.2f}
Accuracies:{acc_avg:.2f}
###############\n\n"""


def list_to_string(list):
    return ','.join([str(i) for i in list])


def makedir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise e
        pass
