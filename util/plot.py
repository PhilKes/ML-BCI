"""
Helper functions for Plotting using matplotlib
"""
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import torch  # noqa
import torch.nn.functional as F  # noqa
import torch.optim as optim  # noqa
from matplotlib import lines
from torch import nn, Tensor  # noqa
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler, Subset  # noqa
from torch.utils.data.dataset import ConcatDataset as _ConcatDataset  # noqa

from config import PLOT_TO_PDF, eeg_config

colors = ['tab:orange', 'tab:blue', 'tab:green', 'tab:red', 'tab:purple', 'black']


# Plots data with Matplot
# data: either 1d or 2d datasets
# labels: if 2d data, provide labels for legend
# save_path: if plot + data array should be saved, declare save location
# bar_plot: Plot as bars with average line (for Accuracies)
# vspans: List of vertical Rectangles to draw
#         Item Tuple: (X1,X2,color_idx)
# vlines: List of vertical Lines to draw
#         Item Tuple: (X,color_idx)
def matplot(data, title='', xlabel='', ylabel='', labels=[], max_y=None, save_path=None, bar_plot=False,
            x_values=None, ticks=None, fig_size=None,
            vspans=[], vlines=[], vlines_label=None, legend_loc=None, show_legend=True,
            min_x=None, max_x=None, color_offset=0):
    # use LaTeX fonts in the plot
    plt.rc('font', family='serif')
    if (fig_size is not None):
        plt.rcParams.update({'font.size': 22})
    else:
        plt.rcParams.update({'font.size': 10})

    fig, ax = plt.subplots()
    if fig_size is not None:
        fig.set_size_inches(fig_size[0], fig_size[1])

    plt.title(title)
    plt.xlabel(xlabel)
    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
    plt.ylabel(ylabel)
    if max_y is not None:
        plt.ylim(top=max_y)
    if min_x is not None:
        ax.set_xlim(xmin=min_x)
    if max_x is not None:
        ax.set_xlim(xmax=max_x)
    # Avoid X-Labels overlapping
    if ticks is not None:
        if ticks.shape[0] > 30:
            plt.xticks(rotation=90)
        ax.set_xticks(ticks)
        ax.set_xticklabels(x_values)
        # plt.xticks(ticks=ticks,labels=x_values)
    elif data.shape[-1] > 30:
        multiple = 5 if data.shape[-1] % 5 == 0 else 4
        plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(multiple))
        plt.xticks(rotation=90)
    # Plot multiple lines
    if data.ndim == 2:
        for i in range(len(data)):
            plt.plot(data[i], label=labels[i] if len(labels) >= i else "", color=colors[i + color_offset])
        plt.grid()
    else:
        if bar_plot:
            ax.bar(np.arange(len(data)), data, 0.35, )
            ax.axhline(np.average(data), color='red', linestyle='--')
        else:
            plt.plot(data, label=labels[0] if len(labels) > 0 else "", color=colors[0])
            plt.grid()

    for vspan in vspans:
        plt.axvspan(vspan[0], vspan[1], color=colors[vspan[2]], alpha=0.5)
    for vline in vlines:
        plt.axvline(vline[0], color=colors[vline[1]], alpha=0.75, linestyle='--')

    handles, labels = ax.get_legend_handles_labels()
    if len(vlines) > 0 & (vlines_label is not None):
        vertical_line = lines.Line2D([], [], linestyle='--', color=colors[vlines[0][1]],
                                     markersize=10, markeredgewidth=1.5)
        handles.append(vertical_line)
        labels.append(vlines_label)
    if show_legend:
        plt.legend(handles, labels, loc='best' if legend_loc is None else legend_loc)

    if save_path is not None:
        fig = plt.gcf()
        # np.save(f"{save_path}/{title}.npy", data)
        # save as PDF
        fig.savefig(f"{save_path}/{title}.png")
        if PLOT_TO_PDF:
            fig.savefig(f"{save_path}/{title}.pdf", bbox_inches='tight')
    # fig.tight_layout()

    plt.show()


# Plots Benchmarking (Batch Latencies) for given configurations data (config_idx,batch_size_idx)
def matplot_grouped_configs(configs_data, batch_sizes, class_idx, title="", ylabel="", save_path=None):
    x = np.arange(len(configs_data))  # the label locations
    width = (1.0 / len(batch_sizes)) - 0.1  # the width of the bars

    # use LaTeX fonts in the plot
    plt.rc('font', family='serif')

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
    if save_path is not None:
        fig = plt.gcf()
        # save as PDF
        if PLOT_TO_PDF:
            fig.savefig(f"{save_path}/{title}.pdf", bbox_inches='tight')
        else:
            fig.savefig(f"{save_path}/{title}.png")
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


# Plots Losses, Accuracies of Training, Validation, Testing
def plot_training_statistics(dir_results, tag, n_class, accuracies, avg_class_accuracies, epoch_losses_train,
                             epoch_losses_valid,
                             best_fold, batch_size, folds, early_stop):
    matplot(np.append(np.roll(accuracies, 1), accuracies[-1]), f"{n_class}class Cross Validation", "Fold",
            "Accuracy in %",
            save_path=dir_results, show_legend=False,
            bar_plot=True, max_y=100.0)
    matplot(avg_class_accuracies, f"{n_class}class Accuracies{'' if tag is None else tag}", "Class",
            "Accuracy in %", show_legend=False,
            save_path=dir_results,
            bar_plot=True, max_y=100.0)
    matplot(epoch_losses_train, f"{n_class}class Training Losses{'' if tag is None else tag}", 'Epoch',
            f'loss per batch (size = {batch_size})',
            labels=[f"Fold {i + 1}" for i in range(folds)], save_path=dir_results)
    # Plot Test loss during Training if early stopping is used
    matplot(epoch_losses_valid,
            f"{n_class}class Test Losses{'' if tag is None else tag}", 'Epoch',
            f'loss per batch (size = {batch_size})',
            labels=[f"Fold {i + 1}" for i in range(folds)], save_path=dir_results)
    train_valid_data = np.zeros((2, epoch_losses_train.shape[1]))
    train_valid_data[0] = epoch_losses_train[best_fold]
    train_valid_data[1] = epoch_losses_valid[best_fold]
    matplot(train_valid_data,
            f"{n_class}class Train-Test Losses of best Fold", 'Epoch',
            f'loss per batch (size = {batch_size})',
            labels=['Training Loss', 'Testing Loss'], save_path=dir_results)


# Generate consecutives Rectangles (vspans) to highlight Areas in plot
def create_plot_vspans(vspan_start_xs, color_idx, max_x):
    vspans = []
    for vspan in range(vspan_start_xs.shape[0]):
        if vspan == vspan_start_xs.shape[0] - 1:
            vspans.append((vspan_start_xs[vspan], max_x, color_idx[vspan]))
        else:
            vspans.append((vspan_start_xs[vspan], vspan_start_xs[vspan + 1], color_idx[vspan]))
    return vspans


# Generate vertical Lines (vlines) for plot
# Of Trials where interval of Training is highlighted
def create_vlines_from_trials_epochs(raw, vline_xs, tdelta, slices):
    vlines = []
    for trial_start_time in vline_xs:
        for i in range(1, slices + 1):
            trial_tdelta_sample = raw.time_as_index(trial_start_time + (tdelta / slices) * i)
            # -1: color_idx -> see plot.py colors[]
            vlines.append((trial_tdelta_sample, -1))
    return vlines
