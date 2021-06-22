"""
Helper functions for Plotting using matplotlib
"""
import itertools
import os
import pylab
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib import lines
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support as score

from config import PLOT_TO_PDF, N_CLASSES, BATCH_SIZE, SHOW_PLOTS
from data.datasets.phys.phys_dataset import class_labels

colors = ['tab:orange', 'tab:blue', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown',
          'skyblue', 'darkorange', 'tab:gray', 'tab:pink', 'black']


def get_color(idx):
    return colors[idx % len(colors)]


# Plots data with Matplot
# data: either 1d or 2d datasets
# labels: if 2d data, provide labels for legend
# save_path: if plot + data array should be saved, declare save location
# bar_plot: Plot as bars with average line (for Accuracies)
# vspans: List of vertical Rectangles to draw
#         Item Tuple: (X1,X2,color_idx)
# vlines: List of vertical Lines to draw
#         Item Tuple: (X,color_idx)
# hlines: List of horizontal dotted Lines to draw
#         Item Tuple: (X,color_idx)
def matplot(data, title='', xlabel='', ylabel='', labels=None, max_y=None, save_path=None, bar_plot=False,
            x_values=None, ticks=None, fig_size=None, font_size=17.0,
            hlines=[], hlines_colors=None, hlines_legends=None,
            legend_hor=False, legend_loc=None, show_legend=True,
            marker=None,
            vspans=[], vlines=[], vlines_label=None,
            min_x=None, max_x=None, color_offset=0):
    # use LaTeX fonts in the plot
    plt.rc('font', family='serif')
    if (fig_size is not None):
        plt.rcParams.update({'font.size': 22})
    else:
        plt.rcParams.update({'font.size': 10})
    if font_size is not None:
        plt.rcParams.update({'font.size': font_size})

    fig, ax = plt.subplots()
    if fig_size is not None:
        fig.set_size_inches(fig_size[0], fig_size[1])

    plt.title(title)
    plt.xlabel(xlabel)
    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
    plt.ylabel(ylabel)

    ax.set_xlim(xmin=min_x if min_x else -0.5, xmax=max_x if max_x else data.shape[-1] - 0.5)
    if x_values is not None:
        ax.set_xticklabels(x_values)

    # Avoid X-Labels overlapping
    if ticks is not None:
        if ticks.shape[0] > 30:
            plt.xticks(rotation=90)
        ax.set_xticks(ticks)
        # plt.xticks(ticks=ticks,labels=x_values)
    elif data.shape[-1] > 60:
        multiple = 10 if data.shape[-1] % 10 == 0 else 5
        plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(multiple))
        plt.xticks(rotation=90)
    elif data.shape[-1] > 30:
        multiple = 5 if data.shape[-1] % 5 == 0 else 4
        plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(multiple))
        plt.xticks(rotation=90)
    # Plot multiple lines
    if data.ndim == 2:
        for i in range(len(data)):
            plt.plot(data[i], label=labels[i] if labels is not None else "", color=get_color(i + color_offset),
                     marker=marker)
        plt.grid()
    else:
        if bar_plot:
            ax.bar(np.arange(len(data)), data, 0.35, tick_label=labels)
        else:
            plt.plot(data, label=labels[0] if labels is not None else "", color=get_color(0))
            plt.grid()

    handles, llabels = ax.get_legend_handles_labels()

    # X Labels font size
    # for tick in ax.xaxis.get_major_ticks():
    #     tick.label.set_fontsize(13)
    #     # specify integer or one of preset strings, e.g.
    #     # tick.label.set_fontsize('x-small')

    for i, hline in enumerate(hlines):
        avline = ax.axhline(hline, color=hlines_colors[i] if hlines_colors is not None else 'red', linestyle='--')

        if (show_legend is True) & (hlines_legends is not None):
            if (hlines_legends[i] is not None):
                handles.append(avline)
                llabels.append(hlines_legends[i])
    for vspan in vspans:
        plt.axvspan(vspan[0], vspan[1], color=get_color(vspan[2]), alpha=0.5)
    for vline in vlines:
        plt.axvline(vline[0], color=get_color(vline[1]), alpha=0.75, linestyle='--')

    if len(vlines) > 0 & (vlines_label is not None):
        vertical_line = lines.Line2D([], [], linestyle='--', color=get_color(vlines[0][1]),
                                     markersize=10, markeredgewidth=1.5)
        handles.append(vertical_line)
        llabels.append(vlines_label)
    if show_legend:
        if not legend_hor:
            plt.legend(handles, llabels, loc='best' if legend_loc is None else legend_loc)
        # Plot Legend below axis horizontally
        else:
            plt.legend(handles, llabels, loc='lower center', bbox_to_anchor=(0.5, -0.4),
                       ncol=4, facecolor='white', framealpha=1, edgecolor='black', fancybox=False)
            plt.subplots_adjust(bottom=0.25)

    if save_path is not None:
        fig = plt.gcf()
        # np.save(f"{save_path}/{title}.npy", data)
        # save as PDF
        fig.savefig(f"{save_path}/{title.replace(' ','_')}.png")
        if PLOT_TO_PDF:
            fig.savefig(f"{save_path}/{title.replace(' ','_')}.pdf", bbox_inches='tight')
    # fig.tight_layout()
    if SHOW_PLOTS:
        plt.show(block=True)


# Plot only Legend
def matplot_legend(labels=[], font_size=None, hor=True, save_path=None, title=None):
    # use LaTeX fonts in the plot
    plt.rc('font', family='serif')

    fig = pylab.figure()
    figlegend = pylab.figure()
    ax = fig.add_subplot(111)
    handles = []
    for i, l in enumerate(labels):
        handles.append(ax.bar(np.arange(1), np.random.randn(1), 0.1, color=get_color(i)))

    if font_size is not None:
        plt.rcParams.update({'font.size': font_size})

    fig, ax = plt.subplots()

    if not hor:
        figlegend.legend(handles, labels, loc='best')
        # Plot Legend below axis horizontally
    else:
        figlegend.legend(handles, labels, loc='lower center',
                         ncol=4, facecolor='white', framealpha=1, edgecolor='black', fancybox=False)
    # lines = ax.plot(range(10), pylab.randn(10), range(10), pylab.randn(10))
    # figlegend.legend(handles, labels, 'center')
    if SHOW_PLOTS:
        fig.show()
        figlegend.show()
    if save_path is not None:

        # np.save(f"{save_path}/{title}.npy", data)
        # save as PDF
        figlegend.savefig(f"{save_path}/{title}.png")
        if PLOT_TO_PDF:
            figlegend.savefig(f"{save_path}/{title}.pdf", bbox_inches='tight')


# Plots Benchmarking (Batch Latencies) for given configurations data (config_idx,batch_size_idx)
def matplot_grouped_configs(configs_data, batch_sizes, class_idx, title="", ylabel="",
                            save_path=None, font_size=16.0, fig_size=None, xlabel=None, hor=False,
                            min_x=None, max_x=None, conf_offset=0,
                            legend=False, legend_pos='best', xlabels=[]):
    x = np.arange(len(configs_data))  # the label locations
    width = min(0.6, (1.0 / len(batch_sizes)) - 0.1)  # the width of the bars
    x = x + (width - 0.1)
    # use LaTeX fonts in the plot
    plt.rc('font', family='serif')
    plt.rcParams.update({'font.size': font_size})
    fig, ax = plt.subplots()
    if (min_x is not None) & (max_x is not None):
        ax.set_xlim(xmin=min_x, xmax=max_x)

    if fig_size is not None:
        fig.set_size_inches(fig_size[0], fig_size[1])
    plt.grid(zorder=0)

    bs_rects = []
    bs_handles = []
    bs_labels = []
    for bs_idx, bs in enumerate(batch_sizes):
        bs_rects.append([])
        bs_labels.append(f"BS {bs}")
        for conf_idx in range(len(configs_data)):
            conf_data = configs_data[conf_idx]
            print(conf_data)
            rect_x = (conf_idx) - width / len(batch_sizes) + bs_idx * width
            if len(batch_sizes) == 1:
                rect_x = rect_x * rect_x + 0.1
            bs_rects[bs_idx].append(ax.bar(rect_x, conf_data[bs_idx][class_idx], width, color=colors[bs_idx], zorder=2))
        bs_handles.append(bs_rects[bs_idx][0])

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    plt.xlabel(xlabel)
    xlabels = xlabels
    ax.set_xticklabels(xlabels)
    # ax.legend(bs_rects, bs_labels)
    # plt.legend(bs_handles, bs_labels, loc=legend_pos)

    if legend:
        if not hor:
            plt.legend(bs_handles, bs_labels, loc=legend_pos)
            # Plot Legend below axis horizontally
        else:
            plt.legend(bs_handles, bs_labels, loc='lower center', bbox_to_anchor=(0.5, -0.4),
                       ncol=4, facecolor='white', framealpha=1, edgecolor='black', fancybox=False)
            plt.subplots_adjust(bottom=0.25)

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.6f}",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    # for rects in bs_rects:
    #     for rect in rects:
    #         autolabel(rect)

    fig.tight_layout()
    if save_path is not None:
        fig = plt.gcf()
        # save as PDF
        if PLOT_TO_PDF:
            fig.savefig(f"{save_path}/{title}.pdf", bbox_inches='tight')
        else:
            fig.savefig(f"{save_path}/{title}.png")
    if SHOW_PLOTS:
        plt.show(block=False)


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
def plot_training_statistics(dir_results, tag, run_data, batch_size, folds, early_stop):
    n_class = run_data.n_class
    accuracies = run_data.fold_accuracies
    matplot(np.append(np.roll(accuracies, 1), accuracies[-1]), f"{n_class}class Cross Validation", "Fold",
            "Accuracy in %", min_x=0.5, max_x=accuracies.shape[-1] + 0.5,
            hlines=[np.average(accuracies)],
            save_path=dir_results, show_legend=False,
            bar_plot=True, max_y=100.0)
    class_accuracies = run_data.class_accuracies[run_data.best_fold]
    matplot(class_accuracies, f"{n_class}class Accuracies{'' if tag is None else tag} of best Fold", "Class",
            "Accuracy in %", show_legend=False,
            x_values=['0'] + class_labels[n_class],
            save_path=dir_results, hlines=[np.average(class_accuracies)],
            bar_plot=True, max_y=100.0)
    matplot(run_data.epoch_losses_train, f"{n_class}class Training Losses{'' if tag is None else tag}", 'Epoch',
            f'loss per batch (size = {batch_size})', min_x=-5, max_x=run_data.epoch_losses_train.shape[-1] + 5,
            labels=[f"Fold {i + 1}" for i in range(folds)], save_path=dir_results)
    # Plot Test loss during Training if early stopping is used
    matplot(run_data.epoch_losses_test,
            f"{n_class}class Test Losses{'' if tag is None else tag}", 'Epoch',
            f'loss per batch (size = {batch_size})', min_x=-5,
            max_x=run_data.epoch_losses_test.shape[-1] + 5,
            labels=[f"Fold {i + 1}" for i in range(folds)], save_path=dir_results)
    train_test_data = np.zeros((2, run_data.epoch_losses_train.shape[1]))
    train_test_data[0] = run_data.epoch_losses_train[run_data.best_fold]
    train_test_data[1] = run_data.epoch_losses_test[run_data.best_fold]
    matplot(train_test_data,
            f"{n_class}class Train-Test Losses of best Fold", 'Epoch',
            f'loss per batch (size = {batch_size})', min_x=-5,
            max_x=train_test_data.shape[-1] + 5,
            labels=['Training Loss', 'Testing Loss'], save_path=dir_results)


# Load previous training results and replot all statistics
def load_and_plot_training(model_path):
    # TODO save best fold nr in training-results.npzs
    # -> have to manually tell which was best fold from results.txt
    best_folds = {2: 2, 3: 2, 4: 2}
    for n_class in N_CLASSES:
        stats = np.load(os.path.join(model_path, f'{n_class}class-training.npz'))
        class_accs = stats['class_accs']
        # avg_class_accuracies = get_class_avgs(n_class, stats['class_accs'])
        avg_class_accuracies = np.zeros(n_class)
        print(f"{n_class} class accs:")
        print(f"{class_accs[best_folds[n_class]]}")
        for cl in range(n_class):
            avg_class_accuracies[cl] = float(class_accs[best_folds[n_class]][cl])
        plot_training_statistics(model_path, None, n_class,
                                 stats['test_accs'],
                                 avg_class_accuracies,
                                 stats['train_losses'],
                                 stats['test_losses'],
                                 best_folds[n_class],
                                 BATCH_SIZE,
                                 5,
                                 False
                                 )


# Plot 2,3,4class Accuracy values in 1 plot for a Parameter Testing
# e.g. for Trials slicing:
# accs_2cl = [acc no_slices, acc 2_slices, acc 4_slices,...]
# defaults: 2,3,4 class accuracies of defaults configuration
# will be plotted as dotted lines
def plot_accuracies(accs_2cl, accs_3cl, accs_4cl, title,
                    x_values, save_path, defaults=[],
                    xlabel=None):
    labels = ['2class', '3class', '4class']
    x = np.zeros((3, accs_2cl.shape[-1]))
    x[0], x[1], x[2], = accs_2cl, accs_3cl, accs_4cl
    x_values = ['0'] + x_values
    matplot(data=x, title=title,
            ylabel='Accuracy in %',
            x_values=x_values,
            xlabel=xlabel,
            labels=labels,
            marker='D',
            show_legend=True,
            legend_hor=True,
            hlines=defaults,
            hlines_colors=['tab:orange', 'tab:blue', 'tab:green'],
            hlines_legends=['Defaults', None, None],
            save_path=save_path
            )


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


def plot_confusion_matrix(cm, classes, recalls, precisions, acc,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          save_path=None
                          ):
    plt.rc('font', family='serif')
    plt.rcParams.update({'font.size': 15})

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # Recall + Precision Values in x- and y-labels
    classes_with_precision = []
    for i, cl in enumerate(classes):
        classes_with_precision.append(cl + f'\n{precisions[i]:.3f}')

    classes_with_recall = []
    for i, cl in enumerate(classes):
        classes_with_recall.append(cl + f'\n{recalls[i]:.3f}')

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    # plt.title(title)
    # plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes_with_precision)
    plt.yticks(tick_marks, classes_with_recall)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    ax = plt.gca()
    ax.set_title(title, pad=28)
    ax.yaxis.tick_right()
    plt.tick_params(axis='y', which='both', labelleft=False, labelright=True)

    plt.tight_layout()
    plt.ylabel('Actual class')
    plt.xlabel('Predicted class')
    ax.xaxis.set_label_coords(0.5, 1.08)

    plt.text(1.02, 1.03, 'Recall', transform=ax.transAxes)
    plt.text(-0.33, -.14, 'Precision', transform=ax.transAxes)

    plt.text(1.04, -.14, f'{acc:.3f}', transform=ax.transAxes)

    # Grid for Precision + Recall
    ax2 = plt.axes([0, 0, 1, 1], facecolor=(0, 0, 0, 0))
    ax2.set_axis_off()
    x_axis_length = 1.5
    if len(classes) == 3:
        x_axis_length -= 0.165
    elif len(classes) == 4:
        x_axis_length -= 0.25
    for i in range(len(classes) + 1):
        x_cl = 0.0 + (x_axis_length / (len(classes) + 1)) * i
        line = lines.Line2D([x_cl, x_cl], [1.0, -.15], lw=1.5, color='black', alpha=1, transform=ax.transAxes)
        line2 = lines.Line2D([0.0, 1.225], [x_cl, x_cl], lw=1.5, color='black', alpha=1, transform=ax.transAxes)
        ax2.add_line(line)
        ax2.add_line(line2)

    if save_path is not None:
        fig = plt.gcf()
        # np.save(f"{save_path}/{title}.npy", data)
        # save as PDF
        fig.savefig(f"{save_path}/{title}.png")
        if PLOT_TO_PDF:
            fig.savefig(f"{save_path}/{title}.pdf", bbox_inches='tight')
    if SHOW_PLOTS:
        plt.show(block=False)


# Plot n_classes Confusion Matrices of Training Results
def plot_confusion_matrices(model_path, n_classes=N_CLASSES):
    for n_class in n_classes:
        actual_predicted = np.load(os.path.join(model_path, f"{n_class}class_training_actual_predicted.npz"))
        y_true, y_pred = actual_predicted['actual_labels'], actual_predicted['pred_labels']
        precisions, recalls, fscore, support = score(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred)
        conf_mat = confusion_matrix(y_true, y_pred)
        plot_confusion_matrix(conf_mat, class_labels[n_class], recalls, precisions, acc,
                              title=f'{n_class}class Confusion Matrix of best Fold',
                              save_path=model_path)

