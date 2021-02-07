# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Jean-Rémi KING <jeanremi.king@gmail.com>
#
# License: BSD Style.

import copy
import functools
import math
import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import mne
import numpy as np
import torch  # noqa
import torch.nn.functional as F  # noqa
import torch.optim as optim  # noqa
from mne import Epochs, pick_types
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf
from torch import nn, Tensor  # noqa
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler, Subset  # noqa
from torch.utils.data.dataset import ConcatDataset as _ConcatDataset  # noqa
from tqdm import tqdm

from config import VERBOSE, EEG_TMIN, EEG_TMAX, results_folder, datasets_folder, LR, TRANSFORM


def combine_dims(a, i=0, n=1):
    """
    Combines dimensions of numpy array `a`,
    starting at index `i`,
    and combining `n` dimensions
    """
    s = list(a.shape)
    combined = functools.reduce(lambda x, y: x * y, s[i:i + n + 1])
    return np.reshape(a, s[:i] + [combined] + s[i + n + 1:])


# TRIALS_PER_SUBJECT = 84
CHANNELS = 64
SAMPLES = 1281

# see https://physionet.org/content/eegmmidb/1.0.0/
excluded_subjects = [88, 92, 100, 104]
ALL_SUBJECTS = [i for i in range(1, 110) if i not in excluded_subjects]

runs_rest = [1]  # Baseline, eyes open
runs_t1 = [3, 7, 11]  # Task 1 (open and close left or right fist)
runs_t2 = [4, 8, 12]  # Task 2 (imagine opening and closing left or right fist)
runs_t3 = [5, 9, 13]  # Task 3 (open and close both fists or both feet)
runs_t4 = [6, 10, 14]  # Task 4 (imagine opening and closing both fists or both feet)

runs = [runs_rest, runs_t1, runs_t2, runs_t3, runs_t4]


# Dataset for EEG Trials Data (divided by subjects)
class TrialsDataset(Dataset):

    def __init__(self, subjects, n_classes, device, transform=TRANSFORM):
        self.subjects = subjects
        # Buffers for last loaded Subject data+labels
        self.loaded_subject = -1
        self.loaded_subject_data = None
        self.loaded_subject_labels = None
        self.n_classes = n_classes
        self.runs = []
        self.device = device
        self.transform = transform
        # TODO run 0 has no Epochs?
        if (self.n_classes > 3):
            self.trials_per_subject = 147
        elif (self.n_classes == 3):
            self.trials_per_subject = 84
        elif (self.n_classes == 2):
            self.trials_per_subject = 42

    # Length of Dataset (84 Trials per Subject)
    def __len__(self):
        return len(self.subjects) * self.trials_per_subject

    # Determines corresponding Subject of trial and loads subject's data+labels
    # Uses buffer for last loaded subject
    # event: event idx (trial)
    # returns trial data(X) and trial label(y)
    def load_trial(self, trial):
        local_trial_idx = trial % self.trials_per_subject

        # determine required subject for trial
        subject = self.subjects[int(trial / self.trials_per_subject)]

        # If Subject in current buffer, skip MNE Loading
        if self.loaded_subject == subject:
            return self.loaded_subject_data[local_trial_idx], self.loaded_subject_labels[local_trial_idx]

        subject_data, subject_labels = load_n_classes_tasks(subject, self.n_classes)
        # mne_load_subject(subject, self.runs)
        self.loaded_subject = subject
        self.loaded_subject_data = subject_data
        # BCELoss excepts one-hot encoded, Cross Entropy not:
        #   labels (0,1,2) to categorical/one-hot encoded: 0 = [1 0 0], 1 =[0 1 0],...
        #   self.loaded_subject_labels = np.eye(self.n_classes, dtype='uint8')[subject_labels]
        self.loaded_subject_labels = subject_labels
        # Return single trial from all Subject's Trials
        X, y = self.loaded_subject_data[local_trial_idx], self.loaded_subject_labels[local_trial_idx]
        return X, y

    # Returns a single trial as Tensor with Labels
    def __getitem__(self, trial):
        X, y = self.load_trial(trial)
        X = torch.as_tensor(X[None, ...], device=self.device)
        #X = self.transform(X)
        return X, y


# Finds indices of label-value occurences in y and
# deletes them from X,y
def remove_label_occurences(X, y, label):
    rest_indices = np.where(y == label)
    return np.delete(X, rest_indices, axis=0), np.delete(y, rest_indices)


# Loads corresponding tasks for n_classes Classification
def load_n_classes_tasks(subject, n_classes):
    tasks = []
    if (n_classes == 4):
        tasks = [2, 4]
    elif (n_classes == 3):
        tasks = [2]
    elif (n_classes == 2):
        tasks = [2]
    return load_task_runs(subject, tasks, exclude_rest=(n_classes == 2),
                          exclude_bothfists=(n_classes == 4))


# If multiple tasks are used (4classes classification)
# labels need to be adjusted because different events from
# different tasks have the same numbers
inc_label = lambda label: label + 2 if label != 0 else label
increase_label = np.vectorize(inc_label)
# Both fists("1") gets removed, both feet("2") becomes the new "1"
map_feet_to_fists = lambda label: label - 1 if label == 2 else label
map_labels = np.vectorize(map_feet_to_fists)


# Merges runs from different tasks + correcting labels for n_class classification
def load_task_runs(subject, tasks, exclude_rest=False, exclude_bothfists=False):
    global map_label
    all_data = np.zeros((0, SAMPLES, CHANNELS))
    all_labels = np.zeros((0), dtype=np.int)
    for i, task in enumerate(tasks):
        data, labels = mne_load_subject(subject, runs[task])
        # for 2class classification exclude Rest events("0")
        # (see Paper "An Accurate EEGNet-based Motor-Imagery Brain–Computer ... ")
        if exclude_rest:
            data, labels = remove_label_occurences(data, labels, 0)
            labels = labels - 1
        # for 4class classification exclude both fists event("1")
        if exclude_bothfists & (task == 4):
            data, labels = remove_label_occurences(data, labels, 1)
            labels = map_labels(labels)
        # Correct labels if multiple tasks are loaded
        # e.g. in Task 2: "1": left fist, in Task 4: "1": both fists
        for n in range(i):
            labels = increase_label(labels)
        all_data = np.concatenate((all_data, data))
        all_labels = np.concatenate((all_labels, labels))
    return all_data, all_labels


# Loads single Subject from mne
# returns EEG data (X) and corresponding Labels (y)
def mne_load_subject(subject, runs):
    if VERBOSE:
        print(f"MNE loading Subject {subject}")
    # for 4 Class: need to map to 0,1,2,3
    # split reading in run lists (runs_t1,runs_t2,...)
    # give unique labels
    raw_fnames = eegbci.load_data(subject, runs, datasets_folder)
    raw_files = [read_raw_edf(f, preload=True) for f in raw_fnames]
    raw = concatenate_raws(raw_files)
    raw.rename_channels(lambda x: x.strip('.'))
    events, event_ids = mne.events_from_annotations(raw, event_id='auto')
    picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                       exclude='bads')
    epochs = Epochs(raw, events, event_ids, EEG_TMIN, EEG_TMAX, picks=picks,
                    baseline=None, preload=True)
    # [trials (84), timepoints (1281), channels (64)]
    subject_data = np.swapaxes(epochs.get_data(), 2, 1)
    # Labels (0-index based)
    subject_labels = epochs.events[:, -1] - 1

    return subject_data, subject_labels


# Training
# see https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#train-the-network
def train(net, data_loader, epochs=1, device=torch.device("cpu"), lr=LR):
    net.train()
    # Init Loss Function + Optimizer with Learning Rate Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr['start'])
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr['milestones'], gamma=lr['gamma'])
    print("###### Training started")
    loss_values = np.zeros((epochs))
    for epoch in range(epochs):
        print(f"## Epoch {epoch} ")
        running_loss = 0.0
        # Wrap in tqdm for Progressbar
        pbar = tqdm(data_loader, file=sys.stdout)
        # Training in batches from the DataLoader
        for idx_batch, (inputs, labels) in enumerate(pbar):
            # Convert to correct types + put on GPU
            inputs, labels = inputs.float(), labels.long().to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            # if math.isnan(running_loss):
            #     print("running_loss", running_loss)
            #     print("loss.item()", loss.item())
            #     print("outputs", outputs.shape)
            #     print("labels", labels.shape)
            # pbar.set_description(
            #     desc=f"Batch {idx_batch} avg. loss: {running_loss / (idx_batch + 1):.4f}")
        pbar.close()
        lr_scheduler.step()
        # Loss of entire epoch / amount of batches
        loss_values[epoch] = (running_loss / len(data_loader))
        print('[%3d] Total loss: %f' %
              (epoch, running_loss))
        if math.isnan(loss_values[epoch]):
            print("loss_values[epoch]", loss_values[epoch])
            print("running_loss", running_loss)
            print("len(data_loader)", len(data_loader))
            break
    print("Training finished ######")

    return loss_values


def test(net, data_loader, device=torch.device("cpu")):
    print("###### Testing started")
    total = 0.0
    correct = 0.0
    with torch.no_grad():
        net.eval()
        for data in data_loader:
            inputs, labels = data
            inputs, labels = inputs.float(), labels.float()
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data.cpu(), 1)
            # For BCELOSS:
            #   revert np.eye -> [0 0 1] = 2, [1 0 0] = 0
            #   labels = np.argmax(labels.cpu(), axis=1)
            labels = labels.cpu()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = (100 * correct / total)
    print(F'Accuracy on the {len(data_loader.dataset)} test trials: %0.2f %%' % (
        acc))
    print("Testing finished ######")
    return acc


def subjects_without_excluded(subjects):
    return [i for i in subjects if i not in [88, 89, 92, 100]]


# Plots data with Matplot
# data: either 1d or 2d datasets
# labels: if 2d data, provide labels for legend
# save_path: if plot + data array should be saved, declare save location
def matplot(data, title='', xlabel='', ylabel='', labels=[], max_y=None, save_path=None, box_plot=False):
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
        if box_plot:
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


# Saves config + results.txt in dir_results
def save_results(str_conf, n_class, accuracies, epoch_losses, elapsed, dir_results):
    str_elapsed = str(elapsed)
    file_result = open(f"{dir_results}/{n_class}class-results.txt", "w+")
    file_result.write(str_conf)
    file_result.write(f"Elapsed Time: {str_elapsed}\n")
    file_result.write(f"Accuracies of Splits:\n")
    for i in [f"\tSplits {i}: {accuracies[i - 1]:.2f}\n" for i in range(1, len(accuracies) + 1)]:
        file_result.write(i)
    file_result.write(f"Average acc: {accuracies.sum() / len(accuracies):.2f}")
    file_result.close()


def print_subjects_ranges(train, test):
    if (train[0] < test[0]) & (train[-1] < test[0]):
        print(f"Subjects for Training:\t[{train[0]}-{train[-1]}]")
    elif (train[0] < test[0]) & (train[-1] > test[0]):
        print(f"Subjects for Training:\t[{train[0]}-{test[0] - 1}],[{test[-1] + 1}-{train[-1]}]")
    elif (train[0] > test[0]):
        print(f"Subjects for Training:\t[{train[0]}-{train[-1]}]")
    print(f"Subjects for Testing:\t[{test[0]}-{test[-1]}]")
    return


# Create results folder with current DateTime-PLATFORM as name
def create_results_folders(datetime, platform="PC"):
    now_string = datetime.strftime("%Y-%m-%d %H_%M_%S")
    results = f"{results_folder}/{now_string}-{platform}"
    try:
        os.mkdir(results)
    except OSError as err:
        pass
    return results


str_n_classes = ["", "", "Left/Right Fist", "Left/Right-Fist / Rest", "Left/Right-Fist / Rest / Both-Feet"]


def get_str_n_classes(n_classes):
    return f'Classes: {[str_n_classes[i] for i in n_classes]}'


def config_str(config, n_class=None):
    return f"""#### Config ####
CUDA: {config['cuda']}
Nr. of classes: {config['n_classes'] if n_class is None else n_class}
{get_str_n_classes(config['n_classes'] if n_class is None else [n_class])}
Dataset split in {config['splits']} Subject Groups, {config['splits'] - 1} for Training, {1} for Testing (Cross Validation)
Batch Size: {config['batch_size']}
Epochs: {config['num_epochs']}
Learning Rate: initial = {config['lr']['start']}, Epoch milestones = {config['lr']['milestones']}, gamma = {config['lr']['gamma']}
###############\n"""


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

########################### NOT USED ######################################################

# def _do_train(model, loader, optimizer, criterion, device):
#     # training loop
#     model.train()
#     pbar = tqdm(loader)
#     train_loss = np.zeros(len(loader))
#     for idx_batch, (batch_x, batch_y) in enumerate(pbar):
#         optimizer.zero_grad()
#         batch_x = batch_x.to(device=device, dtype=torch.float32)
#         batch_y = batch_y.to(device=device, dtype=torch.int64)
#
#         output = model(batch_x)
#         loss = criterion(output, batch_y)
#
#         loss.backward()
#         optimizer.step()
#
#         train_loss[idx_batch] = loss.item()
#         pbar.set_description(
#             desc="avg train loss: {:.4f}".format(
#                 np.mean(train_loss[:idx_batch + 1])))
#
#
# def _validate(model, loader, criterion, device):
#     # validation loop
#     pbar = tqdm(loader)
#     val_loss = np.zeros(len(loader))
#     accuracy = 0.
#     with torch.no_grad():
#         model.eval()
#
#         for idx_batch, (batch_x, batch_y) in enumerate(pbar):
#             batch_x = batch_x.to(device=device, dtype=torch.float32)
#             batch_y = batch_y.to(device=device, dtype=torch.int64)
#             output = model.forward(batch_x)
#
#             loss = criterion(output, batch_y)
#             val_loss[idx_batch] = loss.item()
#
#             _, top_class = output.topk(1, dim=1)
#             top_class = top_class.flatten()
#             # print(top_class.shape, batch_y.shape)
#             accuracy += \
#                 torch.sum((batch_y == top_class).to(torch.float32))
#
#             pbar.set_description(
#                 desc="avg val loss: {:.4f}".format(
#                     np.mean(val_loss[:idx_batch + 1])))
#
#     accuracy = accuracy / len(loader.dataset)
#     print("---  Accuracy : %s" % accuracy.item(), "\n")
#     return np.mean(val_loss)
#
#
# class PreloadedTrialsDataset(Dataset):
#
#     def __init__(self, subjects, n_classes, device):
#         self.subjects = subjects
#         # Buffers for last loaded Subject data+labels
#         self.loaded_subject = -1
#         self.loaded_subject_data = None
#         self.loaded_subject_labels = None
#         self.n_classes = n_classes
#         self.runs = []
#         self.device = device
#         # TODO run 0 has no Epochs?
#         if (self.n_classes > 3):
#             self.trials_per_subject = 147
#         elif (self.n_classes == 3):
#             self.trials_per_subject = 84
#         elif (self.n_classes == 2):
#             self.trials_per_subject = 42
#
#     # Length of Dataset (84 Trials per Subject)
#     def __len__(self):
#         return len(self.subjects) * self.trials_per_subject
#
#     # Determines corresponding Subject of trial and loads subject's data+labels
#     # Uses buffer for last loaded subject
#     # event: event idx (trial)
#     # returns trial data(X) and trial label(y)
#     def load_trial(self, trial):
#         local_trial_idx = trial % self.trials_per_subject
#
#         # determine required subject for trial
#         subject = self.subjects[int(trial / self.trials_per_subject)]
#
#         # If Subject in current buffer, skip MNE Loading
#         if self.loaded_subject == subject:
#             return self.loaded_subject_data[local_trial_idx], self.loaded_subject_labels[local_trial_idx]
#
#         subject_data, subject_labels = load_n_classes_tasks(subject, self.n_classes)
#         # mne_load_subject(subject, self.runs)
#         self.loaded_subject = subject
#         self.loaded_subject_data = subject_data
#         # BCELoss excepts one-hot encoded, Cross Entropy not
#         # labels (0,1,2) to categorical/one-hot encoded: 0 = [1 0 0], 1 =[0 1 0],...
#         # self.loaded_subject_labels = np.eye(self.n_classes, dtype='uint8')[subject_labels]
#         self.loaded_subject_labels = subject_labels
#         # Return single trial from all Subject's Trials
#         X, y = self.loaded_subject_data[local_trial_idx], self.loaded_subject_labels[local_trial_idx]
#         return X, y
#
#     # Returns a single trial
#     def __getitem__(self, trial):
#         X, y = self.load_trial(trial)
#         X = torch.as_tensor(X[None, ...], device=self.device, dtype=torch.float)
#         return X, y

#######
