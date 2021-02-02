# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Jean-Rémi KING <jeanremi.king@gmail.com>
#
# License: BSD Style.

import copy
import functools
import os
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import mne
import numpy as np
import torch  # noqa
import torch  # noqa
import torch  # noqa
import torch.nn.functional as F  # noqa
import torch.nn.functional as F  # noqa
import torch.nn.functional as F  # noqa
import torch.optim as optim  # noqa
import torch.optim as optim  # noqa
import torch.optim as optim  # noqa
from mne import Epochs, pick_types
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf
from torch import nn, Tensor  # noqa
from torch import nn, Tensor  # noqa
from torch import nn, Tensor  # noqa
from torch.utils.data import Dataset, DataLoader  # noqa
from torch.utils.data import Dataset, DataLoader  # noqa
from torch.utils.data import Dataset, DataLoader  # noqa
from torch.utils.data import RandomSampler  # noqa
from torch.utils.data import RandomSampler  # noqa
from torch.utils.data import RandomSampler  # noqa
from torch.utils.data import SequentialSampler  # noqa
from torch.utils.data import SequentialSampler  # noqa
from torch.utils.data import SequentialSampler  # noqa
from torch.utils.data import Subset  # noqa
from torch.utils.data import Subset  # noqa
from torch.utils.data import Subset  # noqa
from torch.utils.data.dataset import ConcatDataset as _ConcatDataset  # noqa
from tqdm import tqdm

from config import VERBOSE, EEG_TMIN, EEG_TMAX, results_folder, datasets_folder, LR


def combine_dims(a, i=0, n=1):
    """
    Combines dimensions of numpy array `a`,
    starting at index `i`,
    and combining `n` dimensions
    """
    s = list(a.shape)
    combined = functools.reduce(lambda x, y: x * y, s[i:i + n + 1])
    return np.reshape(a, s[:i] + [combined] + s[i + n + 1:])


TRIALS_PER_SUBJECT = 84
CHANNELS = 64
SAMPLES = 1281

# see https://physionet.org/content/eegmmidb/1.0.0/
runs_rest = [1]  # Baseline, eyes open
runs_t1 = [3, 7, 11]  # Task 1 (open and close left or right fist)
runs_t2 = [4, 8, 12]  # Task 2 (imagine opening and closing left or right fist)
runs_t3 = [5, 9, 13]  # Task 3 (open and close both fists or both feet)
runs_t4 = [6, 10, 14]  # Task 4 (imagine opening and closing both fists or both feet)


# Dataset for EEG Trials Data (divided by subjects)
class TrialsDataset(Dataset):

    def __init__(self, subjects, n_classes):
        self.subjects = subjects
        # Buffers for last loaded Subject data+labels
        self.loaded_subject = -1
        self.loaded_subject_data = None
        self.loaded_subject_labels = None
        self.n_classes = n_classes
        self.runs = []

        if (self.n_classes == 3):
            self.runs = runs_rest + runs_t2
        elif (self.n_classes == 2):
            self.runs = runs_t2

    # Length of Dataset (84 Trials per Subject)
    def __len__(self):
        return len(self.subjects) * TRIALS_PER_SUBJECT

    # Determines corresponding Subject of trial and loads subject's data+labels
    # Uses buffer for last loaded subject
    # event: event idx (trial)
    # returns trial data(X) and trial label(y)
    def load_trial(self, trial):
        local_trial_idx = trial % TRIALS_PER_SUBJECT

        # determine required subject for trial
        subject = self.subjects[int(trial / TRIALS_PER_SUBJECT)]

        # If Subject in current buffer, skip MNE Loading
        if self.loaded_subject == subject:
            return self.loaded_subject_data[local_trial_idx], self.loaded_subject_labels[local_trial_idx]

        subject_data, subject_labels = mne_load_subject(subject, self.runs)
        self.loaded_subject = subject
        self.loaded_subject_data = subject_data
        # BCELoss excepts one-hot encoded, Cross Entropy not
        # labels (0,1,2) to categorical/one-hot encoded: 0 = [1 0 0], 1 =[0 1 0],...
        # self.loaded_subject_labels = np.eye(self.n_classes, dtype='uint8')[subject_labels]
        self.loaded_subject_labels = subject_labels
        # Return single trial from all Subject's Trials
        X, y = self.loaded_subject_data[local_trial_idx], self.loaded_subject_labels[local_trial_idx]
        return X, y

    # Returns a single trial
    def __getitem__(self, trial):
        X, y = self.load_trial(trial)
        X = torch.as_tensor(X[None, ...])
        return X, y


excluded_subjects = [88, 92, 100, 104]
ALL_SUBJECTS = [i for i in range(1, 110) if i not in excluded_subjects]


# Loads single Subject from mne
# returns EEG data (X) and corresponding Labels (y)
def mne_load_subject(subject, runs):
    if VERBOSE:
        print(f"MNE loading Subject {subject}")
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
    subject_labels = epochs.events[:, -1] - 1

    return subject_data, subject_labels


# Training
# see https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#train-the-network
def train(net, data_loader, epochs=1, device=torch.device("cpu"), lr=LR):
    # Init Loss Function + Optimizer with Learning Rate Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr['start'])
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr['milestones'], gamma=lr['gamma'])
    print("###### Training started")
    loss_values = []
    for epoch in range(epochs):
        print(f"## Epoch {epoch} ")
        running_loss = 0.0
        # Training in batches from the DataLoader
        for i, data in enumerate(data_loader):
            inputs, labels = data
            # Convert to correct types + put on GPU
            inputs, labels = inputs.float().to(device), labels.long().to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        lr_scheduler.step()
        # Loss of entire epoch / amount of batches
        loss_values.append(running_loss / len(data_loader))
        print('[%3d] loss: %f' %
              (epoch + 1, running_loss))
    print("Training finished ######")

    return loss_values


def test(net, data_loader, device=torch.device("cpu")):
    print("###### Testing started")
    total = 0.0
    correct = 0.0
    with torch.no_grad():
        for data in data_loader:
            inputs, labels = data
            inputs, labels = Tensor(inputs.float()).to(device), Tensor(labels.float()).to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data.cpu(), 1)
            # For BCELOSS:
            # revert np.eye -> [0 0 1] = 2, [1 0 0] = 0
            # labels = np.argmax(labels.cpu(), axis=1)
            labels = labels.cpu()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = (100 * correct / total)
    print(F'Accuracy on the {len(data_loader.dataset)} test events: %0.2f %%' % (
        acc))
    print("Testing finished ######")
    return acc


def subjects_without_excluded(subjects):
    x = [i for i in subjects if i not in [88, 89, 92, 100]]
    return x


# Plots data with Matplot
# data: either 1d or 2d datasets
# labels: if 2d data, provide labels for legend
# save_path: if plot should be saved, declare save location
def matplot(data, title='', xlabel='', ylabel='', labels=[], save_path=None):
    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
    plt.ylabel(ylabel)
    plt.grid()
    if data.ndim == 2:
        for i in range(len(data)):
            plt.plot(data[i], label=labels[i] if len(labels) >= i else "")
            plt.legend()
    else:
        plt.plot(data, label=labels[0] if len(labels) > 0 else "")
    if save_path is not None:
        fig = plt.gcf()
        fig.savefig(f"{save_path}/{title}.png")
    plt.show()


# Saves config + results.txt in dir_results
def save_results(str_conf, accuracies, epoch_losses, elapsed, dir_results):
    str_elapsed = str(elapsed)
    file_result = open(f"{dir_results}/results.txt", "w+")
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


def get_str_config(config):
    return f"""#### Config ####
CUDA: {config['cuda']}
Number of classes: {config['n_classes']}
Dataset split in {config['splits']} Subject Groups, {config['splits'] - 1} for Training, {1} for Testing (Cross Validation)
Batch Size: {config['batch_size']}
Epochs: {config['num_epochs']}
Learning Rate: initial = {config['lr']['start']}, Epoch milestones = {config['lr']['milestones']}, gamma = {config['lr']['gamma']}
###############\n"""


################################################### NOT USED ######################################################
class ConcatDataset(_ConcatDataset):
    """
    Same as torch.utils.data.dataset.ConcatDataset, but exposes an extra
    method for querying the group structure (index if dataset
    each sample comes from)
    """

    def get_groups(self):
        """Return the group index of each sample

        Returns
        -------
        groups : array of int, shape (n_samples,)
            The group indices.
        """
        groups = [k * np.ones(len(d)) for k, d in enumerate(self.datasets)]
        return np.concatenate(groups)


class EpochsDataset(Dataset):
    """Class to expose an MNE Epochs object as PyTorch dataset

    Parameters
    ----------
    epochs_data : 3d array, shape (n_epochs, n_channels, n_times)
        The epochs data.
    epochs_labels : array of int, shape (n_epochs,)
        The epochs labels.
    transform : callable | None
        The function to eventually apply to each epoch
        for preprocessing (e.g. scaling). Defaults to None.
    """

    def __init__(self, epochs_data, epochs_labels, transform=None):
        assert len(epochs_data) == len(epochs_labels)
        self.epochs_data = epochs_data
        self.epochs_labels = epochs_labels
        self.transform = transform

    def __len__(self):
        return len(self.epochs_labels)

    def __getitem__(self, idx):
        X, y = self.epochs_data[idx], self.epochs_labels[idx]
        if self.transform is not None:
            X = self.transform(X)
        X = torch.as_tensor(X[None, ...])
        print("X", X.shape)
        print("y", y.shape)
        return X, y


def _do_train(model, loader, optimizer, criterion, device):
    # training loop
    model.train()
    pbar = tqdm(loader)
    train_loss = np.zeros(len(loader))
    for idx_batch, (batch_x, batch_y) in enumerate(pbar):
        optimizer.zero_grad()
        batch_x = batch_x.to(device=device, dtype=torch.float32)
        batch_y = batch_y.to(device=device, dtype=torch.int64)

        output = model(batch_x)
        loss = criterion(output, batch_y)

        loss.backward()
        optimizer.step()

        train_loss[idx_batch] = loss.item()
        pbar.set_description(
            desc="avg train loss: {:.4f}".format(
                np.mean(train_loss[:idx_batch + 1])))


def _validate(model, loader, criterion, device):
    # validation loop
    pbar = tqdm(loader)
    val_loss = np.zeros(len(loader))
    accuracy = 0.
    with torch.no_grad():
        model.eval()

        for idx_batch, (batch_x, batch_y) in enumerate(pbar):
            batch_x = batch_x.to(device=device, dtype=torch.float32)
            batch_y = batch_y.to(device=device, dtype=torch.int64)
            output = model.forward(batch_x)

            loss = criterion(output, batch_y)
            val_loss[idx_batch] = loss.item()

            _, top_class = output.topk(1, dim=1)
            top_class = top_class.flatten()
            # print(top_class.shape, batch_y.shape)
            accuracy += \
                torch.sum((batch_y == top_class).to(torch.float32))

            pbar.set_description(
                desc="avg val loss: {:.4f}".format(
                    np.mean(val_loss[:idx_batch + 1])))

    accuracy = accuracy / len(loader.dataset)
    print("---  Accuracy : %s" % accuracy.item(), "\n")
    return np.mean(val_loss)


def train_old(model, loader_train, loader_valid, optimizer, n_epochs, patience,
              device):
    """Training function

    Parameters
    ----------
    model : instance of nn.Module
        The model.
    loader_train : instance of Sampler
        The generator of EEG samples the model has to train on.
        It contains n_train samples
    loader_valid : instance of Sampler
        The generator of EEG samples the model has to validate on.
        It contains n_val samples. The validation samples are used to
        monitor the training process and to perform early stopping
    optimizer : instance of optimizer
        The optimizer to use for training.
    n_epochs : int
        The maximum of epochs to run.
    patience : int
        The patience parameter, i.e. how long to wait for the
        validation error to go down.
    device : str | instance of torch.device
        The device to train the model on.

    Returns
    -------
    best_model : instance of nn.Module
        The model that lead to the best prediction on the validation
        dataset.
    """
    # put model on cuda if not already
    device = torch.device(device)
    # model.to(device)

    # define criterion
    criterion = F.nll_loss

    best_val_loss = + np.infty
    best_model = copy.deepcopy(model)
    waiting = 0

    for epoch in range(n_epochs):
        print("\nStarting epoch {} / {}".format(epoch + 1, n_epochs))
        _do_train(model, loader_train, optimizer, criterion, device)
        val_loss = _validate(model, loader_valid, criterion, device)

        # model saving
        if np.mean(val_loss) < best_val_loss:
            print("\nbest val loss {:.4f} -> {:.4f}".format(
                best_val_loss, np.mean(val_loss)))
            best_val_loss = np.mean(val_loss)
            best_model = copy.deepcopy(model)
            waiting = 0
        else:
            print("Waiting += 1")
            waiting += 1

        # model early stopping
        if waiting >= patience:
            print("Stop training at epoch {}".format(epoch + 1))
            print("Best val loss : {:.4f}".format(best_val_loss))
            break

    return best_model
#######
