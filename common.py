"""
Helper functions
"""
import math
import sys

import numpy as np
import torch  # noqa
import torch.nn.functional as F  # noqa
import torch.optim as optim  # noqa
from torch import nn, Tensor  # noqa
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler, Subset  # noqa
from torch.utils.data.dataset import ConcatDataset as _ConcatDataset  # noqa
from tqdm import tqdm

from config import LR


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


# Tests labeled data with trained net
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





########################### NOT USED ######################################################

# def combine_dims(a, i=0, n=1):
#     """
#     Combines dimensions of numpy array `a`,
#     starting at index `i`,
#     and combining `n` dimensions
#     """
#     s = list(a.shape)
#     combined = functools.reduce(lambda x, y: x * y, s[i:i + n + 1])
#     return np.reshape(a, s[:i] + [combined] + s[i + n + 1:])

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
