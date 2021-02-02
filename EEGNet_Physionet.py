"""
Main Python Script to run
PyTorch Training + Testing for EEGNet with PhysioNet Dataset
On initial run MNE downloads the PhysioNet dataset into ./datasets
"""
from datetime import datetime

import mne
import numpy as np
import torch  # noqa
import torch.nn.functional as F  # noqa
import torch.optim as optim  # noqa
from sklearn.model_selection import GroupKFold
from torch import nn, Tensor  # noqa
from torch.utils.data import Dataset, DataLoader  # noqa
from torch.utils.data import RandomSampler  # noqa
from torch.utils.data import SequentialSampler  # noqa
from torch.utils.data import Subset  # noqa

from EEGNet_pytorch import EEGNet
from common import TrialsDataset, ALL_SUBJECTS, \
    print_subjects_ranges, train, test, matplot, create_results_folders, save_results, get_str_config  # noqa
from config import BATCH_SIZE, LR, PLATFORM, SPLITS, CUDA, N_CLASSES, EPOCHS


def run_eegnet(num_epochs=EPOCHS, batch_size=BATCH_SIZE, splits=SPLITS, lr=LR, cuda=CUDA, n_classes=N_CLASSES):
    config = dict(num_epochs=num_epochs, batch_size=batch_size, splits=splits, lr=lr, cuda=cuda, n_classes=n_classes)

    mne.set_log_level('WARNING')

    # Use GPU for model if available
    dev = None
    if cuda & torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)

    print(get_str_config(config))
    start = datetime.now()
    dir_results = create_results_folders(start, PLATFORM)

    # Group labels (subjects in same group need same group label)
    groups = np.zeros(len(ALL_SUBJECTS), dtype=np.int)
    group_size = int(len(ALL_SUBJECTS) / splits)
    for i in range(splits):
        groups[group_size * i:(group_size * (i + 1))] = i

    # Split Data into training + test  (84 Subjects Training, 21 Testing)
    cv = GroupKFold(n_splits=splits)
    cv_split = cv.split(X=ALL_SUBJECTS, groups=groups)

    for i, n_classes in enumerate(n_classes):
        accuracies = np.zeros((splits))
        epoch_losses = np.zeros((splits, num_epochs))
        # Training of the different splits (5)
        for split in range(splits):
            print(f"############ RUN {split + 1} ############")
            # Next Splits Combination of Train/Test Datasets
            subjects_train_idxs, subjects_test_idxs = next(cv_split)
            subjects_train = [ALL_SUBJECTS[idx] for idx in subjects_train_idxs]
            subjects_test = [ALL_SUBJECTS[idx] for idx in subjects_test_idxs]
            print_subjects_ranges(subjects_train, subjects_test)

            ds_train, ds_test = TrialsDataset(subjects_train, n_classes), TrialsDataset(
                subjects_test, n_classes)
            # Sample the trials in sequential order
            sampler_train, sampler_test = SequentialSampler(ds_train), SequentialSampler(ds_test)

            loader_train, loader_test = DataLoader(ds_train, BATCH_SIZE, sampler=sampler_train, pin_memory=cuda,
                                                   num_workers=0), \
                                        DataLoader(ds_test, BATCH_SIZE, sampler=sampler_test, pin_memory=cuda,
                                                   num_workers=0)

            model = EEGNet(n_classes)
            model.to(device)

            epoch_losses[split] = train(model, loader_train, epochs=num_epochs, device=device)
            accuracies[split] = test(model, loader_test, device)
        # Statistics
        print("Accuracies: ", accuracies)
        print("Avg. Accuracy: ", (sum(accuracies) / len(accuracies)))
        matplot(accuracies, "Accuracies", "Splits Iteration", "Accuracy in %", save_path=dir_results)
        matplot(epoch_losses, 'Losses over epochs', 'Epoch', 'loss / batchsize',
                labels=[f"Splits {i}" for i in range(1, splits + 1)], save_path=dir_results)
        time = datetime.now()
        elapsed = time - start
        # Store config + results in ./results/{datetime}/results.txt
        save_results(get_str_config(config), accuracies, epoch_losses, elapsed, dir_results)


run_eegnet()
run_eegnet(num_epochs=4, lr=dict(start=0.001, milestones=[], gamma=1))
