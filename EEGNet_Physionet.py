"""
Main Python Script to run
PyTorch Training + Testing for EEGNet with PhysioNet Dataset
On initial run MNE downloads the PhysioNet dataset into ./datasets
"""
import sys
from datetime import datetime

import mne
import numpy as np
import torch  # noqa
import torch.nn.functional as F  # noqa
import torch.optim as optim  # noqa
from sklearn.model_selection import GroupKFold
from torch import nn, Tensor  # noqa
from torch.autograd import profiler
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler, Subset  # noqa
from tqdm import tqdm
from EEGNet_pytorch import EEGNet
from common import TrialsDataset, ALL_SUBJECTS, \
    print_subjects_ranges, train, test, matplot, create_results_folders, save_results, config_str, \
    CHANNELS, load_n_classes_tasks, SAMPLES, trials_for_classes, load_all_subjects, create_loaders_from_splits  # noqa
from config import BATCH_SIZE, LR, PLATFORM, SPLITS, CUDA, N_CLASSES, EPOCHS, DATA_PRELOAD, TEST_OVERFITTING


# Runs EEGNet Training + Testing
# Cross Validation with 5 Splits (á 21 Subjects' Data)
# Can run 2/3/4-Class Classifications
# save_model: Saves trained model with highest accuracy
def eegnet_training_cv(num_epochs=EPOCHS, batch_size=BATCH_SIZE, splits=SPLITS, lr=LR, cuda=CUDA, n_classes=N_CLASSES,
                       save_model=True):
    config = dict(num_epochs=num_epochs, batch_size=batch_size, splits=splits, lr=lr, cuda=cuda, n_classes=n_classes)
    # Dont print MNE loading logs
    mne.set_log_level('WARNING')

    # Use GPU for model & tensors if available
    dev = None
    if cuda & torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)

    start = datetime.now()
    print(config_str(config))
    dir_results = create_results_folders(start, PLATFORM)

    # Group labels (subjects in same group need same group label)
    groups = np.zeros(len(ALL_SUBJECTS), dtype=np.int)
    group_size = int(len(ALL_SUBJECTS) / splits)
    for i in range(splits):
        groups[group_size * i:(group_size * (i + 1))] = i

    # Split Data into training + test  (84 Subjects Training, 21 Testing)
    cv = GroupKFold(n_splits=splits)

    best_trained_model = None
    for i, n_class in enumerate(n_classes):
        preloaded_data, preloaded_labels = None, None
        if DATA_PRELOAD:
            print("PRELOADING ALL DATA IN MEMORY")
            preloaded_data, preloaded_labels = load_all_subjects(n_class)

        cv_split = cv.split(X=ALL_SUBJECTS, groups=groups)
        start = datetime.now()
        print(f"######### {n_class}Class-Classification")
        accuracies = np.zeros((splits))
        accuracies_overfitting = np.zeros((splits))
        epoch_losses = np.zeros((splits, num_epochs))
        # Training of the 5 different splits-combinations
        for split in range(splits):
            print(f"############ RUN {split} ############")
            # Next Splits Combination of Train/Test Datasets
            loader_train, loader_test = create_loaders_from_splits(next(cv_split), n_class, device, preloaded_data,
                                                                   preloaded_labels)

            model = EEGNet(n_class)
            model.to(device)

            epoch_losses[split] = train(model, loader_train, epochs=num_epochs, device=device)
            print("## Validation ##")
            test_accuracy = test(model, loader_test, device)
            # Test overfitting by validating on Training Dataset
            if TEST_OVERFITTING:
                print("## Validation on Training Dataset ##")
                accuracies_overfitting[split] = test(model, loader_train, device)
            if save_model & (n_class == 3):
                if accuracies[split] >= accuracies.max():
                    best_trained_model = model

            accuracies[split] = test_accuracy
        # Statistics
        print("Accuracies on Test Dataset: ", accuracies)
        print("Avg. Accuracy: ", np.average(accuracies))
        if TEST_OVERFITTING:
            print("Accuracies on Training Dataset: ", accuracies_overfitting)
            print("Avg. Accuracy: ", np.average(accuracies_overfitting))
            print("Avg. Accuracy difference (Test-Training): ",
                  np.average(accuracies) - np.average(accuracies_overfitting))

        matplot(accuracies, f"{n_class}class Cross Validation", "Splits Iteration", "Accuracy in %",
                save_path=dir_results,
                bar_plot=True, max_y=100.0)
        matplot(epoch_losses, f'{n_class}class-Losses over epochs', 'Epoch',
                f'loss per batch (size = {batch_size})',
                labels=[f"Run {i}" for i in range(splits)], save_path=dir_results)
        elapsed = datetime.now() - start
        print(f"Elapsed time: {elapsed}")
        # Store config + results in ./results/{datetime}-PLATFORM/results.txt
        save_results(config_str(config, n_class), n_class, accuracies, epoch_losses, elapsed, dir_results,
                     accuracies_overfitting=accuracies_overfitting if TEST_OVERFITTING else None)
    if save_model & (best_trained_model is not None):
        torch.save(best_trained_model.state_dict(), f"{dir_results}/trained_model.pt")


# torch.autograd.set_detect_anomaly(True)
eegnet_training_cv()
# run_eegnet(num_epochs=20, lr=dict(start=1e-3, milestones=[], gamma=1))
# run_eegnet(num_epochs=60)
# run_eegnet(num_epochs=60, lr=dict(start=1e-3, milestones=[], gamma=1))
