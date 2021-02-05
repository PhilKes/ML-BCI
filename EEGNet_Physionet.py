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
from torch.autograd import profiler
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler, Subset  # noqa

from EEGNet_pytorch import EEGNet
from common import TrialsDataset, ALL_SUBJECTS, \
    print_subjects_ranges, train, test, matplot, create_results_folders, save_results, config_str, SAMPLES, \
    CHANNELS, load_n_classes_tasks, PreloadedTrialsDataset  # noqa
from config import BATCH_SIZE, LR, PLATFORM, SPLITS, CUDA, N_CLASSES, EPOCHS

def run_eegnet(num_epochs=EPOCHS, batch_size=BATCH_SIZE, splits=SPLITS, lr=LR, cuda=CUDA, n_classes=N_CLASSES,
               num_workers=0):
    config = dict(num_epochs=num_epochs, batch_size=batch_size, splits=splits, lr=lr, cuda=cuda, n_classes=n_classes)

    mne.set_log_level('WARNING')

    # Use GPU for model if available
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


    for i, n_class in enumerate(n_classes):

        cv_split = cv.split(X=ALL_SUBJECTS, groups=groups)
        start = datetime.now()
        print(f"######### {n_class}Class-Classification")
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

            ds_train, ds_test = TrialsDataset(subjects_train, n_class, device), TrialsDataset(
                subjects_test, n_class, device)
            # ds_train, ds_test = PreloadedTrialsDataset(subjects_train, n_class, device), PreloadedTrialsDataset(
            #     subjects_test, n_class, device)


            # Sample the trials in sequential order
            # TODO Random Sampler?
            sampler_train, sampler_test = SequentialSampler(ds_train), SequentialSampler(ds_test)

            loader_train, loader_test = DataLoader(ds_train, BATCH_SIZE, sampler=sampler_train, pin_memory=False,
                                                   num_workers=num_workers), \
                                        DataLoader(ds_test, BATCH_SIZE, sampler=sampler_test, pin_memory=False,
                                                   num_workers=num_workers)

            model = EEGNet(n_class)
            model.to(device)
            # with profiler.profile(profile_memory=True) as prof:
            #     with profiler.record_function("model_inference"):
            epoch_losses[split] = train(model, loader_train, epochs=num_epochs, device=device)
            accuracies[split] = test(model, loader_test, device)
            # print("PROFILING:")
            # print(prof.key_averages().table(
            #     sort_by="cpu_time_total",
            #     row_limit=10
            # ))
            # print(prof.key_averages().table(
            #     sort_by="cpu_memory_usage",
            #     row_limit=10
            # ))
        # Statistics
        print("Accuracies: ", accuracies)
        print("Avg. Accuracy: ", (sum(accuracies) / len(accuracies)))
        matplot(accuracies, f"{n_class}class Cross Validation", "Splits Iteration", "Accuracy in %", save_path=dir_results,
                box_plot=True, max_y=100.0)
        matplot(epoch_losses, f'{n_class}class-Losses over epochs', 'Epoch', f'loss per batch (size = {batch_size})',
                labels=[f"Splits {i}" for i in range(splits)], save_path=dir_results)
        elapsed = datetime.now() - start
        print(f"Elapsed time: {elapsed}")
        # Store config + results in ./results/{datetime}/results.txt
        save_results(config_str(config, n_class), n_class, accuracies, epoch_losses, elapsed, dir_results)

run_eegnet()
# run_eegnet(num_epochs=20, lr=dict(start=1e-3, milestones=[], gamma=1))
# run_eegnet(num_epochs=60)
# run_eegnet(num_epochs=60, lr=dict(start=1e-3, milestones=[], gamma=1))
