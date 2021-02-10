"""
* Training and Validating of Physionet Dataset with EEGNet PyTorch implementation
* Performance Benchmarking of Inference on EEGNet pretrained with Physionet Data
"""
from datetime import datetime

import mne
import numpy as np
import torch  # noqa
import torch.nn.functional as F  # noqa
import torch.optim as optim  # noqa
from sklearn.model_selection import GroupKFold
from torch import nn, Tensor  # noqa
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler, Subset  # noqa

from EEGNet_model import EEGNet
from common import train, test, benchmark
from config import BATCH_SIZE, LR, PLATFORM, SPLITS, CUDA, N_CLASSES, EPOCHS, DATA_PRELOAD, TEST_OVERFITTING, \
    trained_model_path, SAMPLES, CHANNELS
# Runs EEGNet Training + Testing
# Cross Validation with 5 Splits (á 21 Subjects' Data)
# Can run 2/3/4-Class Classifications
# save_model: Saves trained model with highest accuracy
from data_loading import ALL_SUBJECTS, load_subjects_data, create_loaders_from_splits, create_loader_from_subjects
from utils import training_config_str, create_results_folders, matplot, save_training_results, benchmark_config_str, \
    save_benchmark_results

# TensorRT
# https://github.com/NVIDIA-AI-IOT/torch2trt
if torch.cuda.is_available():
    import ctypes
    from torch2trt import torch2trt

    _cudart = ctypes.CDLL('libcudart.so')


def eegnet_training_cv(num_epochs=EPOCHS, batch_size=BATCH_SIZE, splits=SPLITS, lr=LR, n_classes=N_CLASSES,
                       save_model=True, device=torch.device("cpu")):
    config = dict(num_epochs=num_epochs, batch_size=batch_size, splits=splits, lr=lr, cuda=CUDA, n_classes=n_classes)
    # Dont print MNE loading logs
    mne.set_log_level('WARNING')

    start = datetime.now()
    print(training_config_str(config))

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
            preloaded_data, preloaded_labels = load_subjects_data(ALL_SUBJECTS, n_class)

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
        save_training_results(training_config_str(config, n_class), n_class, accuracies, epoch_losses, elapsed,
                              dir_results,
                              accuracies_overfitting=accuracies_overfitting if TEST_OVERFITTING else None)
    if save_model & (best_trained_model is not None):
        torch.save(best_trained_model.state_dict(), f"{dir_results}/trained_model.pt")


def eegnet_benchmark(batch_size=BATCH_SIZE, n_classes=N_CLASSES, device=torch.device("cpu"), warm_ups=5,
                     subjects=ALL_SUBJECTS, tensorRT=False):
    config = dict(batch_size=batch_size, cuda=CUDA, n_classes=n_classes, subjects=len(subjects))
    # Dont print MNE loading logs
    mne.set_log_level('WARNING')

    start = datetime.now()
    print(benchmark_config_str(config))
    dir_results = create_results_folders(start, PLATFORM, type='benchmark')

    for i, n_class in enumerate(n_classes):
        preloaded_data, preloaded_labels = None, None
        if DATA_PRELOAD:
            print("PRELOADING ALL DATA IN MEMORY")
            preloaded_data, preloaded_labels = load_subjects_data(subjects, n_class)

        start = datetime.now()
        print(f"######### {n_class}Class-Classification")

        loader_data = create_loader_from_subjects(subjects, n_class, device, preloaded_data,
                                                  preloaded_labels)

        # Load pretrained model
        model = EEGNet(n_class)
        model.load_state_dict(torch.load(trained_model_path))
        model.to(device)
        # Get optimized model from TensorRT
        if tensorRT:
            sample_data = torch.randn((batch_size, 1, SAMPLES, CHANNELS)).to(device)
            model = torch2trt(model,[sample_data],max_batch_size=batch_size)
            print("Optimized EEGNet model with TensorRT")

        # Warm up GPU
        if CUDA:
            print("Warm up GPU")
            data_iter = iter(loader_data)
            for i in range(warm_ups):
                data, labels = next(data_iter)
                y = model(data)
        batch_lat, trial_inf_time = benchmark(model, loader_data, device)
        elapsed = datetime.now() - start
        print(f"Batch Latency:{batch_lat}")
        print(f"Inference time per Trial:{trial_inf_time}")
        print(f"Trials per second:{(1 / trial_inf_time):.2f}")
        save_benchmark_results(benchmark_config_str(config), n_class, batch_lat, trial_inf_time, elapsed, model,
                               dir_results)
