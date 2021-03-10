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
from models.EEGNet_model import EEGNet
from models.EEGNet_model_v2 import EEGNetv2
from models.ERDS_PyTorch_EEGNet import ERDS_EEGNet
from models.QEEGNet import QEEGNet
from common import train, test, benchmark
from config import BATCH_SIZE, LR, SPLITS, N_CLASSES, EPOCHS, DATA_PRELOAD, TEST_OVERFITTING, \
    trained_model_path, SAMPLES, GPU_WARMUPS, MNE_CHANNELS
from data_loading import ALL_SUBJECTS, load_subjects_data, create_loaders_from_splits, create_loader_from_subjects, \
    load_subjects_without_mne
from utils import training_config_str, create_results_folders, matplot, save_training_results, benchmark_config_str, \
    save_benchmark_results, split_list_into_chunks, save_training_numpy_data


# Torch to TensorRT for model optimizations
# https://github.com/NVIDIA-AI-IOT/torch2trt
# Comment out if TensorRt is not installed
# if torch.cuda.is_available():
#     import ctypes
#     from torch2trt import torch2trt
#
#     _cudart = ctypes.CDLL('libcudart.so')


# Runs EEGNet Training + Testing
# Cross Validation with 5 Splits (á 21 Subjects' Data)
# Can run 2/3/4-Class Classifications
# Saves Accuracies + Epochs in ./results/training/{DateTime}
# save_model: Saves trained model with highest accuracy in results folder
def eegnet_training_cv(num_epochs=EPOCHS, batch_size=BATCH_SIZE, splits=SPLITS, lr=LR, n_classes=N_CLASSES,
                       save_model=True, device=torch.device("cpu"), name=None, tag=None, ch_names=MNE_CHANNELS,
                       equal_trials=False):
    config = dict(num_epochs=num_epochs, batch_size=batch_size, splits=splits, lr=lr, device=device,
                  n_classes=n_classes, ch_names=ch_names)
    chs = len(ch_names)
    # Dont print MNE loading logs
    mne.set_log_level('WARNING')

    start = datetime.now()
    print(training_config_str(config))

    if name is None:
        dir_results = create_results_folders(datetime=start)
    else:
        dir_results = create_results_folders(path=name)

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
            preloaded_data, preloaded_labels = load_subjects_data(ALL_SUBJECTS, n_class, ch_names, equal_trials,normalize=False)
            #preloaded_data, preloaded_labels = load_subjects_without_mne(ALL_SUBJECTS, n_class)

        cv_split = cv.split(X=ALL_SUBJECTS, groups=groups)
        start = datetime.now()
        print(f"######### {n_class}Class-Classification")
        accuracies = np.zeros((splits))
        class_accuracies = np.zeros((splits, n_class))
        class_trials = np.zeros(n_class)
        accuracies_overfitting = np.zeros((splits)) if TEST_OVERFITTING else None
        epoch_losses = np.zeros((splits, num_epochs))
        # Training of the 5 different splits-combinations
        for split in range(splits):
            print(f"############ RUN {split} ############")
            # Next Splits Combination of Train/Test Datasets
            loader_train, loader_test = create_loaders_from_splits(next(cv_split), n_class, device,
                                                                   preloaded_data, preloaded_labels,
                                                                   batch_size, ch_names, equal_trials)

            # model = EEGNet(n_class, chs)
            model = QEEGNet(N=n_class, C=chs, T=SAMPLES)
            model.to(device)

            epoch_losses[split] = train(model, loader_train, epochs=num_epochs, device=device)
            print("## Validation ##")
            test_accuracy, test_class_hits = test(model, loader_test, device, n_class)
            # Test overfitting by validating on Training Dataset
            if TEST_OVERFITTING:
                print("## Validation on Training Dataset ##")
                accuracies_overfitting[split], train_class_hits = test(model, loader_train, device, n_class)
            if save_model & (n_class == 3):
                if accuracies[split] >= accuracies.max():
                    best_trained_model = model

            accuracies[split] = test_accuracy
            test_class_accuracies = np.zeros(n_class)
            print("Trials for classes:")
            for cl in range(n_class):
                class_trials[cl] = len(test_class_hits[cl])
                test_class_accuracies[cl] = (100 * (sum(test_class_hits[cl]) / len(test_class_hits[cl])))
            class_accuracies[split] = test_class_accuracies
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
        avg_class_accuracies = np.zeros(n_class)
        for j in range(n_class):
            avg_class_accuracies[j] = np.average([float(class_accuracies[sp][j]) for sp in range(splits)])

        print("Trials per class: ", *class_trials, sep="\t")
        print("Avg. Class Accuracies: ", avg_class_accuracies)
        matplot(avg_class_accuracies, f"{n_class}classes Accuracies", "Class", "Accuracy in %",
                save_path=dir_results,
                bar_plot=True, max_y=100.0)
        matplot(epoch_losses, f'{n_class}class-Losses over epochs', 'Epoch',
                f'loss per batch (size = {batch_size})',
                labels=[f"Run {i}" for i in range(splits)], save_path=dir_results)

        elapsed = datetime.now() - start
        print(f"Elapsed time: {elapsed}")
        # Store config + results in ./results/{datetime}/results.txt
        save_training_results(training_config_str(config, n_class), n_class, accuracies, avg_class_accuracies,
                              class_trials,
                              epoch_losses,
                              elapsed,
                              dir_results, accuracies_overfitting, tag)
        save_training_numpy_data(accuracies, class_accuracies, epoch_losses, dir_results, n_class)
    if save_model & (best_trained_model is not None):
        torch.save(best_trained_model.state_dict(), f"{dir_results}/trained_model.pt")


# Benchmarks pretrained EEGNet (option to use TensorRT optimizations available)
# with Physionet Dataset (3class-Classification)
# Returns Batch Latency + Time per EEG Trial inference
# saves results in ./results/benchmark/{DateTime}
def eegnet_benchmark(batch_size=BATCH_SIZE, n_classes=N_CLASSES, device=torch.device("cpu"), warm_ups=GPU_WARMUPS,
                     subjects_cs=len(ALL_SUBJECTS), tensorRT=False, iters=1, fp16=False, name=None, tag=None,
                     ch_names=MNE_CHANNELS):
    config = dict(batch_size=batch_size, device=device.type, n_classes=n_classes, subjects_cs=subjects_cs,
                  trt=tensorRT, iters=iters, fp16=fp16, ch_names=ch_names)
    chs = len(ch_names)
    # Dont print MNE loading logs
    mne.set_log_level('WARNING')

    start = datetime.now()
    print(benchmark_config_str(config))
    if name is None:
        dir_results = create_results_folders(datetime=start, type='benchmark')
    else:
        dir_results = create_results_folders(path=name, type='benchmark')

    for class_idx, n_class in enumerate(n_classes):
        print(f"######### {n_class}Class-Classification Benchmarking")
        print(f"Loading pretrained model from '{trained_model_path}'")
        model = EEGNet(n_class, chs)
        #model = QEEGNet(T=SAMPLES, C=chs)
        model.load_state_dict(torch.load(trained_model_path))
        model.to(device)
        model.eval()
        # Get optimized model with TensorRT
        # if tensorRT:
        #     t = torch.randn((batch_size, 1, SAMPLES, chs)).to(device)
        #     # add_constant() TypeError: https://github.com/NVIDIA-AI-IOT/torch2trt/issues/440
        #     # TensorRT either with fp16 ("half") or fp32
        #     if fp16:
        #         t = t.half()
        #         model = model.half()
        #     model = torch2trt(model, [t], max_batch_size=batch_size, fp16_mode=fp16)
        #     print(f"Optimized EEGNet model with TensorRT (fp{'16' if fp16 else '32'})")

        # Split ALL_SUBJECTS into chunks according to Subjects Chunk Size Parameter (due to high memory usage)
        preload_chunks = split_list_into_chunks(ALL_SUBJECTS, subjects_cs)
        chunks = len(preload_chunks)
        accuracies = np.zeros((chunks * iters))
        batch_lats = np.zeros((chunks * iters))
        trial_inf_times = np.zeros((chunks * iters))

        start = datetime.now()
        # Infer over the entire Dataset multiple times
        for i in range(iters):
            # Benchmarking is executed per chunk of subjects
            for ch_idx, subjects_chunk in enumerate(preload_chunks):
                preloaded_data, preloaded_labels = None, None
                if DATA_PRELOAD:
                    print(f"Preloading Subjects [{subjects_chunk[0]}-{subjects_chunk[-1]}] Data in memory")
                    preloaded_data, preloaded_labels = load_subjects_data(subjects_chunk, n_class)

                loader_data = create_loader_from_subjects(subjects_chunk, n_class, device, preloaded_data,
                                                          preloaded_labels, batch_size)
                # Warm up GPU with random data
                if device.type != 'cpu':
                    print("Warming up GPU")
                    for u in range(warm_ups):
                        with torch.no_grad():
                            data = torch.randn((batch_size, 1, SAMPLES, chs)).to(device)
                            y = model(data.half() if fp16 else data)
                batch_lats[chunks * i + ch_idx], trial_inf_times[chunks * i + ch_idx], accuracies[
                    chunks * i + ch_idx] = benchmark(model,
                                                     loader_data,
                                                     device,
                                                     fp16)
        elapsed = datetime.now() - start
        acc_avg = np.average(accuracies)
        batch_lat_avg = np.average(batch_lats)
        trial_inf_time_avg = np.average(trial_inf_times)
        print(f"Batch Latency:{batch_lat_avg}")
        print(f"Inference time per Trial:{trial_inf_time_avg}")
        print(f"Trials per second:{(1 / trial_inf_time_avg):.2f}")
        print(f"Elapsed Time: {str(elapsed)}")
        save_benchmark_results(benchmark_config_str(config), n_class, batch_lat_avg, trial_inf_time_avg, acc_avg,
                               elapsed, model,
                               dir_results, tag=tag)
        return batch_lat_avg, trial_inf_time_avg
