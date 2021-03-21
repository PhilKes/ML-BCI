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

from common import train, test, benchmark, predict
from config import BATCH_SIZE, LR, SPLITS, N_CLASSES, EPOCHS, DATA_PRELOAD, TEST_OVERFITTING, SAMPLES, GPU_WARMUPS, \
    MNE_CHANNELS, trained_model_name, training_results_folder, VALIDATION_SUBJECTS, live_sim_results_folder, \
    global_config
from data_loading import ALL_SUBJECTS, load_subjects_data, create_loaders_from_splits, create_loader_from_subjects, \
    mne_load_subject_raw, get_data_from_raw, get_label_at_idx, n_classes_live_run
from models.eegnet import EEGNet
from util.dot_dict import DotDict
from util.utils import training_config_str, create_results_folders, matplot, save_training_results, \
    benchmark_config_str, \
    save_benchmark_results, split_list_into_chunks, save_training_numpy_data, benchmark_result_str, save_config, \
    training_result_str, live_sim_config_str


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
# Saves Accuracies + Epochs in ./results/{DateTime/name}/training
# save_model: Saves trained model with highest accuracy in results folder
def physionet_training_cv(num_epochs=EPOCHS, batch_size=BATCH_SIZE, folds=SPLITS, lr=LR, n_classes=N_CLASSES,
                          save_model=True, device=torch.device("cpu"), name=None, tag=None, ch_names=MNE_CHANNELS,
                          equal_trials=False, early_stop=True, excluded=[]):
    config = DotDict(num_epochs=num_epochs, batch_size=batch_size, folds=folds, lr=lr, device=device,
                     n_classes=n_classes, ch_names=ch_names, early_stop=early_stop, excluded=excluded)
    chs = len(ch_names)
    # Dont print MNE loading logs
    mne.set_log_level('WARNING')
    start = datetime.now()
    print(training_config_str(config))
    if name is None:
        dir_results = create_results_folders(datetime=start)
    else:
        dir_results = create_results_folders(path=name)
    save_config(training_config_str(config), ch_names, dir_results, tag)

    available_subjects = [i for i in ALL_SUBJECTS if i not in excluded]
    # 84 Subjects for Train + 21 for Test (get split in 5 different Splits)
    used_subjects = available_subjects
    validation_subjects = []
    # Currently if early_stop=true:
    # Validation Set = Test Set
    # 0 validation subjects, train() evaluates valid_loss on loader_test
    if early_stop & (VALIDATION_SUBJECTS > 0):
        # 76 Subjects (~72%) for Train + 19 (~18%) for Test (get split in 5 different Splits)
        used_subjects = available_subjects[:(len(available_subjects) - VALIDATION_SUBJECTS)]
        # 10 (~10%) Subjects for Validation (always the same)
        validation_subjects = available_subjects[(len(available_subjects) - VALIDATION_SUBJECTS):]
        print(f"Validation Subjects: [{validation_subjects[0]}-{validation_subjects[-1]}]")

    # Group labels (subjects in same group need same group label)
    groups = np.zeros(len(used_subjects), dtype=np.int)
    group_size = int(len(used_subjects) / folds)
    for i in range(folds):
        groups[group_size * i:(group_size * (i + 1))] = i

    # Split Data into training + test
    cv = GroupKFold(n_splits=folds)

    best_n_class_models = {}
    for i, n_class in enumerate(n_classes):
        preloaded_data, preloaded_labels = None, None
        if DATA_PRELOAD:
            print("PRELOADING ALL DATA IN MEMORY")
            preloaded_data, preloaded_labels = load_subjects_data(used_subjects + validation_subjects, n_class,
                                                                  ch_names, equal_trials,
                                                                  normalize=False)

        cv_split = cv.split(X=used_subjects, groups=groups)
        start = datetime.now()
        print(f"######### {n_class}Class-Classification")
        accuracies = np.zeros((folds))
        best_losses_valid, best_epochs_valid = np.full((folds), fill_value=np.inf), np.zeros((folds), dtype=np.int)
        best_fold = -1
        class_accuracies = np.zeros((folds, n_class))
        class_trials = np.zeros(n_class)
        accuracies_overfitting = np.zeros((folds)) if TEST_OVERFITTING else None
        epoch_losses_train, epoch_losses_valid = np.zeros((folds, num_epochs)), np.zeros((folds, num_epochs))
        # Training of the 5 Folds with the different splits
        for fold in range(folds):
            print(f"############ RUN {fold} ############")
            # Next Splits Combination of Train/Test Datasets + Validation Set Loader
            loaders = create_loaders_from_splits(next(cv_split), validation_subjects, n_class, device, preloaded_data,
                                                 preloaded_labels, batch_size, ch_names, equal_trials)
            loader_train, loader_test, loader_valid = loaders

            model = get_model(n_class, chs, device)

            train_results = train(model, loader_train, loader_test, num_epochs, device, early_stop)
            epoch_losses_train[fold], epoch_losses_valid[fold], best_model, best_epochs_valid[fold] = train_results

            # Load best model state of this fold to Test global accuracy
            if early_stop:
                model.load_state_dict(best_model)
                best_epoch_loss_valid = epoch_losses_valid[fold][best_epochs_valid[fold]]
                print(
                    f"Best Epoch: {best_epochs_valid[fold]} with loss on Validation Data: {best_epoch_loss_valid}")
                # Determine Fold with lowest test_loss on best epoch of fold
                if best_epoch_loss_valid < best_losses_valid.min():
                    best_n_class_models[n_class] = best_model
                    best_fold = fold
                best_losses_valid[fold] = best_epoch_loss_valid

            print("## Testing ##")
            test_accuracy, test_class_hits = test(model, loader_test, device, n_class)
            # Test overfitting by testing on Training Dataset
            if TEST_OVERFITTING:
                print("## Testing on Training Dataset ##")
                accuracies_overfitting[fold], train_class_hits = test(model, loader_train, device, n_class)
            # If not using early stopping, determine which fold has the highest accuracy
            if not early_stop & (test_accuracy >= accuracies.max()):
                best_n_class_models[n_class] = model.state_dict().copy()
                best_fold = fold
            accuracies[fold] = test_accuracy
            fold_class_accuracies = np.zeros(n_class)
            print("Trials for classes:")
            for cl in range(n_class):
                class_trials[cl] = len(test_class_hits[cl])
                fold_class_accuracies[cl] = (100 * (sum(test_class_hits[cl]) / len(test_class_hits[cl])))
            class_accuracies[fold] = fold_class_accuracies
        elapsed = datetime.now() - start
        # Calculate average accuracies per class
        avg_class_accuracies = np.zeros(n_class)
        for cl in range(n_class):
            avg_class_accuracies[cl] = np.average([float(class_accuracies[sp][cl]) for sp in range(folds)])
        res_str = training_result_str(accuracies, accuracies_overfitting, class_trials, avg_class_accuracies, elapsed,
                                      best_epochs_valid, best_losses_valid, best_fold, early_stop=early_stop)
        print(res_str)

        # Store config + results in ./results/{datetime}/training/{n_class}class_results.txt
        save_training_results(n_class, res_str, dir_results, tag)
        save_training_numpy_data(accuracies, class_accuracies, epoch_losses_train, dir_results, n_class)
        # Plot Statistics and save as .png s
        plot_training_statistics(dir_results, tag, n_class, accuracies, avg_class_accuracies, epoch_losses_train,
                                 epoch_losses_valid, best_fold, batch_size, folds, early_stop)
        # Save best trained Model state
        # if early_stop = True: Model state of epoch with the lowest test_loss during Training on small Test Set
        # else: Model state after epochs of Fold with the highest accuracy on Training Set
        if save_model:
            torch.save(best_n_class_models[n_class], f"{dir_results}/{n_class}class_{trained_model_name}")


# Benchmarks pretrained EEGNet (option to use TensorRT optimizations available)
# with Physionet Dataset
# Returns Batch Latency + Time per EEG Trial inference
# saves results in model_path/benchmark
def physionet_benchmark(model_path, name=None, batch_size=BATCH_SIZE, n_classes=N_CLASSES, device=torch.device("cpu"),
                        warm_ups=GPU_WARMUPS,
                        subjects_cs=len(ALL_SUBJECTS), tensorRT=False, iters=1, fp16=False, tag=None,
                        ch_names=MNE_CHANNELS, equal_trials=True, continuous=False, ):
    config = DotDict(batch_size=batch_size, device=device.type, n_classes=n_classes, subjects_cs=subjects_cs,
                     trt=tensorRT, iters=iters, fp16=fp16, ch_names=ch_names)
    chs = len(ch_names)
    # Dont print MNE loading logs
    mne.set_log_level('WARNING')

    print(benchmark_config_str(config))
    dir_results = create_results_folders(path=f"{model_path}", name=name, type='benchmark')

    class_models = {}
    batch_lat_avgs, trial_inf_time_avgs = np.zeros((len(n_classes))), np.zeros((len(n_classes)))
    for class_idx, n_class in enumerate(n_classes):
        print(f"######### {n_class}Class-Classification Benchmarking")
        model_path = f"{model_path}{training_results_folder}/{n_class}class_{trained_model_name}"
        print(f"Loading pretrained model from '{model_path}'")
        class_models[n_class] = get_model(n_class, chs, device, model_path)
        class_models[n_class].eval()
        # Get optimized model with TensorRT
        if tensorRT:
            t = torch.randn((batch_size, 1, chs, SAMPLES)).to(device)
            # add_constant() TypeError: https://github.com/NVIDIA-AI-IOT/torch2trt/issues/440
            # TensorRT either with fp16 ("half") or fp32
            if fp16:
                t = t.half()
                class_models[n_class] = class_models[n_class].half()
            class_models[n_class] = torch2trt(class_models[n_class], [t], max_batch_size=batch_size, fp16_mode=fp16)
            print(f"Optimized EEGNet model with TensorRT (fp{'16' if fp16 else '32'})")

        # Split ALL_SUBJECTS into chunks according to Subjects Chunk Size Parameter (due to high memory usage)
        preload_chunks = split_list_into_chunks(ALL_SUBJECTS, subjects_cs)
        chunks = len(preload_chunks)
        accuracies = np.zeros((chunks * iters) if not continuous else (iters))
        batch_lats = np.zeros((chunks * iters) if not continuous else (iters))
        trial_inf_times = np.zeros((chunks * iters) if not continuous else (iters))

        preloaded_data, preloaded_labels = None, None
        start = datetime.now()

        # Preloads 1 chunk of Subjects and executes 1 gpu_warmup
        if continuous:
            subjects_chunk = preload_chunks[0]
            print(f"Preloading Subjects [{subjects_chunk[0]}-{subjects_chunk[-1]}] Data in memory")
            preloaded_data, preloaded_labels = load_subjects_data(subjects_chunk, n_class, ch_names,
                                                                  equal_trials=equal_trials)
            loader_data = create_loader_from_subjects(subjects_chunk, n_class, device, preloaded_data,
                                                      preloaded_labels, batch_size, ch_names, equal_trials)
            # Warm up GPU with random data
            if device.type != 'cpu':
                gpu_warmup(device, warm_ups, class_models[n_class], batch_size, chs, fp16)
        # Infer multiple times
        for i in range(iters):
            if not continuous:
                # Benchmarking is executed per subject chunk over all Subjects
                # Infers over 1 subject chunks, loads next subject chunk + gpu_warmup, ...
                for ch_idx, subjects_chunk in enumerate(preload_chunks):
                    if DATA_PRELOAD:
                        print(f"Preloading Subjects [{subjects_chunk[0]}-{subjects_chunk[-1]}] Data in memory")
                        preloaded_data, preloaded_labels = load_subjects_data(subjects_chunk, n_class, ch_names,
                                                                              equal_trials=equal_trials)
                    loader_data = create_loader_from_subjects(subjects_chunk, n_class, device, preloaded_data,
                                                              preloaded_labels, batch_size, equal_trials=equal_trials)
                    # Warm up GPU with random data
                    if device.type != 'cpu':
                        gpu_warmup(device, warm_ups, class_models[n_class], batch_size, chs, fp16)
                    idx = chunks * i + ch_idx
                    bench_results = benchmark(class_models[n_class], loader_data, device, fp16)
                    batch_lats[idx], trial_inf_times[idx], accuracies[idx] = bench_results
            else:
                # Benchmarking is executed continuously only over 1 subject chunk
                # Infers over the same subject chunk i times without loading in between
                bench_results = benchmark(class_models[n_class], loader_data, device, fp16)
                batch_lats[i], trial_inf_times[i], accuracies[i] = bench_results
        elapsed = datetime.now() - start
        acc_avg = np.average(accuracies)
        batch_lat_avgs[class_idx] = np.average(batch_lats)
        trial_inf_time_avgs[class_idx] = np.average(trial_inf_times)
        # Print and store Benchmark Config + Results in /results/benchmark/{DateTime}
        res_str = benchmark_result_str(config, n_class, batch_lat_avgs[class_idx], trial_inf_time_avgs[class_idx],
                                       acc_avg, elapsed)
        print(res_str)
        save_benchmark_results(benchmark_config_str(config), n_class, res_str, class_models[n_class],
                               dir_results, tag=tag)
    return batch_lat_avgs, trial_inf_time_avgs


def physionet_live_sim(model_path, subject=1, name=None, ch_names=MNE_CHANNELS,
                       n_classes=N_CLASSES,
                       device=torch.device("cpu"), tag=None, equal_trials=True):
    config = DotDict(subject=subject, device=device.type, n_classes=n_classes, ch_names=ch_names)
    chs = len(ch_names)
    # Dont print MNE loading logs
    mne.set_log_level('WARNING')

    print(live_sim_config_str(config))
    dir_results = create_results_folders(path=f"{model_path}", name=name, type='live_sim')

    class_models = {}
    for class_idx, n_class in enumerate(n_classes):
        print(f"######### {n_class}Class-Classification Live Simulation")
        model_path = f"{model_path}{training_results_folder}/{n_class}class_{trained_model_name}"
        print(f"Loading pretrained model from '{model_path}'")
        class_models[n_class] = get_model(n_class, chs, device, model_path)
        class_models[n_class].eval()

        raw = mne_load_subject_raw(subject, n_classes_live_run[n_class], fmin=global_config.FREQ_FILTER_HIGHPASS,
                                   fmax=global_config.FREQ_FILTER_LOWPASS,notch=global_config.USE_NOTCH_FILTER,
                                   ch_names=ch_names)
        max_sample = raw.n_times
        # X, times, annot = crop_time_and_label(raw, 8)
        X = get_data_from_raw(raw)
        last_label = None
        for now_sample in range(max_sample):
            if now_sample < SAMPLES:
                continue
            # get_label_at_idx( times, annot, 10)
            label, now_time = get_label_at_idx(raw.times, raw.annotations, now_sample)
            pred = predict(class_models[n_class], X[:, (now_sample - SAMPLES):now_sample])
            #if last_label != label:
            print(f"Label from {now_time} is: {label}, pred: {pred}")
            last_label = label

    return


def gpu_warmup(device, warm_ups, model, batch_size, chs, fp16):
    print("Warming up GPU")
    for u in range(warm_ups):
        with torch.no_grad():
            data = torch.randn((batch_size, 1, chs, SAMPLES)).to(device)
            y = model(data.half() if fp16 else data)


# Plots Losses, Accuracies of Training, Validation, Testing
def plot_training_statistics(dir_results, tag, n_class, accuracies, avg_class_accuracies, epoch_losses_train,
                             epoch_losses_valid,
                             best_fold, batch_size, folds, early_stop):
    matplot(accuracies, f"{n_class}class Cross Validation", "Folds Iteration", "Accuracy in %",
            save_path=dir_results,
            bar_plot=True, max_y=100.0)
    matplot(avg_class_accuracies, f"{n_class}class Accuracies{'' if tag is None else tag}", "Class",
            "Accuracy in %",
            save_path=dir_results,
            bar_plot=True, max_y=100.0)
    matplot(epoch_losses_train, f"{n_class}class Training Losses{'' if tag is None else tag}", 'Epoch',
            f'loss per batch (size = {batch_size})',
            labels=[f"Run {i}" for i in range(folds)], save_path=dir_results)
    # Plot Test loss during Training if early stopping is used
    matplot(epoch_losses_valid,
            f"{n_class}class {'Validation' if early_stop else 'Training'} Losses{'' if tag is None else tag}", 'Epoch',
            f'loss per batch (size = {batch_size})',
            labels=[f"Run {i}" for i in range(folds)], save_path=dir_results)
    train_valid_data = np.zeros((2, epoch_losses_train.shape[1]))
    train_valid_data[0] = epoch_losses_train[best_fold]
    train_valid_data[1] = epoch_losses_valid[best_fold]
    matplot(train_valid_data,
            f"{n_class}class Train-{'Valid' if early_stop else 'Test'} Losses of best Fold", 'Epoch',
            f'loss per batch (size = {batch_size})',
            labels=['Training Loss', f'{"Validation" if early_stop else "Testing"} Loss'], save_path=dir_results)


def get_model(n_class, chs, device, state_path=None):
    model = EEGNet(N=n_class, T=SAMPLES, C=chs)
    if state_path is not None:
        model.load_state_dict(torch.load(state_path))
    model.to(device)
    return model
