"""
All available Machine Learning modes implemented with PyTorch
* Training and Testing of EEGNet on PhysioNet Dataset with Cross Validation
* Subject-specific Training with Transfer Learning
* Performance Benchmarking of Inference on pretrained EEGNet
* Live Simulation of real time Classification of a PhysioNet Dataset Run

Author:
  Originally developed by Philipp Kessler as part of his bachelor theses

History:
  2021-05-10: cv_training() changed so that BCIC dataset now can be
              used too - ms (Manfred Strahnen)
"""
import os
from datetime import datetime

import numpy as np
import torch  # noqa
from sklearn.model_selection import GroupKFold

from config import BATCH_SIZE, LR, SPLITS, N_CLASSES, EPOCHS, DATA_PRELOAD, TEST_OVERFITTING, GPU_WARMUPS, \
    trained_model_name, VALIDATION_SUBJECTS, eeg_config
from data.data_loading import PHYS_ALL_SUBJECTS, mne_load_subject_raw, \
    create_preloaded_loader, create_n_class_loaders_from_subject
from data.data_utils import map_trial_labels_to_classes, get_data_from_raw, map_times_to_samples
from data.datasets_confs import NAME, AVAILABLE_SUBJECTS, LOAD_SUBJECTS, LOADERS_FROM_SPLITS, FOLDS, DS_DICTS
from data.physionet_dataset import PHYS_CHANNELS, n_classes_live_run, PHYS_short_name
from machine_learning.configs_results import training_config_str, create_results_folders, save_training_results, \
    benchmark_config_str, get_excluded_if_present, load_global_conf_from_results, load_npz, get_results_file, \
    save_benchmark_results, save_training_numpy_data, benchmark_result_str, save_config, \
    training_result_str, live_sim_config_str, training_ss_config_str, training_ss_result_str, save_live_sim_results, \
    live_sim_result_str
from machine_learning.inference_training import do_train, do_test, do_benchmark, do_predict_on_samples
from machine_learning.util import get_class_accuracies, get_trials_per_class, get_tensorrt_model, gpu_warmup, get_model
from util.dot_dict import DotDict
from util.misc import split_list_into_chunks, groups_labels, get_class_avgs
from util.plot import plot_training_statistics, matplot, create_plot_vspans, create_vlines_from_trials_epochs


# Runs Training + Testing
# Cross Validation
# Can run 2/3/4-Class Classifications
# Saves + Plots Accuracies + Epoch Losses in ./results/{DateTime/name}/training
# save_model: Saves trained model with highest accuracy in results folder
# mi_ds: Used Dataset as String
# only_fold: Specify single Fold to be trained on if only 1 Fold should be trained on
# return n_class Accuracies + Overfittings
def training_cv(num_epochs=EPOCHS, batch_size=BATCH_SIZE, folds=SPLITS, lr=LR, n_classes=N_CLASSES,
                save_model=True, device=torch.device("cpu"), name=None, tag=None, ch_names=PHYS_CHANNELS,
                equal_trials=True, early_stop=False, excluded=[], mi_ds=PHYS_short_name, only_fold=None):
    ds_dict = DS_DICTS[mi_ds]
    folds = ds_dict[FOLDS]

    config = DotDict(num_epochs=num_epochs, batch_size=batch_size, folds=folds, lr=lr, device=device,
                     n_classes=n_classes, ch_names=ch_names, early_stop=early_stop, excluded=excluded,
                     mi_ds=mi_ds,only_fold=only_fold)

    if only_fold is None:
        print(f"Cross validation training with '{ds_dict[NAME]}' started!")
    else:
        print(f"Training of Fold {only_fold} with '{ds_dict[NAME]}' started!")

    start = datetime.now()
    print(training_config_str(config))
    if name is None:
        dir_results = create_results_folders(datetime=start)
    else:
        dir_results = create_results_folders(path=name)

    if len(excluded) > 0:
        if tag is None:
            tag = 'excluded'
        else:
            tag += '_excluded'
    save_config(training_config_str(config), ch_names, dir_results, tag)

    available_subjects = [i for i in ds_dict[AVAILABLE_SUBJECTS] if i not in excluded]

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
    groups = groups_labels(len(used_subjects), folds)

    # Split Data into training + test
    cv = GroupKFold(n_splits=folds)

    best_n_class_models = {}
    n_class_accuracy, n_class_overfitting_diff = np.zeros(len(n_classes)), np.zeros(
        len(n_classes))

    # Only Train specified fold
    if only_fold is not None:
        folds = 1

    for i, n_class in enumerate(n_classes):
        preloaded_data, preloaded_labels = None, None

        if DATA_PRELOAD:
            print("PRELOADING ALL DATA IN MEMORY")
            preloaded_data, preloaded_labels = ds_dict[LOAD_SUBJECTS](used_subjects + validation_subjects, n_class,
                                                                      ch_names, equal_trials, normalize=False)

        cv_split = cv.split(X=used_subjects, groups=groups)
        start = datetime.now()
        print(f"######### {n_class}Class-Classification")
        fold_accuracies = np.zeros((folds))
        best_losses_test, best_epochs_valid = np.full((folds), fill_value=np.inf), np.zeros((folds), dtype=np.int)
        best_fold_act_labels, best_fold_pred_labels = None, None
        best_fold = -1
        class_accuracies, class_trials = np.zeros((folds, n_class)), np.zeros(n_class)
        accuracies_overfitting = np.zeros((folds)) if TEST_OVERFITTING else None
        epoch_losses_train, epoch_losses_test = np.zeros((folds, num_epochs)), np.zeros((folds, num_epochs))

        # Skip folds until specified fold is reached
        if only_fold is not None:
            for f in range(only_fold):
                next(cv_split)

        # Training of the Folds with the different splits
        for fold in range(folds):
            print(f"############ Fold {fold + 1} ############")
            # Next Splits Combination of Train/Test Datasets + Validation Set Loader
            loaders = ds_dict[LOADERS_FROM_SPLITS](next(cv_split), validation_subjects, n_class, device,
                                                   preloaded_data,
                                                   preloaded_labels, batch_size, ch_names, equal_trials,
                                                   used_subjects=used_subjects)

            loader_train, loader_test, loader_valid = loaders

            model = get_model(n_class, len(ch_names), device)

            train_results = do_train(model, loader_train, loader_test, num_epochs, device, early_stop)
            epoch_losses_train[fold], epoch_losses_test[fold], best_model, best_epochs_valid[fold] = train_results
            # Load best model state of this fold to Test global accuracy
            if early_stop:
                model.load_state_dict(best_model)
                best_epoch_loss_test = epoch_losses_test[fold][best_epochs_valid[fold]]
                print(
                    f"Best Epoch: {best_epochs_valid[fold]} with loss on Validation Data: {best_epoch_loss_test}")
                # Determine Fold with lowest test_loss on best epoch of fold
                if best_epoch_loss_test < np.min(best_losses_test):
                    best_n_class_models[n_class] = best_model
                    best_fold = fold
                best_losses_test[fold] = best_epoch_loss_test

            print("## Testing ##")
            test_accuracy, act_labels, pred_labels = do_test(model, loader_test)
            # Test overfitting by testing on Training Dataset
            if TEST_OVERFITTING:
                print("## Testing on Training Dataset ##")
                accuracies_overfitting[fold], _, __ = do_test(model, loader_train)
            # If not using early stopping, determine which fold has the highest accuracy
            if (not early_stop) & (test_accuracy > np.max(fold_accuracies)):
                best_n_class_models[n_class] = model.state_dict().copy()
                best_fold = fold
                best_fold_act_labels = act_labels
                best_fold_pred_labels = pred_labels
            fold_accuracies[fold] = test_accuracy
            class_trials, class_accuracies[fold] = get_trials_per_class(n_class, act_labels), \
                                                   get_class_accuracies(act_labels, pred_labels)
        elapsed = datetime.now() - start
        avg_class_accuracies = get_class_avgs(n_class, class_accuracies)
        res_str = training_result_str(fold_accuracies, accuracies_overfitting, class_trials, avg_class_accuracies,
                                      elapsed, best_epochs_valid, best_losses_test, best_fold,
                                      (best_fold_act_labels, best_fold_pred_labels,only_fold),
                                      early_stop=early_stop)
        print(res_str)

        # Store config + results in ./results/{datetime}/training/{n_class}class_results.txt
        save_training_results(n_class, res_str, dir_results, tag)
        save_training_numpy_data(fold_accuracies, class_accuracies, epoch_losses_train, epoch_losses_test, dir_results,
                                 n_class, excluded, mi_ds, labels=(best_fold_act_labels, best_fold_pred_labels))
        # Plot Statistics and save as .png s
        plot_training_statistics(dir_results, tag, n_class, fold_accuracies, avg_class_accuracies, epoch_losses_train,
                                 epoch_losses_test, best_fold, batch_size, folds, early_stop)
        # Save best trained Model state
        # if early_stop = True: Model state of epoch with the lowest test_loss during Training on small Test Set
        # else: Model state after epochs of Fold with the highest accuracy on Training Set
        if save_model:
            torch.save(best_n_class_models[n_class], os.path.join(dir_results, f"{n_class}class_{trained_model_name}"))

        n_class_accuracy[i] = np.average(fold_accuracies)
        n_class_overfitting_diff[i] = n_class_accuracy[i] - np.average(accuracies_overfitting)
    return n_class_accuracy, n_class_overfitting_diff


# Runs Subject-specific Training on pretrained model (model_path)
# Supposed to be used before live_sim mode is executed
# Saves further trained model
def training_ss(model_path, subject=None, num_epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR, n_classes=[3],
                device=torch.device("cpu"), tag=None, ch_names=PHYS_CHANNELS):
    n_test_runs = 1
    config = DotDict(subject=subject, num_epochs=num_epochs, batch_size=batch_size, lr=lr, device=device,
                     n_classes=n_classes, ch_names=ch_names, n_test_runs=n_test_runs)

    start = datetime.now()
    print(training_ss_config_str(config))

    for i, n_class in enumerate(n_classes):
        test_accuracy, test_class_hits = np.zeros(1), []
        n_class_results = load_npz(get_results_file(model_path, n_class))
        mi_ds = n_class_results['mi_ds']
        # TODO use mi_ds
        load_global_conf_from_results(n_class_results)
        used_subject = get_excluded_if_present(n_class_results, subject)

        dir_results = create_results_folders(path=model_path, name=f"S{used_subject:03d}", mode='train_ss')
        save_config(training_ss_config_str(config), ch_names, dir_results, tag)
        print(f"Loading pretrained model from '{model_path}'")

        model = get_model(n_class, len(ch_names), device, model_path)
        # Split subjects' data into Training Set (2 of 3 Runs) + Test Set (1 remaining Run)
        loader_train, loader_test = create_n_class_loaders_from_subject(used_subject, n_class, n_test_runs, batch_size,
                                                                        ch_names, device)

        epoch_losses_train, epoch_losses_test, _, __ = do_train(model, loader_train, loader_test,
                                                                num_epochs, device)
        test_accuracy[0], act_labels, pred_labels = do_test(model, loader_test)

        elapsed = datetime.now() - start
        class_trials, class_accuracies = get_trials_per_class(n_class, act_labels), \
                                         get_class_accuracies(act_labels, pred_labels)

        res_str = training_ss_result_str(test_accuracy[0], class_trials, class_accuracies, elapsed)
        print(res_str)
        save_training_results(n_class, res_str, dir_results, tag)
        save_training_numpy_data(test_accuracy, class_accuracies,
                                 epoch_losses_train, epoch_losses_test,
                                 dir_results, n_class, [used_subject], None)

        torch.save(model.state_dict(), f"{dir_results}/{n_class}class_{trained_model_name}")


# Benchmarks pretrained EEGNet (option to use TensorRT optimizations available)
# with Physionet Dataset
# Returns Batch Latency + Time per EEG Trial inference
# saves results in model_path/benchmark
def benchmarking(model_path, name=None, batch_size=BATCH_SIZE, n_classes=[2], device=torch.device("cpu"),
                 warm_ups=GPU_WARMUPS,
                 subjects_cs=len(PHYS_ALL_SUBJECTS), tensorRT=False, iters=1, fp16=False, tag=None,
                 ch_names=PHYS_CHANNELS, equal_trials=True, continuous=False):
    config = DotDict(batch_size=batch_size, device=device.type, n_classes=n_classes, subjects_cs=subjects_cs,
                     trt=tensorRT, iters=iters, fp16=fp16, ch_names=ch_names)
    chs = len(ch_names)

    print(benchmark_config_str(config))
    dir_results = create_results_folders(path=f"{model_path}", name=name, mode='benchmark')

    class_models = {}
    batch_lat_avgs, trial_inf_time_avgs = np.zeros((len(n_classes))), np.zeros((len(n_classes)))
    for class_idx, n_class in enumerate(n_classes):
        print(f"######### {n_class}Class-Classification Benchmarking")
        n_class_results = load_npz(get_results_file(model_path, n_class))
        load_global_conf_from_results(n_class_results)

        print(f"Loading pretrained model from '{model_path} ({n_class}class)'")
        class_models[n_class] = get_model(n_class, chs, device, model_path)
        class_models[n_class].eval()
        # Get optimized model with TensorRT
        if tensorRT:
            class_models[n_class] = get_tensorrt_model(class_models[n_class], batch_size, chs, device, fp16)

        # Split ALL_SUBJECTS into chunks according to Subjects Chunk Size Parameter (due to high memory usage)
        preload_chunks = split_list_into_chunks(PHYS_ALL_SUBJECTS, subjects_cs)
        chunks = len(preload_chunks)
        accuracies = np.zeros((chunks * iters) if not continuous else (iters))
        batch_lats = np.zeros((chunks * iters) if not continuous else (iters))
        trial_inf_times = np.zeros((chunks * iters) if not continuous else (iters))

        preloaded_data, preloaded_labels = None, None
        start = datetime.now()

        # Preloads 1 chunk of Subjects and executes 1 gpu_warmup
        if continuous:
            loader_data = create_preloaded_loader(preload_chunks[0], n_class, ch_names, batch_size, device,
                                                  equal_trials)
            # Warm up GPU with random data
            if device.type != 'cpu':
                gpu_warmup(device, warm_ups, class_models[n_class], batch_size, chs, fp16)
        # Infer multiple times to get an average benchmark
        for i in range(iters):
            if not continuous:
                # Benchmarking is executed per subject chunk over all Subjects
                # Infers over 1 subject chunks, loads next subject chunk + gpu_warmup, ...
                for ch_idx, subjects_chunk in enumerate(preload_chunks):
                    loader_data = create_preloaded_loader(subjects_chunk, n_class, ch_names, batch_size, device,
                                                          equal_trials)
                    # Warm up GPU with random data
                    if device.type != 'cpu':
                        gpu_warmup(device, warm_ups, class_models[n_class], batch_size, chs, fp16)
                    idx = chunks * i + ch_idx
                    benchmark_results = do_benchmark(class_models[n_class], loader_data, device, fp16)
                    batch_lats[idx], trial_inf_times[idx], accuracies[idx] = benchmark_results

            else:
                # Benchmarking is executed continuously only over 1 subject chunk
                # Infers over the same subject chunk i times without loading in between
                benchmark_results = do_benchmark(class_models[n_class], loader_data, device, fp16)
                batch_lats[i], trial_inf_times[i], accuracies[i] = benchmark_results
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


# Simulates Live usage
# Loads pretrained model of model_path
# Loads Example Run of Subject for n_class
# Predicts classes on every available sample
# Plots Prediction values (in percent)
# Stores Prediction array as .npy
def live_sim(model_path, subject=None, name=None, ch_names=PHYS_CHANNELS,
             n_classes=N_CLASSES,
             device=torch.device("cpu"), tag=None):
    dir_results = create_results_folders(path=f"{model_path}", name=name, mode='live_sim')
    for class_idx, n_class in enumerate(n_classes):
        start = datetime.now()
        n_class_results = load_npz(get_results_file(model_path, n_class))
        load_global_conf_from_results(n_class_results)
        used_subject = get_excluded_if_present(n_class_results, subject)

        run = n_classes_live_run[n_class]
        config = DotDict(subject=used_subject, device=device.type,
                         n_classes=n_classes, ch_names=ch_names, run=run)
        print(live_sim_config_str(config))

        print(f"######### {n_class}Class-Classification Live Simulation")
        print(f"Loading pretrained model from '{model_path}'")
        model = get_model(n_class, len(ch_names), device, model_path)
        model.eval()

        # Load Raw Subject Run for n_class
        raw = mne_load_subject_raw(used_subject, n_classes_live_run[n_class], ch_names=ch_names)
        # Get Data from raw Run
        X = get_data_from_raw(raw)

        max_sample = raw.n_times
        slices = eeg_config.TRIALS_SLICES
        # times = raw.times[:max_sample]
        trials_start_times = raw.annotations.onset
        trials_classes = map_trial_labels_to_classes(raw.annotations.description)

        # Get samples of Trials Start Times
        trials_start_samples = map_times_to_samples(raw, trials_start_times)
        # if eeg_config.TRIALS_SLICES is not None:
        #     used_samples = math.floor(used_samples / eeg_config.TRIALS_SLICES)
        sample_predictions = do_predict_on_samples(model, n_class, X, max_sample, device)
        # sample_predictions = sample_predictions * 100
        sample_predictions = np.swapaxes(sample_predictions, 0, 1)

        # Highlight Trials and mark the trained on positions of each Trial
        vspans = create_plot_vspans(trials_start_samples, trials_classes, max_sample)
        tdelta = eeg_config.TMAX - eeg_config.TMIN
        vlines = create_vlines_from_trials_epochs(raw, trials_start_times, tdelta, slices)

        # trials_correct_areas_relative = get_correctly_predicted_areas(n_class, sample_predictions, trials_classes,
        #                                                               trials_start_samples,
        #                                                               max_sample)
        trials_correct_areas_relative = np.zeros((len(trials_classes)))

        for trial, trial_correct_area_relative in enumerate(trials_correct_areas_relative):
            print(f"Trial {trial:02d}: {trial_correct_area_relative:.3f} (Class {trials_classes[trial]})")
        print(f"Average Correct Area per Trial: {np.average(trials_correct_areas_relative):.3f}")

        # matplot(sample_predictions,
        #         f"{n_class}class Live Simulation_S{subject:03d}",
        #         'Time in sec.', f'Prediction in %', fig_size=(80.0, 10.0), max_y=100.5,
        #         vspans=vspans, vlines=vlines, ticks=trials_start_samples, x_values=trials_start_times,
        #         labels=[f"T{i}" for i in range(n_class)], save_path=dir_results)
        np.save(os.path.join(dir_results, f"{n_class}class_predictions"), sample_predictions)
        # Split into multiple plots, otherwise too long
        plot_splits = 5
        trials_split_size = int(trials_start_samples.shape[0] / plot_splits)
        n_class_offset = 0 if n_class > 2 else 1
        for i in range(plot_splits):
            first_trial = i * trials_split_size
            last_trial = (i + 1) * trials_split_size - 1
            first_sample = trials_start_samples[first_trial]
            if i == plot_splits - 1:
                last_sample = max_sample
            else:
                last_sample = trials_start_samples[last_trial + 1]
            matplot(sample_predictions,
                    f"{n_class}class Live Simulation_S{used_subject:03d}_R{run:02d}_{i + 1}",
                    'Time in sec.', f'Prediction Outputs', fig_size=(30.0, 8.0),
                    vspans=vspans[first_trial:last_trial + 1],
                    color_offset=n_class_offset, font_size=30.0,
                    legend_hor=True,
                    vlines=vlines[(first_trial * slices):(last_trial + 1) * slices],
                    vlines_label="Trained timepoints", legend_loc='lower left',
                    ticks=trials_start_samples[first_trial:last_trial + 1],
                    min_x=first_sample, max_x=last_sample,
                    x_values=trials_start_times[first_trial:last_trial + 1],
                    labels=[f"T{i + n_class_offset}" for i in range(n_class)], save_path=dir_results)

        res_str = live_sim_result_str(n_class, trials_correct_areas_relative, datetime.now() - start)
        print(res_str)
        save_live_sim_results(n_class, res_str, dir_results, tag)
    return
