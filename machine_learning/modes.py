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
from typing import List, Type

import numpy as np
import torch  # noqa
from sklearn.model_selection import GroupKFold

from config import TEST_OVERFITTING, CONFIG
from data.MIDataLoader import MIDataLoader
from data.datasets.datasets import DATASETS
from data.datasets.phys.phys_dataset import PHYS
from machine_learning.configs_results import training_config_str, create_results_folders, save_training_results, \
    benchmark_config_str, get_excluded_if_present, load_global_conf_from_results, load_npz, get_results_file, \
    save_benchmark_results, save_training_numpy_data, benchmark_result_str, save_config, \
    training_result_str, live_sim_config_str, training_ss_config_str, training_ss_result_str, save_live_sim_results, \
    live_sim_result_str
from machine_learning.inference_training import do_train, do_test, do_benchmark, do_predict_on_samples
from machine_learning.util import get_class_accuracies, get_trials_per_class, get_tensorrt_model, gpu_warmup, get_model, \
    ML_Run_Data
from paths import trained_model_name
from util.dot_dict import DotDict
from util.misc import split_list_into_chunks, groups_labels
from util.plot import plot_training_statistics, matplot, create_plot_vspans, create_vlines_from_trials_epochs
import torch.types


def training_cv(mi_ds: str, num_epochs: int, batch_size: int, n_classes: List[int],
                name: str, tag: str, ch_names: List[str], equal_trials=True, early_stop=False, excluded=[],
                only_fold=None, save_model=True, lr: DotDict = CONFIG.MI.LR):
    """
    Runs Training + Testing
    Cross Validation
    Can run 2/3/4-Class Classifications
    Saves + Plots Accuracies + Epoch Losses in ./results/{DateTime/name}/training
    :param save_model: Saves trained model with highest accuracy in results folder
    :param mi_ds: sed Dataset as String
    :param only_fold: Specify single Fold to be trained on if only 1 Fold should be trained on
    :return: n_class Accuracies + n_class Overfittings
    """
    dataset = DATASETS[mi_ds]
    folds = dataset.CONSTANTS.cv_folds

    config = DotDict(num_epochs=num_epochs, batch_size=batch_size, folds=folds, lr=lr, device=CONFIG.DEVICE,
                     n_classes=n_classes, ch_names=ch_names, early_stop=early_stop, excluded=excluded,
                     mi_ds=mi_ds, only_fold=only_fold)

    if only_fold is None:
        print(f"Cross validation training with '{dataset.CONSTANTS.name}' started!")
    else:
        print(f"Training of Fold {only_fold} with '{dataset.CONSTANTS.name}' started!")

    start = datetime.now()
    print(training_config_str(config))
    dir_results = create_results_folders(datetime=start) if name is None else create_results_folders(path=name)
    if len(excluded) > 0:
        tag = 'excluded' if tag is None else (tag + '_excluded')

    save_config(training_config_str(config), ch_names, dir_results, tag)

    available_subjects = [i for i in dataset.CONSTANTS.ALL_SUBJECTS if i not in excluded]

    used_subjects = available_subjects
    validation_subjects = []
    # Currently if early_stop=true:
    # Validation Set = Test Set
    # 0 validation subjects, train() evaluates valid_loss on loader_test
    if early_stop & (CONFIG.MI.VALIDATION_SUBJECTS > 0):
        # 76 Subjects (~72%) for Train + 19 (~18%) for Test (get split in 5 different Splits)
        used_subjects = available_subjects[:(len(available_subjects) - CONFIG.MI.VALIDATION_SUBJECTS)]
        # 10 (~10%) Subjects for Validation (always the same)
        validation_subjects = available_subjects[(len(available_subjects) - CONFIG.MI.VALIDATION_SUBJECTS):]
        print(f"Validation Subjects: [{validation_subjects[0]}-{validation_subjects[-1]}]")

    # Group labels (subjects in same group need same group label)
    groups = groups_labels(len(used_subjects), folds)

    # Split Data into training + util
    cv = GroupKFold(n_splits=folds)

    best_n_class_models = {}
    n_class_accuracy, n_class_overfitting_diff = np.zeros(len(n_classes)), np.zeros(
        len(n_classes))

    # Only Train specified fold
    if only_fold is not None:
        folds = 1

    for i, n_class in enumerate(n_classes):
        print("PRELOADING ALL DATA IN MEMORY")
        preloaded_data, preloaded_labels = dataset.load_subjects_data(used_subjects + validation_subjects, n_class,
                                                                      ch_names, equal_trials)
        run_data, best_model = do_n_class_training_cv(cv, used_subjects, groups, folds, n_class, num_epochs, only_fold,
                                                      dataset, validation_subjects,
                                                      preloaded_data, preloaded_labels, batch_size, ch_names,
                                                      equal_trials=True,
                                                      early_stop=False)
        n_class_accuracy[i], n_class_overfitting_diff[i] = save_n_class_results(n_class, mi_ds, run_data, folds,
                                                                                only_fold, batch_size, excluded,
                                                                                best_model, dir_results, tag,
                                                                                save_model)
    return n_class_accuracy, n_class_overfitting_diff


def save_n_class_results(n_class: int, mi_ds: str, run_data: ML_Run_Data, folds: int, only_fold: int, batch_size: int,
                         excluded: List[int], best_model, dir_results: str, tag: str = None, early_stop=False,
                         save_model=True):
    """
    Save n-class Cross Validation Results from run_data
    :return: n_class_accuracy, n_class_overfitting_diff
    """
    res_str = training_result_str(run_data, only_fold,
                                  early_stop=early_stop)
    print(res_str)

    # Store config + results in ./results/{datetime}/training/{n_class}class_results.txt
    save_training_results(n_class, res_str, dir_results, tag)
    save_training_numpy_data(run_data, dir_results, n_class, excluded, mi_ds)
    # Plot Statistics and save as .png s
    plot_training_statistics(dir_results, tag, run_data, batch_size, folds, early_stop)
    # Save best trained Model state
    # if early_stop = True: Model state of epoch with the lowest test_loss during Training on small Test Set
    # else: Model state after epochs of Fold with the highest accuracy on Training Set
    if save_model:
        torch.save(best_model, os.path.join(dir_results, f"{n_class}class_{trained_model_name}"))

    n_class_accuracy = np.average(run_data.fold_accuracies)
    n_class_overfitting_diff = n_class_accuracy - np.average(run_data.accuracies_overfitting)
    return n_class_accuracy, n_class_overfitting_diff


def do_n_class_training_cv(cv: GroupKFold, used_subjects: List[int], groups: np.ndarray, folds: int, n_class: int,
                           num_epochs: int, only_fold: int, dataset: Type[MIDataLoader], validation_subjects: List[int],
                           preloaded_data: np.ndarray, preloaded_labels: np.ndarray, batch_size: int,
                           ch_names: List[str], equal_trials=True, early_stop=False):
    """
    Executes n-class Cross Validation Training based on given parameters + preloaded Data
    :return: ML_Run_Data, best Fold Model state dict
    """
    cv_split = cv.split(X=used_subjects, groups=groups)
    best_model = None
    run_data = ML_Run_Data(folds, n_class, num_epochs, cv_split)
    run_data.start_run()

    # Skip folds until specified fold is reached
    if only_fold is not None:
        for f in range(only_fold):
            next(cv_split)

    print(f"######### {n_class}Class-Classification")
    # Training of the Folds with the different splits
    for fold in range(folds):
        print(f"############ Fold {fold + 1} ############")
        # Next Splits Combination of Train/Test Datasets + Validation Set Loader
        loaders = dataset.create_loaders_from_splits(next(cv_split), validation_subjects, n_class,
                                                     preloaded_data, preloaded_labels, batch_size, ch_names,
                                                     equal_trials, used_subjects=used_subjects)
        loader_train, loader_test, loader_valid = loaders

        model = get_model(n_class, len(ch_names))

        train_results = do_train(model, loader_train, loader_test, num_epochs, CONFIG.DEVICE, early_stop)
        run_data.set_train_results(fold, train_results)
        # Load best model state of this fold to Test global accuracy if using Early Stopping
        if early_stop:
            model.load_state_dict(run_data.best_model[fold])
            best_epoch_loss_test = run_data.best_epoch_loss_test(fold)
            print(f"""Best Epoch: {run_data.best_epochs_test[fold]} with 
            loss on Validation Data: {best_epoch_loss_test}""")
            # Determine Fold with lowest test_loss on best epoch of fold
            if best_epoch_loss_test < np.min(run_data.best_losses_test):
                best_model = run_data.best_model[fold]
                run_data.set_best_fold(fold)
            run_data.best_losses_test[fold] = best_epoch_loss_test

        print("## Testing ##")
        test_accuracy, act_labels, pred_labels = do_test(model, loader_test)
        # Test overfitting by testing on Training Dataset
        if TEST_OVERFITTING:
            print("## Testing on Training Dataset ##")
            run_data.accuracies_overfitting[fold], _, __ = do_test(model, loader_train)

        # If not using early stopping, determine which fold has the highest accuracy
        if (not early_stop) & (test_accuracy > np.max(run_data.fold_accuracies)):
            best_model = model.state_dict().copy()
            run_data.set_best_fold(fold, act_labels, pred_labels)
        run_data.set_test_results(fold, test_accuracy, act_labels, pred_labels)
        if only_fold is not None:
            run_data.set_best_fold(only_fold)
        run_data.end_run()
    return run_data, best_model


def testing(n_class, model_path, ch_names):
    """
    Test pretrained model (Best Fold)
    Determines Accuracy on Best-Fold's Test Set
    :param model_path: Path to trained_model.pt File (Folder)
    :return: Accuracy on Test Set
    """
    n_class_results = load_npz(get_results_file(model_path, n_class))
    ds_short_name = n_class_results['mi_ds'].item()
    dataset = DATASETS[ds_short_name]
    # Get Best Fold Nr. of trained model
    # best_fold = np.argmax(n_class_results['test_accs']).item()
    best_fold = n_class_results['best_fold'].item()

    print(f"Testing '{model_path}' {n_class}-classification of Best Fold ({best_fold + 1})")
    print(f"TMIN: {n_class_results['tmin']}, TMAX: {n_class_results['tmax']}"
          f" FMIN: {CONFIG.FILTER.FREQ_FILTER_HIGHPASS},  FMAX: {CONFIG.FILTER.FREQ_FILTER_LOWPASS}")

    # Group labels (subjects in same group need same group label)
    groups = groups_labels(len(dataset.CONSTANTS.ALL_SUBJECTS), dataset.CONSTANTS.cv_folds)

    # Split Data into training + util
    cv = GroupKFold(n_splits=dataset.CONSTANTS.cv_folds)
    cv_split = cv.split(X=dataset.CONSTANTS.ALL_SUBJECTS, groups=groups)
    # Skip to best fold
    for f in range(best_fold):
        next(cv_split)

    # Get Test Subjects of Best Fold
    test_subjects_idxs = next(cv_split)[1].tolist()
    subjects_test = [dataset.CONSTANTS.ALL_SUBJECTS[idx] for idx in test_subjects_idxs]

    model = get_model(n_class, len(ch_names), model_path)

    print("PRELOADING ALL DATA IN MEMORY")
    preloaded_data, preloaded_labels = dataset.load_subjects_data(subjects_test, n_class,
                                                                  ch_names)

    # Test with Best-Fold Test set subjects
    loader_test = dataset.create_loader_from_subjects(subjects_test, n_class,
                                                      preloaded_data, preloaded_labels,
                                                      CONFIG.MI.BATCH_SIZE, dataset.CONSTANTS.CHANNELS)
    test_accuracy, act_labels, pred_labels = do_test(model, loader_test)
    print(f"Test Accuracy: {test_accuracy}")
    return test_accuracy


def training_ss(model_path, subject=None, num_epochs=CONFIG.MI.EPOCHS, batch_size=CONFIG.MI.BATCH_SIZE,
                lr=CONFIG.MI.LR, n_classes=[3], tag=None, ch_names=PHYS.CHANNELS):
    """
    Runs Subject-specific Training on pretrained model (model_path)
    Supposed to be used before live_sim mode is executed
    Saves further trained model
    :param model_path: Path to trained_model.pt File (/training Folder of training_cv())
    :param subject: Subject to train specifically
    """
    n_test_runs = 1
    config = DotDict(subject=subject, num_epochs=num_epochs, batch_size=batch_size, lr=lr, device=CONFIG.DEVICE,
                     n_classes=n_classes, ch_names=ch_names, n_test_runs=n_test_runs)

    start = datetime.now()
    print(training_ss_config_str(config))

    for i, n_class in enumerate(n_classes):
        test_accuracy, test_class_hits = np.zeros(1), []
        n_class_results = load_npz(get_results_file(model_path, n_class))
        mi_ds = n_class_results['mi_ds'].item()
        dataset = DATASETS[mi_ds]
        CONFIG.set_eeg_config(dataset.CONSTANTS.CONFIG)
        load_global_conf_from_results(n_class_results, dataset.CONSTANTS.CONFIG.CUE_OFFSET)
        used_subject = get_excluded_if_present(n_class_results, subject)

        dir_results = create_results_folders(path=model_path, name=f"S{used_subject:03d}", mode='train_ss')
        save_config(training_ss_config_str(config), ch_names, dir_results, tag)
        print(f"Loading pretrained model from '{model_path}'")

        model = get_model(n_class, len(ch_names), model_path)
        # Split subjects' data into Training Set (2 of 3 Runs) + Test Set (1 remaining Run)
        loader_train, loader_test = dataset.create_n_class_loaders_from_subject(used_subject, n_class, n_test_runs,
                                                                                batch_size,
                                                                                ch_names)
        run_data = ML_Run_Data(1, n_class, num_epochs, None)
        run_data.start_run()
        train_results = do_train(model, loader_train, loader_test, num_epochs, CONFIG.DEVICE)
        epoch_losses_train, epoch_losses_test, _, __ = train_results
        run_data.set_train_results(0, train_results)

        test_accuracy[0], act_labels, pred_labels = do_test(model, loader_test)
        run_data.set_test_results(0, test_accuracy, act_labels, pred_labels)
        elapsed = datetime.now() - start
        class_trials, class_accuracies = get_trials_per_class(n_class, act_labels), \
                                         get_class_accuracies(act_labels, pred_labels)
        run_data.end_run()
        res_str = training_ss_result_str(test_accuracy[0], class_trials, class_accuracies, elapsed)
        print(res_str)
        save_training_results(n_class, res_str, dir_results, tag)
        save_training_numpy_data(run_data, dir_results, n_class, [used_subject], mi_ds)

        torch.save(model.state_dict(), f"{dir_results}/{n_class}class_{trained_model_name}")


def benchmarking(model_path, name=None, batch_size=CONFIG.MI.BATCH_SIZE, n_classes=[2], warm_ups=CONFIG.MI.GPU_WARMUPS,
                 subjects_cs=len(PHYS.ALL_SUBJECTS), tensorRT=False, iters=1, fp16=False, tag=None,
                 ch_names=PHYS.CHANNELS, equal_trials=True, continuous=False):
    """
    Benchmarks pretrained EEGNet (option to use TensorRT optimizations available)
    with Physionet Dataset
    Returns Batch Latency + Time per EEG Trial inference
    saves results in model_path/benchmark
    :param model_path: Path to trained_model.pt
    :param warm_ups: Amount of GPU Warm ups before Benchmarking
    :param subjects_cs: Subject Chunk size
    :param tensorRT: Enable TensorRT
    :param iters: Amount of iterations to average Performance
    :param continuous: Benchmark on same Subject Chunk continuously
    :return: Batch Latencies Averages + Trial Inference Times Averages
    """
    config = DotDict(batch_size=batch_size, device=CONFIG.DEVICE, n_classes=n_classes, subjects_cs=subjects_cs,
                     trt=tensorRT, iters=iters, fp16=fp16, ch_names=ch_names)
    chs = len(ch_names)

    print(benchmark_config_str(config))
    dir_results = create_results_folders(path=f"{model_path}", name=name, mode='benchmark')

    class_models = {}
    batch_lat_avgs, trial_inf_time_avgs = np.zeros((len(n_classes))), np.zeros((len(n_classes)))
    for class_idx, n_class in enumerate(n_classes):
        print(f"######### {n_class}Class-Classification Benchmarking")
        n_class_results = load_npz(get_results_file(model_path, n_class))
        dataset = DATASETS[n_class_results['mi_ds']]
        CONFIG.set_eeg_config(dataset.CONSTANTS.CONFIG)
        load_global_conf_from_results(n_class_results, CONFIG.EEG.CUE_OFFSET)

        print(f"Loading pretrained model from '{model_path} ({n_class}class)'")
        class_models[n_class] = get_model(n_class, chs, model_path)
        class_models[n_class].eval()
        # Get optimized model with TensorRT
        if tensorRT:
            class_models[n_class] = get_tensorrt_model(class_models[n_class], batch_size, chs, fp16)

        # Split ALL_SUBJECTS into chunks according to Subjects Chunk Size Parameter (due to high memory usage)
        preload_chunks = split_list_into_chunks(PHYS.ALL_SUBJECTS, subjects_cs)
        chunks = len(preload_chunks)
        accuracies = np.zeros((chunks * iters) if not continuous else (iters))
        batch_lats = np.zeros((chunks * iters) if not continuous else (iters))
        trial_inf_times = np.zeros((chunks * iters) if not continuous else (iters))

        start = datetime.now()

        # Preloads 1 chunk of Subjects and executes 1 gpu_warmup
        if continuous:
            loader_data = dataset.create_preloaded_loader(preload_chunks[0], n_class, ch_names, batch_size,
                                                          equal_trials)
            # Warm up GPU with random data
            if CONFIG.DEVICE.type != 'cpu':
                gpu_warmup(CONFIG.DEVICE, warm_ups, class_models[n_class], batch_size, chs, fp16)
        # Infer multiple times to get an average benchmark
        for i in range(iters):
            if not continuous:
                # Benchmarking is executed per subject chunk over all Subjects
                # Infers over 1 subject chunks, loads next subject chunk + gpu_warmup, ...
                for ch_idx, subjects_chunk in enumerate(preload_chunks):
                    loader_data = dataset.create_preloaded_loader(subjects_chunk, n_class, ch_names, batch_size,
                                                                  equal_trials)
                    # Warm up GPU with random data
                    if CONFIG.DEVICE.type != 'cpu':
                        gpu_warmup(CONFIG.DEVICE, warm_ups, class_models[n_class], batch_size, chs, fp16)
                    idx = chunks * i + ch_idx
                    benchmark_results = do_benchmark(class_models[n_class], loader_data, CONFIG.DEVICE, fp16)
                    batch_lats[idx], trial_inf_times[idx], accuracies[idx] = benchmark_results

            else:
                # Benchmarking is executed continuously only over 1 subject chunk
                # Infers over the same subject chunk i times without loading in between
                benchmark_results = do_benchmark(class_models[n_class], loader_data, CONFIG.DEVICE, fp16)
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


def live_sim(model_path, subject=None, name=None, ch_names=PHYS.CHANNELS, n_classes=CONFIG.MI.N_CLASSES, tag=None):
    """
    Simulates Live usage
    Loads pretrained model of model_path
    Loads Example Run of Subject for n_class
    Predicts classes on every available sample
    Plots Prediction values (in percent)
    Stores Prediction array as .npy
    :param model_path: Path to subject-specific trained_model.pt
    :param subject: Subject to perform Live-Simulation on
    """
    dir_results = create_results_folders(path=f"{model_path}", name=name, mode='live_sim')
    for class_idx, n_class in enumerate(n_classes):
        start = datetime.now()
        n_class_results = load_npz(get_results_file(model_path, n_class))
        dataset = DATASETS[n_class_results['mi_ds'].item()]
        CONFIG.set_eeg_config(dataset.CONSTANTS.CONFIG)
        load_global_conf_from_results(n_class_results, dataset.CONSTANTS.CONFIG.CUE_OFFSET)
        used_subject = get_excluded_if_present(n_class_results, subject)

        run = PHYS.n_classes_live_run[n_class]
        config = DotDict(subject=used_subject, device=CONFIG.DEVICE.type,
                         n_classes=n_classes, ch_names=ch_names, run=run)
        print(live_sim_config_str(config))

        print(f"######### {n_class}Class-Classification Live Simulation")
        print(f"Loading pretrained model from '{model_path}'")
        model = get_model(n_class, len(ch_names), model_path)
        model.eval()

        X, max_sample, slices, trials_classes, trials_start_times, trials_start_samples, trial_tdeltas = dataset.load_live_sim_data(
            used_subject, n_class,
            ch_names)

        # if eeg_config.TRIALS_SLICES is not None:
        #     used_samples = math.floor(used_samples / eeg_config.TRIALS_SLICES)
        sample_predictions = do_predict_on_samples(model, n_class, X, max_sample, CONFIG.DEVICE)
        # sample_predictions = sample_predictions * 100
        sample_predictions = np.swapaxes(sample_predictions, 0, 1)

        # Highlight Trials and mark the trained on positions of each Trial
        vspans = create_plot_vspans(trials_start_samples, trials_classes, max_sample)
        vlines = create_vlines_from_trials_epochs(trial_tdeltas, trials_start_times, slices)

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
        if max_sample > 100000:
            plot_splits = 10
        trials_split_size = int(trials_start_samples.shape[0] / plot_splits)
        n_class_offset = 0
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
