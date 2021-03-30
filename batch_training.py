#!/usr/bin/python3
import os
from datetime import datetime

import numpy
import pandas as pd

from config import results_folder, set_eeg_times, reset_eeg_times, training_results_folder, training_ss_results_folder
from data.physionet_dataset import set_rest_from_bl_run, set_rest_trials_less, set_rests_config
from main import single_run

default_options = ['-train']
train_ss_options = ['-train_ss', '--model']
live_sim_options = ['-live_sim', '--model']
start = datetime.now()

folder = "3class_params"
n_classes = ['3']
excluded_subject = 1
excluded_params = ['--excluded', f'{excluded_subject}']
train_ss = True
live_sim = True
# All Configurations to execute Training with
confs = {
    # 'defaults': {
    #     'params': [[] + excluded_params],
    #     'names': ['defaults']
    # },
    # 'tmax': {
    #     # main.py -train Params for each run
    #     'params': [['--tmin', '0', '--tmax', '1'] + excluded_params,
    #                ['--tmin', '0', '--tmax', '4'] + excluded_params,
    #                ['--tmin', '-1', '--tmax', '5'] + excluded_params],
    #     # name for subfolder for each run
    #     'names': ['tmax_1', 'tmax_4', 'tmin_-1_tmax_5'],
    #     # Initialize methods for each run to set global settings (optional)
    #     # len(params) = len(names) = len(init)
    #     # 'init': [
    #     #     lambda: set_eeg_times(0, 1),
    #     #     lambda: set_eeg_times(0, 4),
    #     #     lambda: set_eeg_times(-1, 5),
    #     # ],
    #     # Execute after all runs finished -> reset changed parameters (optional)
    #     'after': lambda: reset_eeg_times(),
    # },
    # 'slicing_4s': {
    #     'params': [['--tmin', '0', '--tmax', '4'] + excluded_params,
    #                ['--tmin', '0', '--tmax', '4', '--trials_slices', '2'] + excluded_params,
    #                ['--tmin', '0', '--tmax', '4', '--trials_slices', '4'] + excluded_params,
    #                ['--tmin', '0', '--tmax', '4', '--trials_slices', '4'] + excluded_params],
    #     'names': ['no_slices', '2_slices', '4_slices', '4_slices_rests_from_runs'],
    #     'init': [
    #         lambda: None,
    #         lambda: None,
    #         lambda: None,
    #         lambda: set_rest_from_bl_run(False)],
    #     'after': lambda: set_rest_from_bl_run(True)
    # },
    'chs': {
        'params': [['--ch_motorimg', '16'] + excluded_params,
                   ['--ch_motorimg', '16_2'] + excluded_params,
                   ['--ch_motorimg', '16_openbci'] + excluded_params,
                   ['--ch_motorimg', '16_bs'] + excluded_params],
        'names': ['motorimg_16', 'motorimg_16_2', 'motorimg_16_openbci', 'motorimg_16_bs']
    },
    'excluded': {
        'params': [['--excluded', '10'],
                   ['--excluded', '80'],
                   ['--excluded', '10', '80']],
        'names': ['excl_10', 'excl_50', 'excl_10_80']
    },
    'rest_trials': {
        'params': [[] + excluded_params, [] + excluded_params, [] + excluded_params, [] + excluded_params],
        'names': ['from_bl_run_4_less_rests', 'from_bl_run', 'from_runs', 'from_runs_4_less_rests'],
        'init': [
            lambda: set_rests_config(True, 4),
            lambda: set_rests_config(True, 0),
            lambda: set_rests_config(False, 0),
            lambda: set_rests_config(False, 4),
        ],
        'after': lambda: set_rests_config(True, 0)
    },

    # 'non_excluded_defaults': {
    #     'params': [['--tmin', '0', '--tmax', '1'],
    #                ['--tmin', '0', '--tmax', '4'],
    #                ['--tmin', '-1', '--tmax', '5']],
    #     'names': ['tmax_1', 'tmax_4', 'tmin_-1_tmax_5'],
    #     'after': lambda: reset_eeg_times(),
    # },

}

# Loop to execute all Configurations
# Create .csv and .txt files with all Runs of a batch
# e.g. /batch_sizes/..._batch_training.txt
for conf_name in confs:
    conf_folder = f"{folder}/{conf_name}"
    conf = confs[conf_name]
    runs = len(conf['params'])

    # Result array for avg. accuracy + OF for each Run
    # (runs, len(n_classes), 2 (acc + OF))
    classes = len(n_classes)
    runs_results = numpy.zeros((runs, classes, 2))
    # Execute each run consisting of
    # name, params, init(optional)
    for run in range(runs):
        if 'init' in conf.keys():
            conf['init'][run]()
        params = conf['params'][run]
        training_folder = f"{conf_folder}/conf_{conf['names'][run]}"
        n_classes_accs, n_classes_ofs = single_run(
            default_options +
            ['--n_classes'] + n_classes +
            ['--name', training_folder] + params)
        # Store run results (Accuracies/Overfittings)
        for n_class in range(classes):
            runs_results[run, n_class, 0] = n_classes_accs[n_class]
            runs_results[run, n_class, 1] = n_classes_ofs[n_class]
        if train_ss:
            single_run(
                train_ss_options + [
                    f"{results_folder}/{training_folder}{training_results_folder}/"] +
                ['--n_classes'] + n_classes)
        if live_sim:
            single_run(
                live_sim_options + [
                    f"{results_folder}/{training_folder}{training_results_folder}{training_ss_results_folder}/S{excluded_subject:03d}/"] +
                ['--n_classes'] + n_classes)

    if 'after' in conf.keys():
        conf['after']()
    # Prepare results for Pandas
    runs_results = runs_results.reshape((runs, classes * 2), order='F')
    columns = []
    for n_class in n_classes:
        columns.append(f"{n_class}class Acc")
        columns.append(f"{n_class}class OF")
    df = pd.DataFrame(data=runs_results, index=conf['names'], columns=columns)
    # Write results into .csv and .txt
    df.to_csv(f"{results_folder}/{conf_folder}/batch_training_results.csv")
    with open(os.path.join(f"{results_folder}/{conf_folder}", f'{conf_name}_batch_training.txt'),
              'w') as outfile:
        df.to_string(outfile)
    print(df)
