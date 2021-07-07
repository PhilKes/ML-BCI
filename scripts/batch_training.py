#!/usr/bin/python3
import os
from datetime import datetime

import numpy
import pandas as pd

from config import results_folder, training_results_folder, training_ss_results_folder
from data.datasets.bcic.bcic_dataset import BCIC
from data.datasets.phys.phys_dataset import PHYS
from main import single_run
from util.misc import print_pretty_table

default_options = ['-train']
train_ss_options = ['-train_ss', '--model']
live_sim_options = ['-live_sim', '--model']
start = datetime.now()

folder = "defaults_2s_and_4s"
default_n_classes = ['2']

train_ss = False
live_sim = False

# All Configurations to execute Training with
confs = {
    # 'defaults': {
    #     'params': [[]],
    #     'names': ['defaults']
    # },
    # 'slicing': {
    #     'params': [
    #         # ['--tmin', '0', '--tmax', '4'],
    #         # ['--tmin', '0', '--tmax', '4', '--trials_slices', '2'],
    #         # ['--tmin', '0', '--tmax', '4', '--trials_slices', '4'],
    #         ['--tmin', '0', '--tmax', '4', '--trials_slices', '8']],
    #     'names': [
    #         # 'no_slices',
    #         # '2_slices',
    #         # '4_slices',
    #         '8_slices'
    #     ],
    #     'init': [
    #         lambda: None,
    #         lambda: None,
    #         lambda: None
    #     ],
    #     'after': lambda: set_rest_from_bl_run(True)
    # },
    # 'tmin_tmax': {
    #     # main.py -train Params for each run
    #     'params': [
    #         ['--tmin', '0', '--tmax', '1'],
    #         ['--tmin', '0', '--tmax', '3'],
    #         ['--tmin', '-0.5', '--tmax', '3'],
    #         ['--tmin', '0', '--tmax', '4'],
    #         ['--tmin', '-1', '--tmax', '5']
    #     ],
    #     # name for subfolder for each run
    #     'names': [
    #         'tmax_1', 'tmax_3',
    #         'tmin_-05_tmax_3', 'tmax_4', 'tmin_-1_tmax_5'],
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
    # 'rest_trials': {
    #     'params': [[] + excluded_params, [] + excluded_params, [] + excluded_params, [] + excluded_params],
    #     'names': ['from_bl_run_4_less_rests', 'from_bl_run', 'from_runs', 'from_runs_4_less_rests'],
    #     'init': [
    #         lambda: set_rests_config(True, 4),
    #         lambda: set_rests_config(True, 0),
    #         lambda: set_rests_config(False, 0),
    #         lambda: set_rests_config(False, 4),
    #     ],
    #     'after': lambda: set_rests_config(True, 0)
    # },

    # 'pool': {
    #     'params': [[], []],
    #     'names': ['pool_4', 'pool_8'],
    #     'init': [
    #         lambda: set_poolsize(4),
    #         lambda: set_poolsize(8),
    #     ],
    #     'after': lambda: set_poolsize(4)
    # },
    # 'chs': {
    #     'params': [
    #         ['--ch_motorimg', '16_csp'],
    #         ['--ch_motorimg', '16'],
    #         ['--ch_motorimg', '16_2'],
    #         ['--ch_motorimg', '16_openbci'],
    #         ['--ch_motorimg', '16_bs']],
    #     'names': ['chs_16_csp', 'chs_16', 'chs_16_2', 'chs_16_openbci', 'chs_16_bs']
    # },
    # 'excluded': {
    #     'params': [['--excluded', '42']],
    #     'names': ['excl_42']
    # },
    'PHYS': {
        'params': [
            ['--dataset', PHYS.short_name, '--tmin', '0', '--tmax', '2'],
            ['--dataset', PHYS.short_name, '--tmin', '0', '--tmax', '4']
        ],
        'names': ['phys_all_2s', 'phys_all_4s']
    },
    'BCIC': {
        'params': [
            ['--dataset', BCIC.short_name, '--tmin', '0', '--tmax', '2'],
            ['--dataset', BCIC.short_name, '--tmin', '0', '--tmax', '4']
        ],
        'names': ['bcic_all_2s', 'bcic_all_4s']
    },
}


# Run Training for every Configuration in confs for all n_classes
# Returns List of numpy arrays with [conf,run, n_class, (acc/OF))]
def run_batch_training(configs=confs, n_classes=default_n_classes, name=folder):
    # Loop to execute all Configurations
    # Create .csv and .txt files with all Runs of a batch
    # e.g. /batch_sizes/..._batch_training.txt
    results_list = []
    for conf_name in configs:
        conf_folder = f"{name}/{conf_name}"
        conf = configs[conf_name]
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
            if train_ss & live_sim:
                train_ss_folder = f"{results_folder}/{training_folder}{training_results_folder}{training_ss_results_folder}"
                # Live sim on all /training_ss/*
                for x in os.listdir(train_ss_folder):
                    single_run(
                        live_sim_options + [os.path.join(train_ss, x)] +
                        ['--n_classes'] + n_classes)

        if 'after' in conf.keys():
            conf['after']()
        results_list.append(runs_results.copy())
        # Prepare results for Pandas
        runs_results = runs_results.reshape((runs, classes * 2), order='F')
        columns = []
        for n_class in n_classes:
            columns.append(f"{n_class}class Acc")
        for n_class in n_classes:
            columns.append(f"{n_class}class OF")
        df = pd.DataFrame(data=runs_results, index=conf['names'], columns=columns)
        # Write results into .csv and .txt
        classes_str = ",".join(n_classes)
        df.to_csv(f"{results_folder}/{conf_folder}/{classes_str}_batch_training_results.csv")
        with open(os.path.join(f"{results_folder}/{conf_folder}", f'{classes_str}_{conf_name}_batch_training.txt'),
                  'w') as outfile:
            df.to_string(outfile)
        print_pretty_table(df)
    return results_list


if __name__ == '__main__':
    run_batch_training()
