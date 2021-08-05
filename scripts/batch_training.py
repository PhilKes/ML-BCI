#!/usr/bin/python3
import os
from datetime import datetime

import numpy
import pandas as pd

from config import CONFIG
from data.datasets.bcic.bcic_dataset import BCIC
from data.datasets.lsmr21.lmsr_21_dataset import LSMR21
from data.datasets.phys.phys_dataset import PHYS
from main import single_run
from paths import results_folder, training_results_folder, training_ss_results_folder
from util.misc import print_pretty_table

default_options = ['-train']
confs = {
    'PHYS': {
        'params': [
            ['--dataset', PHYS.short_name],
            ['--dataset', PHYS.short_name, '--tmin', '0', '--tmax', '3'],
            ['--dataset', PHYS.short_name, '--tmin', '0', '--tmax', '4'],
            ['--dataset', PHYS.short_name, '--tmin', '-1', '--tmax', '1'],
            ['--dataset', PHYS.short_name],
            ['--dataset', PHYS.short_name],
            ['--dataset', PHYS.short_name],
            ['--dataset', PHYS.short_name, '--ch_motorimg', '16'],
            ['--dataset', PHYS.short_name, '--ch_motorimg', '16_openbci'],
        ],
        'names': [
            'default',
            'tmin_0_tmax_3',
            'tmin_0_tmax_4',
            'tmin_-1_tmax_1',
            'fmin_0_fmax_60',
            'fmin_2_fmax_60',
            'fmin_4_fmax_60',
            'motorimg_16',
            'motorimg_16_openbci',
        ],
        'init': [
            lambda: None,
            lambda: None,
            lambda: None,
            lambda: None,
            lambda: CONFIG.FILTER.set_filters(None, 60),
            lambda: CONFIG.FILTER.set_filters(2, 60),
            lambda: CONFIG.FILTER.set_filters(4, 60),
            lambda: None,
            lambda: None,

        ],
    },
    'BCIC': {
        'params': [
            ['--dataset', BCIC.short_name],
            ['--dataset', BCIC.short_name, '--tmin', '0', '--tmax', '3'],
            ['--dataset', BCIC.short_name, '--tmin', '0', '--tmax', '4'],
            ['--dataset', BCIC.short_name, '--tmin', '-1', '--tmax', '1'],
            ['--dataset', BCIC.short_name],
            ['--dataset', BCIC.short_name],
            ['--dataset', BCIC.short_name],
            ['--dataset', BCIC.short_name, '--ch_motorimg', '16'],
            ['--dataset', BCIC.short_name, '--ch_motorimg', '16_openbci'],
        ],
        'names': [
            'default',
            'tmin_0_tmax_3',
            'tmin_0_tmax_4',
            'tmin_-1_tmax_1',
            'fmin_0_fmax_60',
            'fmin_2_fmax_60',
            'fmin_4_fmax_60',
            'motorimg_16',
            'motorimg_16_openbci',
        ],
        'init': [
            lambda: None,
            lambda: None,
            lambda: None,
            lambda: None,
            lambda: CONFIG.FILTER.set_filters(None, 60),
            lambda: CONFIG.FILTER.set_filters(2, 60),
            lambda: CONFIG.FILTER.set_filters(4, 60),
            lambda: None,
            lambda: None,

        ],
    },
    'LSMR21': {
        'params': [
            ['--dataset', LSMR21.short_name],
            ['--dataset', LSMR21.short_name, '--tmin', '0', '--tmax', '3'],
            ['--dataset', LSMR21.short_name, '--tmin', '0', '--tmax', '4'],
            ['--dataset', LSMR21.short_name, '--tmin', '-1', '--tmax', '1'],
            ['--dataset', LSMR21.short_name],
            ['--dataset', LSMR21.short_name],
            ['--dataset', LSMR21.short_name],
            ['--dataset', LSMR21.short_name, '--ch_motorimg', '16'],
            ['--dataset', LSMR21.short_name, '--ch_motorimg', '16_openbci'],
        ],
        'names': [
            'default',
            'tmin_0_tmax_3',
            'tmin_0_tmax_4',
            'tmin_-1_tmax_1',
            'fmin_0_fmax_60',
            'fmin_2_fmax_60',
            'fmin_4_fmax_60',
            'motorimg_16',
            'motorimg_16_openbci',
        ],
        'init': [
            lambda: None,
            lambda: None,
            lambda: None,
            lambda: None,
            lambda: CONFIG.FILTER.set_filters(None, 60),
            lambda: CONFIG.FILTER.set_filters(2, 60),
            lambda: CONFIG.FILTER.set_filters(4, 60),
            lambda: None,
            lambda: None,

        ],
    },
}

train_ss_options = ['-train_ss', '--model']
live_sim_options = ['-live_sim', '--model']
start = datetime.now()

folder = "batch_trainings/params"
default_n_classes = ['2']

train_ss = False
live_sim = False


# All Configurations to execute Training with


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
            # Reset Global Config before each run
            CONFIG.reset()
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
