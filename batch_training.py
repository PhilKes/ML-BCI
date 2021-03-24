#!/usr/bin/python3
import os
from datetime import datetime

import numpy
import pandas as pd
from config import global_config, eeg_config, eegnet_config, results_folder
from main import single_run

default_options = ['-train']
start = datetime.now()

folder = "2_3_class_params"
n_classes = ['2', '3']
# All Configurations to execute Training with
confs = {
    # Key is the subfolder name
    # 'batch_size': {
    #     # Params for each run
    #     'params': [
    #         ['--bs', '16'],
    #         ['--bs', '32'],
    #     ],
    #     # Name for each run
    #     'names': ['bs_16', 'bs_32']
    # },
    # 'tmax': {
    #     'params': [[], [], [], []],
    #     'names': ['tmax_2', 'tmax_3', 'tmax_4', 'tmin_-1_tmax_5'],
    #     # Initialize method for each run (optional)
    #     # len(params) = len(names) = len(init)
    #     'init': [
    #         lambda: set_eeg_times(0, 2),
    #         lambda: set_eeg_times(0, 3),
    #         lambda: set_eeg_times(0, 4),
    #         lambda: set_eeg_times(-1, 5),
    #     ],
    #     # Execute after all runs finished -> reset parameters (optional)
    #     'after': lambda: set_eeg_times(0, 3),
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
    'chs': {
        'params': [
            ['--ch_motorimg', '16'],
            ['--ch_motorimg', '16_2'],
            ['--ch_motorimg', '16_openbci'],
        ],
        'names': ['chs_16', 'chs_16_2', 'chs_16_openbci']
    }
}


def set_eeg_times(tmin, tmax):
    eeg_config.EEG_TMIN = tmin
    eeg_config.EEG_TMAX = tmax


def set_poolsize(size):
    eegnet_config.pool_size = size


# Loop to exectue alls Configurations
# Create .csv and .txt files with all Runs of a batch
# e.g. /batch_sizes/..._batch_training.txt
for conf_name in confs:
    conf_folder = f"{folder}/{conf_name}"
    conf = confs[conf_name]
    runs = len(conf['params'])

    # Result array for avg. accuracy + OF for each Run
    # (runs, len(n_classes), 2 (acc + OF))
    len_n_classes = 2
    res = numpy.zeros((runs, len(n_classes), 2))
    for run in range(runs):
        if 'init' in conf.keys():
            conf['init'][run]()
        params = conf['params'][run]
        res[run, 0], res[run, 1] = single_run(
            ['-train', '--n_classes'] + n_classes +
            ['--name', f"{conf_folder}/conf_{conf['names'][run]}"] + params)
    if 'after' in conf.keys():
        conf['after']()
    # numpy.savetxt(f"{conf_folder}/acc_of_results.csv", res, delimiter=',', header="Acc,OF", comments="")
    res = res.reshape((runs, len_n_classes * 2),order='F')
    #res_rows = numpy.zeros((runs, len(n_classes) * 2))
    columns = []
    for n_class in n_classes:
        columns.append(f"{n_class}class Acc")
        columns.append(f"{n_class}class OF")
    df = pd.DataFrame(data=res, index=conf['names'], columns=columns)
    df.to_csv(f"{results_folder}/{conf_folder}/batch_training_results.csv")
    with open(os.path.join(f"{results_folder}/{conf_folder}", f'{conf_name}_batch_training.txt'),
              'w') as outfile:
        df.to_string(outfile)
    print(df)
