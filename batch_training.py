#!/usr/bin/python3
import os
from datetime import datetime

import numpy
import pandas as pd
from config import global_config, eeg_config, eegnet_config, results_folder
from main import single_run

default_options = ['-train']
start = datetime.now()
# folder = f"{datetime_to_folder_str(start)}_batch_training_ALL"

#
# global_config.USE_NOTCH_FILTER = False
# global_config.FREQ_FILTER_HIGHPASS = None
# global_config.FREQ_FILTER_LOWPASS = None
#
#
#
# single_run(default_options + ['--name', f"{folder}/batch_size/conf_bs_16", '--bs', '16'])
# single_run(default_options + ['--name', f"{folder}/batch_size/conf_bs_32", '--bs', '32'])
#
# eeg_config.EEG_TMIN = 0
# eeg_config.EEG_TMAX = 2
# single_run(default_options + ['--name', f"{folder}/tmax/conf_tmax_2"])
# eeg_config.EEG_TMAX = 3
# single_run(default_options + ['--name', f"{folder}/tmax/conf_tmax_3"])
# eeg_config.EEG_TMAX = 4
# single_run(default_options + ['--name', f"{folder}/tmax/conf_tmax_4"])
# eeg_config.EEG_TMIN = -1
# eeg_config.EEG_TMAX = 5
# single_run(default_options + ['--name', f"{folder}/tmax/conf_tmin_-1_tmax_5"])
#
# eeg_config.EEG_TMIN = 0
# eeg_config.EEG_TMAX = 3
#
# eegnet_config.pool_size = 4
# single_run(default_options + ['--name', f"{folder}/pool/conf_pool_4"])
# eegnet_config.pool_size = 8
# single_run(default_options + ['--name', f"{folder}/pool/conf_pool_8"])
#
# eegnet_config.pool_size = 4
#
# single_run(default_options + ['--name', f"{folder}/chs16/conf", '--ch_motorimg 16'])
# single_run(default_options + ['--name', f"{folder}/chs16/conf", '--ch_motorimg 16_2'])
# single_run(default_options + ['--name', f"{folder}/chs16/conf", '--ch_motorimg 16_openbci'])

folder = "2_3_class_params"
n_classes = ['2', '3']
confs = {
    # Key is the subfolder name
    'batch_size': {
        # Params for each run
        'params': [
            ['--bs', '16'],
            ['--bs', '32'],
        ],
        # Name for each run
        'names': ['bs_16', 'bs_32']
    },
    'tmax': {
        'params': [[], [], [], []],
        'names': ['tmax_2', 'tmax_3', 'tmax_4', 'tmin_-1_tmax_5'],
        # Initialize method for each run
        # len(params) = len(init)
        'init': [
            lambda: set_eeg_times(0, 2),
            lambda: set_eeg_times(0, 3),
            lambda: set_eeg_times(0, 4),
            lambda: set_eeg_times(-1, 5),
        ],
        # Execute after all runs -> reset parameter
        'after': lambda: set_eeg_times(0, 3),
    },
    'pool': {
        'params': [[], []],
        'names': ['pool_4', 'pool_8'],
        'init': [
            lambda: set_poolsize(4),
            lambda: set_poolsize(8),
        ],
        'after': lambda: set_poolsize(4)
    },
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
