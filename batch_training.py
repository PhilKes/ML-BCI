#!/usr/bin/python3

from datetime import datetime

from config import global_config, eeg_config, eegnet_config
from main import single_run

default_options = ['-train', '--n_classes', '2', '3']
start = datetime.now()
# folder = f"{datetime_to_folder_str(start)}_batch_training_ALL"


global_config.USE_NOTCH_FILTER = False
global_config.FREQ_FILTER_HIGHPASS = None
global_config.FREQ_FILTER_LOWPASS = None

folder = "2_3_class_params"
single_run(default_options + ['--name', f"{folder}/batch_size/conf_bs_16", '--bs', '16'])
single_run(default_options + ['--name', f"{folder}/batch_size/conf_bs_32", '--bs', '32'])

eeg_config.EEG_TMIN = 0
eeg_config.EEG_TMAX = 2
single_run(default_options + ['--name', f"{folder}/tmax/conf_tmax_2"])
eeg_config.EEG_TMAX = 3
single_run(default_options + ['--name', f"{folder}/tmax/conf_tmax_3"])
eeg_config.EEG_TMAX = 4
single_run(default_options + ['--name', f"{folder}/tmax/conf_tmax_4"])
eeg_config.EEG_TMIN = -1
eeg_config.EEG_TMAX = 5
single_run(default_options + ['--name', f"{folder}/tmax/conf_tmin_-1_tmax_5"])

eegnet_config.pool_size = 4
single_run(default_options + ['--name', f"{folder}/pool/conf_pool_4"])
eegnet_config.pool_size = 8
single_run(default_options + ['--name', f"{folder}/pool/conf_pool_8"])
