#!/usr/bin/python3

from datetime import datetime
from main import single_run
from config import global_config

from util.utils import datetime_to_folder_str

default_options = ['-train', '--n_classes', '2']
start = datetime.now()
folder = f"{datetime_to_folder_str(start)}_batch_training"

global_config.USE_NOTCH_FILTER = False
# global_config.FREQ_FILTER_LOWPASS = None
# global_config.FREQ_FILTER_HIGHPASS = None
# single_run(default_options + ['--name', f"{folder}/conf_no_early_stop", "--no_early_stop"])
# single_run(default_options + ['--name', f"{folder}/conf_early_stop"])
# global_config.USE_NOTCH_FILTER = True
# single_run(default_options + ['--name', f"{folder}/conf_no_early_stop_notch", "--no_early_stop"])
# single_run(default_options + ['--name', f"{folder}/conf_early_stop_notch"])
global_config.FREQ_FILTER_HIGHPASS = 2
global_config.FREQ_FILTER_LOWPASS = 60
single_run(default_options + ['--name', f"{folder}/conf_no_early_stop_notch_bandpass", "--no_early_stop"])
single_run(default_options + ['--name', f"{folder}/conf_early_stop_notch_bandpass"])
global_config.USE_NOTCH_FILTER = False
single_run(default_options + ['--name', f"{folder}/conf_no_early_stop_bandpass", "--no_early_stop"])
single_run(default_options + ['--name', f"{folder}/conf_early_stop_bandpass"])
global_config.USE_NOTCH_FILTER = True
global_config.FREQ_FILTER_HIGHPASS = 4
global_config.FREQ_FILTER_LOWPASS = 40
single_run(default_options + ['--name', f"{folder}/conf_no_early_stop_notch_bandpass_440", "--no_early_stop"])
single_run(default_options + ['--name', f"{folder}/conf_early_stop_notch_bandpass_440"])
global_config.USE_NOTCH_FILTER = False
single_run(default_options + ['--name', f"{folder}/conf_no_early_stop_bandpass_440", "--no_early_stop"])
single_run(default_options + ['--name', f"{folder}/conf_early_stop_bandpass_440"])