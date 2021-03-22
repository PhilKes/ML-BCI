#!/usr/bin/python3

from datetime import datetime
from main import single_run
from config import global_config, MOTORIMG_CHANNELS

from util.utils import datetime_to_folder_str

default_options = ['-train', '--n_classes', '2', '3']
start = datetime.now()
# folder = f"{datetime_to_folder_str(start)}_batch_training_ALL"
folder = "all_class_filters_2"
global_config.USE_NOTCH_FILTER = False
# global_config.FREQ_FILTER_LOWPASS = None
# global_config.FREQ_FILTER_HIGHPASS = None
# single_run(default_options + ['--name', f"{folder}/conf_no_early_stop", "--no_early_stop"])
# single_run(default_options + ['--name', f"{folder}/conf_early_stop"])
# global_config.USE_NOTCH_FILTER = True
# single_run(default_options + ['--name', f"{folder}/conf_no_early_stop_notch", "--no_early_stop"])
# single_run(default_options + ['--name', f"{folder}/conf_early_stop_notch"])
# global_config.FREQ_FILTER_HIGHPASS = 2
# global_config.FREQ_FILTER_LOWPASS = None
# # single_run(default_options + ['--name', f"{folder}/conf_no_es_bp_2_None", "--no_early_stop"])
# single_run(default_options + ['--name', f"{folder}/conf_es_bp_2_None"])
# global_config.FREQ_FILTER_HIGHPASS = 2
# global_config.FREQ_FILTER_LOWPASS = 60
# # single_run(default_options + ['--name', f"{folder}/conf_no_es_bp_2_60", "--no_early_stop"])
# single_run(default_options + ['--name', f"{folder}/conf_es_bp_2_60"])
# # global_config.USE_NOTCH_FILTER = False
# # single_run(default_options + ['--name', f"{folder}/conf_no_early_stop_bandpass", "--no_early_stop"])
# # single_run(default_options + ['--name', f"{folder}/conf_early_stop_bandpass"])
# # global_config.USE_NOTCH_FILTER = True
# global_config.FREQ_FILTER_HIGHPASS = 2
# global_config.FREQ_FILTER_LOWPASS = 40
# # single_run(default_options + ['--name', f"{folder}/conf_no_es_bp_2_40", "--no_early_stop"])
# single_run(default_options + ['--name', f"{folder}/conf_es_bp_2_40"])
# global_config.USE_NOTCH_FILTER = False
# global_config.FREQ_FILTER_HIGHPASS = None
# global_config.FREQ_FILTER_LOWPASS = None
# single_run(default_options + ['--name', f"{folder}/conf",'--ch_motorimg','16_openbci'])
# single_run(default_options + ['--name', f"{folder}/conf",'--ch_motorimg','14_4'])
# single_run(default_options + ['--name', f"{folder}/conf_chs_64",])

global_config.USE_NOTCH_FILTER = False
global_config.FREQ_FILTER_HIGHPASS = None
global_config.FREQ_FILTER_LOWPASS = None
single_run(default_options + ['--name', f"{folder}/conf", '--ch_motorimg', '16_openbci'])
global_config.FREQ_FILTER_LOWPASS = 60
single_run(default_options + ['--name', f"{folder}/conf_lp_60", '--ch_motorimg', '16_openbci'])
global_config.FREQ_FILTER_LOWPASS = 40
single_run(default_options + ['--name', f"{folder}/conf_lp_40", '--ch_motorimg', '16_openbci'])
global_config.USE_NOTCH_FILTER = True
single_run(default_options + ['--name', f"{folder}/conf_lp_40_notch", '--ch_motorimg', '16_openbci'])
global_config.FREQ_FILTER_LOWPASS = 60
single_run(default_options + ['--name', f"{folder}/conf_lp_60_notch", '--ch_motorimg', '16_openbci'])
global_config.FREQ_FILTER_LOWPASS = None
single_run(default_options + ['--name', f"{folder}/conf_notch", '--ch_motorimg', '16_openbci'])

global_config.USE_NOTCH_FILTER = False
global_config.FREQ_FILTER_HIGHPASS = None
global_config.FREQ_FILTER_LOWPASS = None
single_run(default_options + ['--name', f"{folder}/conf_chs_64"])
global_config.FREQ_FILTER_LOWPASS = 60
single_run(default_options + ['--name', f"{folder}/conf_lp_60_chs_64"])
global_config.FREQ_FILTER_LOWPASS = 40
single_run(default_options + ['--name', f"{folder}/conf_lp_40_chs_64"])
global_config.USE_NOTCH_FILTER = True
single_run(default_options + ['--name', f"{folder}/conf_lp_40_notch_chs_64"])
global_config.FREQ_FILTER_LOWPASS = 60
single_run(default_options + ['--name', f"{folder}/conf_lp_60_notch_chs_64"])
global_config.FREQ_FILTER_LOWPASS = None
single_run(default_options + ['--name', f"{folder}/conf_notch_chs_64"])
