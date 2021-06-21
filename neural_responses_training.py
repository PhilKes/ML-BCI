"""
Script to execute Training to analyze the Frequency bands of neural responses on BCIC and PHYS dataset
- F1: 0-8Hz
- F2: 8-16Hz
- F1: 16-28Hz
For every Frequency Band the impact on different Time Slices (2.0s) are analyzed
Saves results in ./results/neural_resp_YYYY-mm-dd_HH_MM_SS"
Uses Cross Validation (--best argument to use only Best-Fold)
"""
import argparse
from datetime import datetime

import numpy as np

from batch_training import run_batch_training
from config import set_bandpassfilter
from data.datasets.bcic.bcic_dataset import BCIC_time_cue_offset, BCIC_short_name
from data.data_utils import save_accs_panda, subtract_first_config_accs
from data.datasets.phys.physionet_dataset import PHYS_time_cue_offset, PHYS_short_name
from util.misc import datetime_to_folder_str

parser = argparse.ArgumentParser(
    description='Script to analyze influence of Neural Response Frequency Bands (f1/f2/f3)')
parser.add_argument('--best_fold', dest='use_cv', action='store_false',
                    help=f"Use Best-Fold Accuracies to calculate influence of Frequency Bands instead of Cross Validation")

args = parser.parse_args()

ds_used = [BCIC_short_name, PHYS_short_name]
ds_time_cue_offsets = [BCIC_time_cue_offset, PHYS_time_cue_offset]
# Folds to use if --best is used
ds_best_folds = [1, 2]

# Neural Response Frequency bands
fbs = [(None, None), (None, 8), (8, 16), (16, 28)]
fbs_names = ['all', 'f1', 'f2', 'f3']

# 2 Second time slices in steps of 0.5 (max t=4.0)
time_max = 4.0
time_delta = 2.0
time_step = 0.5

n_classes = ['2']

confs = {}
for ds_idx, ds in enumerate(ds_used):
    confs[ds] = {
        'params': [],
        'names': [],
        'init': [],
        'after': lambda: set_bandpassfilter(None, None, False),
    }

tmins = np.arange(0.0, time_max - time_delta + 0.01, time_step)
time_slices = []


def func(x): return lambda: set_bandpassfilter(*x)


for tmin in tmins:
    tmax = tmin + time_delta
    for ds_idx, ds in enumerate(ds_used):
        for fb_idx, fb_name in enumerate(fbs_names):
            print(f'Training with tmin={tmin}, tmax={tmax} for {fb_name} with {ds}')
            confs[ds]['params'].append(
                ['--dataset', str(ds),
                 '--tmin', f'{tmin + ds_time_cue_offsets[ds_idx]}',
                 '--tmax', f'{tmax + ds_time_cue_offsets[ds_idx]}',
                 ])
            confs[ds]['names'].append(f'{ds.lower()}_bp_{fb_name}/t_{tmin}_{tmax}')
            confs[ds]['init'].append(func(fbs[fb_idx]))

            if args.use_cv is False:
                confs[ds]['params'][-1].extend(['--only_fold', str(ds_best_folds[ds_idx])])
    time_slices.append(f'{tmin}-{tmax}s')

print(f"Executing Training for Neural Response Frequency bands")
folderName = f'neural_resp_{datetime_to_folder_str(datetime.now())}'

# results shape: [conf,run, n_class, (acc,OF)]
results = run_batch_training(confs, n_classes, name=folderName)
for ds_idx, ds in enumerate(ds_used):
    ds_acc_diffs = subtract_first_config_accs(results[ds_idx][:, :, 0], len(fbs))

    # Reshape into rows for every frequency band
    ds_acc_diffs = ds_acc_diffs.reshape((len(fbs) - 1, tmins.shape[0]), order='F')

    # Save accuracy differences as .csv and .txt
    save_accs_panda(folderName, ds_acc_diffs, time_slices, fbs_names[1:], n_classes, ds)
