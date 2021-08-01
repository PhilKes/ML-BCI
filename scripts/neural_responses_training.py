"""
Script to execute Training to analyze the Frequency bands of neural responses on BCIC and PHYS dataset
- F1: 0-8Hz
- F2: 8-16Hz
- F1: 16-28Hz
For every Frequency Band the impact on different Time Slices (2.0s) are analyzed
Saves results in ./results/neural_resp_YYYY-mm-dd_HH_MM_SS"
Uses Cross Validation (--best argument to use only Best-Fold)

EXECUTE AS MODULE:
    python3 -m scripts.neural_responses_training --best_fold
"""
import argparse
from datetime import datetime

import numpy as np

from data.datasets.bcic.bcic_dataset import BCIC
from data.datasets.phys.phys_dataset import PHYS
from config import results_folder, FBS_NAMES, FBS, CONFIG
from data.data_utils import save_accs_panda, subtract_first_config_accs
from scripts.batch_training import run_batch_training
from util.misc import datetime_to_folder_str
from util.plot import matplot

parser = argparse.ArgumentParser(
    description='Script to analyze influence of Neural Response Frequency Bands (f1/f2/f3)')
parser.add_argument('--best_fold', dest='use_cv', action='store_false',
                    help=f"Use Best-Fold Accuracies to calculate influence of Frequency Bands instead of Cross Validation")

args = parser.parse_args()
ds_used = [BCIC.short_name, PHYS.short_name]

# Folds to use if --best_fold is used (0-base index)
ds_best_folds = [2, 2]

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
        'after': lambda: CONFIG.FILTER.set_filters(None, None, False),
    }

tmins = np.arange(0.0, time_max - time_delta + 0.01, time_step)
time_slices = []


def func(x): return lambda: CONFIG.FILTER.set_filters(*x)


for tmin in tmins:
    tmax = tmin + time_delta
    for ds_idx, ds in enumerate(ds_used):
        for fb_idx, fb_name in enumerate(FBS_NAMES):
            print(f'Training with tmin={tmin}, tmax={tmax} for {fb_name} with {ds}')
            confs[ds]['params'].append(
                ['--dataset', str(ds),
                 '--tmin', f'{tmin}',
                 '--tmax', f'{tmax}',
                 ])
            confs[ds]['names'].append(f'{ds.lower()}_bp_{fb_name}/t_{tmin}_{tmax}')
            confs[ds]['init'].append(func(FBS[fb_idx]))

            if args.use_cv is False:
                confs[ds]['params'][-1].extend(['--only_fold', str(ds_best_folds[ds_idx])])
    time_slices.append(f'{tmin}-{tmax}s')

print(f"Executing Training for Neural Response Frequency bands")
folderName = f'neural_resp_{datetime_to_folder_str(datetime.now())}_{"CV" if args.use_cv is True else "Best_Fold"}'

# results shape: [conf,run, n_class, (acc,OF)]
results = run_batch_training(confs, n_classes, name=folderName)
for ds_idx, ds in enumerate(ds_used):
    plot_data = results[ds_idx][:, :, 0]
    plot_data = np.reshape(plot_data, (len(time_slices), len(FBS))).T

    matplot(plot_data, title=f'{ds} Frequency Band Accuracies', fig_size=(8.0, 6.0),
            xlabel='2s Time Slice Interval', ylabel='Accuracy in %', labels=FBS_NAMES,
            x_values=['-'] + time_slices, min_x=0, marker='o', save_path=f"{results_folder}/{folderName}/{ds}")

    ds_acc_diffs = subtract_first_config_accs(results[ds_idx][:, :, 0], len(FBS))

    # Reshape into rows for every frequency band
    ds_acc_diffs = ds_acc_diffs.reshape((len(FBS) - 1, tmins.shape[0]), order='F')

    # Save accuracy differences as .csv and .txt
    save_accs_panda("neural_responses_accs", f"{results_folder}/{folderName}/{ds}",
                    ds_acc_diffs, time_slices, FBS_NAMES[1:], ds)
