"""

"""
import argparse
from datetime import datetime

import numpy as np

from batch_training import run_batch_training
from config import set_bandpassfilter
from data.data_utils import save_accs_panda, subtract_first_config_accs
from data.datasets.bcic.bcic_dataset import BCIC_short_name
from data.datasets.datasets import DATASETS
from data.datasets.phys.phys_dataset import PHYS_short_name
from machine_learning.configs_results import load_npz, get_results_file
from util.misc import datetime_to_folder_str

parser = argparse.ArgumentParser(
    description='Script to Test Accuracy of trained model on f1/f2/f3 Test Data')
parser.add_argument('--model', type=str, default=None,
                    help='Relative Folder path of trained model(in ./results/.../training/ folder), used for -benchmark or -train_ss or -live_sim')
args = parser.parse_args()
model_path = args.model
n_class = 2
n_class_results = load_npz(get_results_file(model_path, n_class))
dataset = DATASETS[n_class_results['mi_ds']]

args = parser.parse_args()
phys = DATASETS[PHYS_short_name]
bcic = DATASETS[BCIC_short_name]
ds_used = [bcic.name_short, phys.name_short]

# Folds to use if --best_fold is used (0-base index)
ds_best_folds = [2, 2]

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
                 '--tmin', f'{tmin}',
                 '--tmax', f'{tmax}',
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
