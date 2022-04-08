"""
Test Accuracies of pretrained model
Determines Accuracies on Test Set of best Fold
with the Test Data being bandpassfiltered (all/f1/f2/f3)
specified model should have subdirectories with Training
of different Time Slices

EXECUTE AS MODULE:
    python3 -m scripts.best_fold_frequency_test --model
"""
import argparse
import logging

import numpy as np

from app.config import CONFIG
from app.data.data_utils import save_accs_panda
from app.data.datasets.datasets import DATASETS
from app.machine_learning.configs_results import load_npz
from app.machine_learning.modes import testing
from app.machine_learning.util import preferred_device
from app.scripts.neural_responses_training import FBS, FBS_NAMES
from app.util.misc import get_subdirs, makedir

parser = argparse.ArgumentParser(
    description='Script to Test Accuracy of trained model on filtered Test Data')
parser.add_argument('--model', type=str, default=None,
                    help='Relative Folder path of trained model(in ./results/.../training/ folder), used for -benchmark or -train_ss or -live_sim')
args = parser.parse_args()
model_path = args.model
n_class = 2

args = parser.parse_args()

device = preferred_device("gpu")
time_slices_dirs = get_subdirs(args.model)
logging.info(time_slices_dirs)
results = np.zeros((len(time_slices_dirs), len(FBS)))
for slice_idx, time_slices_dir in enumerate(time_slices_dirs):
    # Accuracy for every frequency band with every Time Slice (directory)
    training_folder = f"{args.model}/{time_slices_dir}/training"
    n_class_results = load_npz(f"{training_folder}/{n_class}class-training.npz")
    ds = n_class_results['mi_ds'].item()
    dataset = DATASETS[ds]
    tmin = n_class_results['tmin'].item()
    tmax = n_class_results['tmax'].item()
    ch_names = dataset.channels
    CONFIG.set_eeg_config(dataset.eeg_config)
    testing_folder = f"{training_folder}/testing"
    makedir(testing_folder)

    # logging.info("PRELOADING ALL DATA IN MEMORY")
    # preloaded_data, preloaded_labels = dataset.load_subjects_data(dataset.available_subjects, n_class,
    #                                                               ch_names, True, normalize=False)
    for fb_idx, (bp, fb_name) in enumerate(zip(FBS, FBS_NAMES)):
        logging.info(f'Testing with tmin={tmin}, tmax={tmax} for {fb_name} with {ds}')
        CONFIG.EEG.set_times(tmin, tmax, dataset.eeg_config.CUE_OFFSET)
        CONFIG.FILTER.set_filters(*bp)
        results[slice_idx, fb_idx] = testing(n_class, training_folder, device, ch_names)

    save_accs_panda(f"Fx-filtered_Test_accs", testing_folder, results[slice_idx], ['Accuracy in %'],
                    FBS_NAMES, tag=ds)

save_accs_panda(f"Time_Slices_Fx-filtered_Test_accs", args.model, results.T, time_slices_dirs,
                FBS_NAMES, tag=ds)


matplot(results.T, title=f'{ds} Time Slices Fx-filtered Testing', fig_size=(8.0, 6.0),
        xlabel='2s Time Slice Interval', ylabel='Accuracy in %', labels=FBS_NAMES,
        x_values=['-'] + time_slices_dirs, min_x=0, marker='o', save_path=args.model)
