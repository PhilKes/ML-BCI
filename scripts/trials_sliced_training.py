import argparse
import sys
from datetime import datetime

from sklearn.model_selection import GroupKFold

from config import CONFIG
from data.datasets.bcic.bcic_dataset import BCIC
from data.datasets.datasets import DATASETS
from machine_learning.configs_results import save_config, training_config_str, create_results_folders
from machine_learning.modes import do_n_class_training_cv, save_n_class_results
from machine_learning.util import overlapping_trials_slicing, preferred_device
from util.dot_dict import DotDict
from util.misc import groups_labels, datetime_to_folder_str, print_numpy_counts

slice_length = 1.5
time_step = 1.5

n_class = 2
num_epochs = 100


def trials_sliced_training(argv=sys.argv[1:]):
    parser = argparse.ArgumentParser(
        description='Script to execute Training with overlapping Trials slices defined by --slice_length + --time_step')
    parser.add_argument('--slice_length', type=float, default=slice_length,
                        help=f'Time Length of 1 Slice (default:{slice_length})')
    parser.add_argument('--time_step', type=float, default=time_step,
                        help=f'Time step between 2 Slices (default:{time_step})')
    parser.add_argument('--dataset', type=str, default=BCIC.short_name,
                        help=f'Name of the MI dataset (available: {",".join([ds for ds in DATASETS])})')
    args = parser.parse_args(argv)

    if args.dataset not in DATASETS:
        parser.error(f"Dataset '{args.dataset}' does not exist (available: {','.join([ds for ds in DATASETS])}))")

    # Use GPU for model & tensors if available
    CONFIG.DEVICE = preferred_device("gpu")

    mi_ds = args.dataset
    dataset = DATASETS[mi_ds]
    # Dataset dependent EEG config structure re-initialization
    CONFIG.set_eeg_config(dataset.CONSTANTS.CONFIG)
    trial_tmin, trial_tmax = dataset.CONSTANTS.TRIAL_TMIN, dataset.CONSTANTS.TRIAL_TMAX
    # Set tmin, tmax to load entire Trial
    CONFIG.EEG.set_times(trial_tmin, trial_tmax)
    used_subjects = dataset.CONSTANTS.ALL_SUBJECTS
    ch_names = dataset.CONSTANTS.CHANNELS
    folds = dataset.CONSTANTS.cv_folds
    only_fold = None
    batch_size = CONFIG.MI.BATCH_SIZE
    cue_offset = dataset.CONSTANTS.CONFIG.CUE_OFFSET
    dir_results = f'trials_sliced_training/{mi_ds}_slice_length_{args.slice_length}_time_step_{args.time_step}-{datetime_to_folder_str(datetime.now())}'
    dir_results = create_results_folders(path=dir_results)
    config = DotDict(num_epochs=num_epochs, batch_size=batch_size, folds=folds, lr=CONFIG.MI.LR, device=CONFIG.DEVICE,
                     n_classes=[n_class], ch_names=ch_names, early_stop=False, excluded=[],
                     mi_ds=mi_ds, only_fold=only_fold)
    save_config(training_config_str(config), ch_names, dir_results, None)

    preloaded_data, preloaded_labels = dataset.load_subjects_data(used_subjects, n_class, ch_names)
    print(preloaded_data.shape)
    act_slices = preloaded_data.shape[1]
    preloaded_data, preloaded_labels = overlapping_trials_slicing(preloaded_data, preloaded_labels, slice_length,
                                                                  time_step, dataset.CONSTANTS.REST_PHASES)
    print("Sliced Shape:")
    print(preloaded_data.shape)
    print("Sliced Labels Stats Subject1:")
    print_numpy_counts(preloaded_labels[0])

    calced_slices = int(((CONFIG.EEG.TMAX - CONFIG.EEG.TMIN) - slice_length) / time_step) + 1
    act_slices = preloaded_data.shape[1] / act_slices
    print("calced_slices:", calced_slices, "actual slices:", act_slices)
    # Group labels (subjects in same group need same group label)
    groups = groups_labels(len(used_subjects), folds)

    # Split Data into training + util
    cv = GroupKFold(n_splits=folds)
    # Sliced Data contains new 'rest' class therefore 2class Data becomes 3class
    train_n_class = n_class + 1
    # Set Trials Slices to amount of slices for a Trial for TrialsDataset's length calculation
    # Data is loaded with n_class=2 then sliced and then trained with n_class=3
    # therefore the self.n_trials_max in bcic_data_loading.py has to be calculated correctly with
    # the Trials Slices
    CONFIG.EEG.set_trials_slices(calced_slices // train_n_class * n_class)
    # Set Amount of Samples to Slice Sample Length, so that EEGNet Model is created with correct parameters in Training
    CONFIG.EEG.set_samples(slice_length * CONFIG.EEG.SAMPLERATE)

    run_data, best_model = do_n_class_training_cv(cv, used_subjects, groups, folds, train_n_class, num_epochs,
                                                  only_fold,
                                                  dataset, [],
                                                  preloaded_data, preloaded_labels, batch_size, ch_names)
    n_class_accuracy, n_class_overfitting_diff = save_n_class_results(train_n_class, mi_ds, run_data, folds, only_fold,
                                                                      batch_size, [], best_model, dir_results)


if __name__ == '__main__':
    trials_sliced_training()
