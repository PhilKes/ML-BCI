"""
Script to check if default Training Routine works for all Datasets
with all available n_classes and only 3 Epochs
Stores results in './results/datasets_training_tests_{DateTime}'
"""
import os.path
import sys
from datetime import datetime

from data.datasets.datasets import DATASETS
from paths import results_folder
from scripts.batch_training import run_batch_training
from util.misc import datetime_to_folder_str, file_write

n_classes = ['2', '3', '4']
datasets = DATASETS.keys()
if __name__ == '__main__':
    confs = {}
    for ds_idx, ds in enumerate(datasets):
        confs[ds] = {
            'params': [
                ['--dataset', ds, '--epochs', '3'],
                ['--dataset', ds, '--epochs', '3', '--trials_slices', '2'],
            ],
            'names': [
                f'{ds}_test',
                f'{ds}_test_slices_2',
            ],
            'init': [
                lambda: None,
                lambda: None,
            ],
            'after': lambda: None,
        }
    folderName = f'datasets_training_tests_{datetime_to_folder_str(datetime.now())}'
    _, errors = run_batch_training(confs, n_classes, name=folderName)
    if len(errors) > 0:
        error_msg = str('\n'.join(errors))
        file_write(os.path.join(results_folder, folderName, 'error_log.txt'), error_msg)
        print(f"Some Errors occurred during Testing of {','.join(datasets)} Datasets' Trainings:\n{error_msg}",
              file=sys.stderr)
    else:
        print(f"Basic Training of all Datasets ({','.join(datasets)}) works as expected")
