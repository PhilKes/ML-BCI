"""
Script to check if default Training Routine works for all Datasets
with all available n_classes and only 3 Epochs
Stores results in './results/datasets_training_tests_{DateTime}'
"""
import os.path
from datetime import datetime
from data.datasets.datasets import DATASETS
from paths import results_folder
from scripts.batch_training import run_batch_training
from util.misc import datetime_to_folder_str, file_write

n_classes = ['2']
if __name__ == '__main__':
    confs = {}
    for ds_idx, ds in enumerate(['BCIC']):
        # for ds_idx, ds in enumerate(DATASETS.keys()):
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
    try:
        run_batch_training(confs, n_classes, name=folderName)
    except Exception as e:
        file_write(os.path.join(results_folder, folderName, 'error_log.txt'), str(e))
        raise e
    print(f"Basic Training of all Datasets ({','.join(DATASETS.keys())}) works as expected")
