"""
Script to execute Training to analyze the Frequency bands of neural responses on BCIC and PHYS dataset
- F1: 0-8Hz
- F2: 8-16Hz
- F1: 16-28Hz
Saves results in ./results/neural_resp_YYYY-mm-dd_HH_MM_SS"
"""
import datetime

from config import set_bandpassfilter
from scripts.batch_training import run_batch_training
from util.misc import datetime_to_folder_str
# Neural Response Frequency bands
f1 = (0, 8)
f2 = (8, 16)
f3 = (16, 28)
confs = {
    'PHYS': {
        'params': [
            ['--dataset', 'PHYS'],
            ['--dataset', 'PHYS'],
            ['--dataset', 'PHYS'],
        ],
        'names': ['phys_bp_f1', 'phys_bp_f2', 'phys_bp_f3'],
        'init': [
            lambda: set_bandpassfilter(*f1),
            lambda: set_bandpassfilter(*f2),
            lambda: set_bandpassfilter(*f3),
        ],
        'after': lambda: set_bandpassfilter(None, None, False),
    },
    'BCIC': {
        'params': [
            ['--dataset', 'BCIC'],
            ['--dataset', 'BCIC'],
            ['--dataset', 'BCIC'],
        ],
        'names': ['bcic_bp_f1', 'bcic_bp_f2', 'bcic_bp_f3'],
        'init': [
            lambda: set_bandpassfilter(*f1),
            lambda: set_bandpassfilter(*f2),
            lambda: set_bandpassfilter(*f3),
        ],
        'after': lambda: set_bandpassfilter(None, None, False),
    },
}
print("Executing Training for Neural Response Frequency bands")
run_batch_training(confs, ['2'], name=f'neural_resp_{datetime_to_folder_str(datetime.now())}')
