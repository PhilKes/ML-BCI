"""
Script to execute Training to analyze the Frequency bands of neural responses on BCIC and PHYS dataset
- F1: 0-8Hz
- F2: 8-16Hz
- F1: 16-28Hz
For every Frequency Band the impact on different Time Slices (2.0s) are analyzed
Saves results in ./results/neural_resp_YYYY-mm-dd_HH_MM_SS"
"""
from datetime import datetime
from config import set_bandpassfilter
from batch_training import run_batch_training
from util.misc import datetime_to_folder_str
import numpy as np

# Neural Response Frequency bands
fbs = [(None, None), (0, 8), (8, 16), (16, 28)]
names = ['all', 'f1', 'f2', 'f3']

# 2 Second time slices in steps of 0.5
time_max = 4.0
time_delta = 2.0
time_step = 0.5
bcic_time_cue_offset = 2.0

n_classes = ['2']

confs = {
    'PHYS': {
        'params': [],
        'names': [],
        'init': [],
        'after': lambda: set_bandpassfilter(None, None, False),
    },
    'BCIC': {
        'params': [],
        'names': [],
        'init': [],
        'after': lambda: set_bandpassfilter(None, None, False),
    },
}
tmins = np.arange(0.0, time_max - time_delta + 0.01, time_step)
for tmin in tmins:
    for i, name in enumerate(names):
        tmax = tmin + time_delta
        confs['PHYS']['params'].append(
            ['--dataset', 'PHYS',
             '--tmin', f'{tmin}',
             '--tmax', f'{tmax}',
             ])
        confs['PHYS']['names'].append(f'phys_bp_{name}/t_{tmin}_{tmax}')
        confs['PHYS']['init'].append(lambda: set_bandpassfilter(*fbs[i]))

        bcic_tmin = tmin + bcic_time_cue_offset
        bcic_tmax = tmax + bcic_time_cue_offset
        confs['BCIC']['params'].append(
            ['--dataset', 'PHYS',
             '--tmin', f'{bcic_tmin}',
             '--tmax', f'{bcic_tmax}',
             ])
        confs['BCIC']['names'].append(f'bcic_bp_{name}/t_{tmin}_{tmax}')
        confs['BCIC']['init'].append(lambda: set_bandpassfilter(*fbs[i]))

print("Executing Training for Neural Response Frequency bands")
# results shape: [conf,run, n_class, 0]
results = run_batch_training(confs, n_classes, name=f'neural_resp_{datetime_to_folder_str(datetime.now())}')
# TODO filter bp_all s
#phys_results = results[0, :, :, 0]

