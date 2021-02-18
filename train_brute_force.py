#!/usr/bin/python3
"""
Runs Brute force to find minimum EEG Channels for 3class Classification
Executes main.py in training mode with every possible combination of EEG Channels
(number of channels given via --channels argument list)
(see config.py MNE_CHANNELS for all available EEG Channels)
"""
import argparse
import itertools
from datetime import datetime
import numpy as np
from config import SUBJECTS_CS, benchmark_results_folder, BATCH_SIZE, MNE_CHANNELS, training_results_folder
from main import single_run
from utils import datetime_to_folder_str

parser = argparse.ArgumentParser(
    description='Running brute force over main.py training mode with all possible combinations of EEG Channels of given lengths')
parser.add_argument('--channels', required=True, nargs='+', type=int, default=[64],
                    help='List of number of EEG Channels to train with (default: [64]')

args = parser.parse_args()

if (len(args.channels) < 1) | any(chs < 1 for chs in args.channels):
    parser.error("Channels (--channels) must be a list of integers > 0")

start = datetime.now()
parent_folder = f"{datetime_to_folder_str(start)}-chs_brute_force"

for i, num_chs in enumerate(args.channels):
    for chs_idx,chs in enumerate(itertools.combinations(MNE_CHANNELS, num_chs)):
        args = ['-train',
                '--device', 'gpu',
                '--name', f"{parent_folder}",
                '--tag', f"chs_{num_chs}_c_{chs_idx}",
                '--ch_names',
                ]
        for ch in chs:
            args.append(ch)
        single_run(args)

# np.savez(f"{training_results_folder}/{parent_folder}/results.npz",
#          batch_sizes=np.array(args.bs),
#          batch_lat_avgs=batch_lat_avgs,
#          trial_inf_time_avgs=trial_inf_time_avgs)
