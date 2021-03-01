#!/usr/bin/python3
"""
Script to execute benchmarking of all possible Configurations
Executes main.py for every Configuration in benchmark mode and
saves results in a parent folder (./results/benchmark/all_confs-{DateTime})
Creates results.npz file containing all Batch Latency Avgs and Inference Time per Trial Avgs
results can be visualized with visualize_results.py (provide --folder {parent_folder}
"""
import argparse
from datetime import datetime
import numpy as np
from config import SUBJECTS_CS, benchmark_results_folder, BATCH_SIZE
from main import single_run
from utils import datetime_to_folder_str

parser = argparse.ArgumentParser(
    description='Script to run Benchmarking of trained EEGNet Model with all possible Configurations')
parser.add_argument('--bs', nargs='+', type=int, default=[BATCH_SIZE],
                    help=f'Trial Batch Size (default:{BATCH_SIZE})')
args = parser.parse_args()

if (len(args.bs) < 1) | any(bs < 1 for bs in args.bs):
    parser.error("Batchsizes (--bs) must be a list of integers > 0")

# if device = cpu, maximum batch size: 15 (on Jetson Nano)
# otherwise Error at outputs=net(inputs)(RuntimeError: NNPACK SpatialConvolution_updateOutput failed)
# maybe related to: https://github.com/pytorch/pytorch/pull/49464
all_confs = [
   # ['--device', 'cpu'],
    ['--device', 'gpu'],
    ['--device', 'gpu', '--trt'],
    ['--device', 'gpu', '--trt', '--fp16'],
]

start = datetime.now()
parent_folder = f"{datetime_to_folder_str(start)}-all_confs"

default_options = ['-benchmark',
                   '--iters', '1',
                   '--subjects_cs', str(SUBJECTS_CS)]
batch_lat_avgs, trial_inf_time_avgs = np.zeros((len(all_confs), len(args.bs))), np.zeros((len(all_confs), len(args.bs)))

# Run all Configurations with all Batchsizes
for i, conf in enumerate(all_confs):
    print(f"Conf {i} {conf}")
    for bs_idx, bs in enumerate(args.bs):
        # Make sure that batch_size !> 15 if CPU is used
        batch_size = str(15) if (conf[1] == 'cpu') & (bs > 15) else str(bs)
        batch_lat_avgs[i][bs_idx], trial_inf_time_avgs[i][bs_idx] = single_run(
            default_options + conf +
            ['--name', f"{parent_folder}/conf_{i}",
             '--tag', f"bs_{bs}",
             '--bs', batch_size])
# Save all results with numpy
# Shapes:
# batch_sizes: (batch size idx)
# batch_lat_avgs: (config idx, batch size idx)
# trial_inf_time_avgs: (config idx, batch size idx)
np.savez(f"{benchmark_results_folder}/{parent_folder}/results.npz",
         batch_sizes=np.array(args.bs),
         batch_lat_avgs=batch_lat_avgs,
         trial_inf_time_avgs=trial_inf_time_avgs)
