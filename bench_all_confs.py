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
from config import SUBJECTS_CS, benchmark_results_folder, BATCH_SIZE, N_CLASSES
from main import single_run
from utils import datetime_to_folder_str, copy_config_txts

parser = argparse.ArgumentParser(
    description='Script to run Benchmarking of trained EEGNet Model with all possible Configurations')
parser.add_argument('--bs', nargs='+', type=int, default=[BATCH_SIZE],
                    help=f'Trial Batch Size (default:{BATCH_SIZE})')
parser.add_argument('--continuous', action='store_true',
                    help=f'If present, will only loop benchmarking over 1 subject chunk, without loading other Subjects in between')
parser.add_argument('--iters', type=int, default=1,
                    help=f'Number of benchmark iterations over the Dataset in a loop (default:1, if --continuous:10)')
parser.add_argument('--tag', type=str, default=None,
                    help=f'Additional tag for the results folder name')
parser.add_argument('--n_classes', nargs='+', type=int, default=N_CLASSES,
                    help="List of n-class Classifications to run (2/3/4-Class possible)")
args = parser.parse_args()

if (len(args.bs) < 1) | any(bs < 1 for bs in args.bs):
    parser.error("Batchsizes (--bs) must be a list of integers > 0")

# if device = cpu, maximum batch size: 15 (on Jetson Nano)
# otherwise Error at outputs=net(inputs)(RuntimeError: NNPACK SpatialConvolution_updateOutput failed)
# maybe related to: https://github.com/pytorch/pytorch/pull/49464
all_confs = [
    # ['--device', 'cpu'],
    # ['--device', 'gpu', '--trt', '--fp16', '--ch_motorimg', '8'],
    ['--device', 'gpu'],
    ['--device', 'gpu', '--trt'],
    ['--device', 'gpu', '--trt', '--fp16'],
]
if args.continuous:
    for conf in all_confs:
        conf.append('--continuous')
    if args.iters == 1:
        args.iters = 10

start = datetime.now()
parent_folder = f"{datetime_to_folder_str(start)}{f'_{args.tag}' if args.tag is not None else ''}-all_confs"

copy_config_txts(parent_folder)

default_options = ['-benchmark',
                   '--iters', str(args.iters),
                   '--subjects_cs', str(SUBJECTS_CS),
                   '--n_classes'] + [str(i) for i in args.n_classes]
batch_lat_avgs, trial_inf_time_avgs = np.zeros((len(all_confs), len(args.bs), len(args.n_classes))), np.zeros(
    (len(all_confs), len(args.bs), len(args.n_classes)))

# Run all Configurations with all Batchsizes
for i, conf in enumerate(all_confs):
    print(f"Conf {i} {conf}")
    for bs_idx, bs in enumerate(args.bs):
        # Make sure that batch_size !> 15 if CPU is used
        batch_size = str(15) if (conf[1] == 'cpu') & (bs > 15) else str(bs)
        batch_lat_avgs[i][bs_idx], trial_inf_time_avgs[i][bs_idx] = single_run(default_options + conf +
                                                                               ['--name', f"{parent_folder}/conf_{i}",
                                                                                '--tag', f"bs_{bs}",
                                                                                '--bs', batch_size])
# Save all results with numpy
# Shapes:
# batch_sizes: (batch size idx)
# batch_lat_avgs: (config idx, batch size idx)
# trial_inf_time_avgs: (config idx, batch size idx)
# n_classes: (len(args.n_classes)
np.savez(f"{benchmark_results_folder}/{parent_folder}/results.npz",
         batch_sizes=np.array(args.bs, dtype=np.int),
         batch_lat_avgs=batch_lat_avgs,
         trial_inf_time_avgs=trial_inf_time_avgs,
         n_classes=np.array(args.n_classes, dtype=np.int))
