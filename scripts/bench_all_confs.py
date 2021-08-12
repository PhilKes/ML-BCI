#!/usr/bin/python3
"""
Script to execute benchmarking of all possible Configurations
Executes main.py for every Configuration in benchmark mode and
saves results in a parent folder (./results/benchmark/all_confs-{DateTime})
Creates results.npz file containing all Batch Latency Avgs and Inference Time per Trial Avgs
results can be visualized with visualize_bench_all.py (provide --folder {parent_folder}
"""
import argparse
import numpy as np
from config import CONFIG
from main import single_run
from paths import benchmark_results_folder
from scripts.visualize_bench_all import visualize_benchmarks

parser = argparse.ArgumentParser(
    description='Script to run Benchmarking of trained EEGNet Model with all possible Configurations')
parser.add_argument('--model', type=str, default=None,
                    help='Relative Folder path of model used to benchmark (in ./results/.../training folder)')
parser.add_argument('--bs', nargs='+', type=int, default=[8, 16, 32],
                    help=f'Trial Batch Size (default:{CONFIG.MI.BATCH_SIZE})')
parser.add_argument('--all', dest='continuous', action='store_false',
                    help=f'If present, will only loop benchmarking over entire Physionet Dataset, with loading Subjects chunks in between Inferences (default: False)')
parser.add_argument('--iters', type=int, default=1,
                    help=f'Number of benchmark iterations over the Dataset in a loop (default:1, if --continuous:10)')
parser.add_argument('--tag', type=str, default=None,
                    help=f'Additional tag for the results folder name')
parser.add_argument('--n_classes', nargs='+', type=int, default=CONFIG.MI.N_CLASSES,
                    help="List of n-class Classifications to run (2/3/4-Class possible)")
args = parser.parse_args()

if args.model is None:
    parser.error("You have to specify a model to use for benchmarking (from ./results)")
if (len(args.bs) < 1) | any(bs < 1 for bs in args.bs):
    parser.error("Batchsizes (--bs) must be a list of integers > 0")

# if device = cpu, maximum batch size: 15 (on Jetson Nano)
# otherwise Error at outputs=net(inputs)(RuntimeError: NNPACK SpatialConvolution_updateOutput failed)
# maybe related to: https://github.com/pytorch/pytorch/pull/49464
all_confs = [
    ['--device', 'cpu'],
    ['--device', 'gpu'],
    ['--device', 'gpu', '--trt'],
    ['--device', 'gpu', '--trt', '--fp16'],
]
conf_names = [
    'c_cpu',
    'c_gpu',
    'c_gpu_trt_fp32',
    'c_gpu_trt_fp16',
]
if args.continuous:
    if args.iters == 1:
        args.iters = 10
else:
    for conf in all_confs:
        conf.append('--all')

parent_folder = f"{args.model}{benchmark_results_folder}"

# copy_config_txts(parent_folder)

default_options = ['-benchmark',
                   '--model', str(args.model),
                   '--iters', str(args.iters),
                   '--subjects_cs', str(CONFIG.MI.SUBJECTS_CS),
                   '--n_classes'] + [str(i) for i in args.n_classes]
batch_lat_avgs, trial_inf_time_avgs = np.zeros((len(all_confs), len(args.bs), len(args.n_classes))), np.zeros(
    (len(all_confs), len(args.bs), len(args.n_classes)))

# Run all Configurations with all Batchsizes
for i, conf in enumerate(all_confs):
    print(f"Conf {i} {conf}")
    for bs_idx, bs in enumerate(args.bs):
        # Make sure that batch_size !> 15 if CPU is used
        if (conf[1] == 'cpu') & (int(bs) > 15):
            continue
        batch_size = str(bs)
        batch_lat_avgs[i][bs_idx], trial_inf_time_avgs[i][bs_idx] = single_run(default_options + conf +
                                                                               ['--name', conf_names[i],
                                                                                '--tag', f"bs_{batch_size}",
                                                                                '--bs', batch_size])
# Save all results with numpy
# Shapes:
# batch_sizes: (batch size idx)
# batch_lat_avgs: (config idx, batch size idx)
# trial_inf_time_avgs: (config idx, batch size idx)
# n_classes: (len(args.n_classes)
np.savez(f"{parent_folder}/results.npz",
         batch_sizes=np.array(args.bs, dtype=np.int),
         batch_lat_avgs=batch_lat_avgs,
         trial_inf_time_avgs=trial_inf_time_avgs,
         n_classes=np.array(args.n_classes, dtype=np.int),
         conf_names=conf_names)

visualize_benchmarks(['--model', parent_folder])
