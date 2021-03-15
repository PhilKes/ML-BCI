"""
Helper Script to plot Benchmarking results.npz files
(Batch Latencies + Trial Inference Time)
"""
import argparse
import os

import numpy as np

from utils import matplot_grouped_configs

parser = argparse.ArgumentParser(
    description='Visualizer Script for Benchmark results.npz files')
parser.add_argument('--folder', type=str,
                    help=f'Result folder with result.npz file (result of main_all_confs.py)')
args = parser.parse_args()
if (args.folder is None):
    parser.error(f"You must specify a results folder")
args.folder = str(args.folder).replace('\\', '/')
if (not (os.path.exists(args.folder))):
    parser.error(f"Invalid folder specified ({args.folder})")

# Save all results
# Shapes:
# batch_sizes: (batch size idx)
# batch_lat_avgs: (config idx, batch size idx)
# trial_inf_time_avgs: (config idx, batch size idx)
results = np.load(f"{args.folder}/results.npz")
batch_sizes, batch_lat_avgs, trial_inf_time_avgs, n_classes = results['batch_sizes'], results['batch_lat_avgs'], \
                                                              results[
                                                                  'trial_inf_time_avgs'], results['n_classes']
# matplot(batch_lat_avgs, title='Average Batch Latencies', xlabel='Config',
#         ylabel='Latency in sec', labels=[f"Conf {i}" for i in range(len(batch_lat_avgs))],
#         save_path=args.folder, bar_plot=True)
for class_idx, n_class in enumerate(n_classes):
    matplot_grouped_configs(configs_data=batch_lat_avgs,
                            batch_sizes=batch_sizes,
                            class_idx=class_idx,
                            title=f'{n_class}class Batch Latencies',
                            ylabel='Latency in Sec.',
                            save_path=args.folder)
    matplot_grouped_configs(configs_data=trial_inf_time_avgs,
                            batch_sizes=batch_sizes,
                            class_idx=class_idx,
                            title=f'{n_class}class Trial Inference times',
                            ylabel='Inference Time in Sec.',
                            save_path=args.folder)
print(f"Plotted Benchmarking Statistics to '{args.folder}'")
