"""
Helper Script to plot Benchmarking results.npz files
(Batch Latencies + Trial Inference Time)
"""
import argparse
import logging
import os
import sys

import numpy as np

from app.config import CONFIG
from app.util.plot import matplot_grouped_configs


def visualize_benchmarks(argv=sys.argv[1:]):
    parser = argparse.ArgumentParser(
        description='Visualizer Script for Benchmark results.npz files')
    parser.add_argument('--model', type=str,
                        help=f'Result folder with result.npz file (result of main_all_confs.py)')
    args = parser.parse_args(argv)
    if (args.model is None):
        parser.error(f"You must specify a results folder")
    args.model = str(args.model).replace('\\', '/')
    if (not (os.path.exists(args.model))):
        parser.error(f"Invalid folder specified ({args.model})")

    # Save all results
    # Shapes:
    # batch_sizes: (batch size idx)
    # batch_lat_avgs: (config idx, batch size idx)
    # trial_inf_time_avgs: (config idx, batch size idx)
    results = np.load(f"{args.model}/results.npz")
    batch_sizes, batch_lat_avgs, trial_inf_time_avgs, n_classes,conf_names = results['batch_sizes'], results['batch_lat_avgs'], \
                                                                  results['trial_inf_time_avgs'], results['n_classes'],\
                                                                    results['conf_names']
    # matplot(batch_lat_avgs, title='Average Batch Latencies', xlabel='Config',
    #         ylabel='Latency in sec', labels=[f"Conf {i}" for i in range(len(batch_lat_avgs))],
    #         save_path=args.model, bar_plot=True)
    # Determine highest used batch size for cpu
    logging.info(batch_sizes)
    cpu_max_bs_idx = 0
    for i, bs in enumerate(batch_sizes):
        if bs > CONFIG.MI.JETSON_CPU_MAX_BS:
            logging.info("CPU batch size %s %s", bs, i)
            cpu_max_bs_idx = i
            break
    for class_idx, n_class in enumerate(n_classes):
        # CPU Performance is always way worse
        # Plot CPU benchmark separately
        # device='cpu' config should always be first config
        # only Plot first batch size (8), cant benchmark with batch size > 15
        matplot_grouped_configs(configs_data=batch_lat_avgs[:1][:cpu_max_bs_idx],
                                batch_sizes=batch_sizes[:cpu_max_bs_idx],
                                class_idx=class_idx,
                                title=f'{n_class}class Batch Latencies CPU',
                                ylabel='latency in sec.',
                                xlabels=conf_names[:1],
                                fig_size=(4.0, 4.0),
                                legend_pos='lower right',
                                min_x=-1.0,
                                max_x=2.0,
                                save_path=args.model)
        matplot_grouped_configs(configs_data=trial_inf_time_avgs[:1][:cpu_max_bs_idx],
                                batch_sizes=batch_sizes[:cpu_max_bs_idx],
                                class_idx=class_idx,
                                title=f'{n_class}class Trial Inference Times CPU',
                                ylabel='inference time in sec.',
                                xlabels=conf_names[:1],
                                fig_size=(4.0, 4.0),
                                legend_pos='lower right',
                                min_x=-1.0,
                                max_x=2.0,
                                save_path=args.model)
        # Plot all configs with device='gpu'
        matplot_grouped_configs(configs_data=batch_lat_avgs[1:],
                                batch_sizes=batch_sizes,
                                class_idx=class_idx,
                                title=f'{n_class}class Batch Latencies GPU',
                                ylabel='Latency in Sec.',
                                xlabels=conf_names[1:],
                                fig_size=(9.0, 4.0),
                                conf_offset=1,
                                save_path=args.model)
        matplot_grouped_configs(configs_data=trial_inf_time_avgs[1:],
                                batch_sizes=batch_sizes,
                                class_idx=class_idx,
                                fig_size=(9.0, 4.0),
                                xlabels=conf_names[1:],
                                conf_offset=1,
                                title=f'{n_class}class Trial Inference times GPU',
                                ylabel='Inference Time in Sec.',
                                save_path=args.model)
    logging.info(f"Plotted Benchmarking Statistics to '{args.model}'")


if __name__ == '__main__':
    visualize_benchmarks()
