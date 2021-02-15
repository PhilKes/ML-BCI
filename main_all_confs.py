"""
Script to execute benchmarking of all possible Configurations
Executes main.py for every Configuration in benchmark mode and
saves results in a parent folder (./results/benchmark/all_confs-{DateTime})
Creates results.npz file containing all Batch Latency Avgs and Inference Time per Trial Avgs
results can be visualized with visualize_results.py (provide --folder {parent_folder}
"""
from datetime import datetime
import numpy as np
from config import SUBJECTS_CS, benchmark_results_folder
from main import single_run
from utils import datetime_to_folder_str

# TODO add CPU run
# TODO add different batch sizes
all_confs = [
    ['--device', 'gpu', ],
    ['--device', 'gpu', '--trt'],
    ['--device', 'gpu', '--trt', '--fp16'],
]

start = datetime.now()
parent_folder = f"{datetime_to_folder_str(start)}-all_confs"

default_options = ['-benchmark',
                   '--iters', '1',
                   '--subjects_cs', str(SUBJECTS_CS)]
batch_lat_avgs, trial_inf_time_avgs = np.zeros((len(all_confs))), np.zeros((len(all_confs)))

for i, conf in enumerate(all_confs):
    print(f"Conf {i} {conf}")
    batch_lat_avgs[i], trial_inf_time_avgs[i] = single_run(
        default_options + conf + ['--name', f"{parent_folder}/conf_{i}"])
np.savez(f"{benchmark_results_folder}/{parent_folder}/results.npz", batch_lat_avgs=batch_lat_avgs, trial_inf_time_avgs=trial_inf_time_avgs)
