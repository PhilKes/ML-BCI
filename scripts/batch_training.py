#!/usr/bin/python3
import os
import sys
from datetime import datetime

import numpy
import pandas as pd

from config import CONFIG
from data.datasets.bcic.bcic_dataset import BCIC
from data.datasets.lsmr21.lmsr_21_dataset import LSMR21
from data.datasets.phys.phys_dataset import PHYS
from main import single_run
from paths import results_folder, training_results_folder, training_ss_results_folder
from util.misc import print_pretty_table, save_dataframe, file_write

parent_folder = "batch_trainings"
"""
All Batch Configurations to execute Training with in 'confs' Dictionary
Add new Keys for new Training batches, e.g.:
    'Example_Batch': {
      # A Training Config consists of:
      #   * Command Line params for 'main.py -train' ('params')
      #   * A name for the result folder of the Config ('names')
      #   * An initialization Method executed before the Training ('init')
      # Every Row represents a Training Configuration
      #   -> e.g. the first Training Config is defined by params[0], names[0], init[0]
      #   -> 'params', 'names', 'init' have to have same amount of rows
        'params': [
            # You can add any available Argument for 'main.py -train'
            ['--dataset', PHYS.short_name, '--tmin', '2', '--tmax', '4'],
            ['--dataset', BCIC.short_name, '--ch_motorimg', '16_openbci'],
        ],
        'names': [
            # Training results will be stored in /results/{parent_folder}/Example_Batch/{names[i]}
            'phys_tmin_2_tmax_4',
            'bcic_16_openbci_fmin_4_fmax_60',
        ],
        'init': [
            # Init. Method as single-line lambda (optional, can also just be 'lambda: None' if no init. is necessary)
            lambda: None,
            lambda: CONFIG.FILTER.set_filter(4,60),
        ],
      # Training Results are summarized per Batch as a Table in:
      # /results/{parent_folder}/Example_Batch/Example_Batch_training.txt
    },
"""
used_n_classes = ['2']

confs = {

    # 'PHYS_new_3_4_class': {
    #     'params': [
    #         ['--dataset', PHYS.short_name],
    #     ],
    #     'names': [
    #         'phys_without_rest_trials',
    #     ],
    #     'init': [
    #         lambda: None,
    #     ],
    # },
    'lsmr_dropout': {
        'params': [
            ['--dataset', LSMR21.short_name],
            ['--dataset', LSMR21.short_name],
            ['--dataset', LSMR21.short_name],
            ['--dataset', LSMR21.short_name],
        ],
        'names': [
            'lmsr_do_0.2',
            'lmsr_do_0.3',
            'lmsr_do_0.4',
            'lmsr_do_0.5',
        ],
        'init': [
            lambda: CONFIG.NET.set_model_params(dropout=0.2),
            lambda: CONFIG.NET.set_model_params(dropout=0.3),
            lambda: CONFIG.NET.set_model_params(dropout=0.4),
            lambda: CONFIG.NET.set_model_params(dropout=0.5),
        ],
    },

}
default_options = ['-train']

train_ss_options = ['-train_ss', '--model']
live_sim_options = ['-live_sim', '--model']

train_ss = False
live_sim = False


# Run Training for every Configuration in confs for all n_classes
# Returns List of numpy arrays with [conf,run, n_class, (acc/OF))]
def run_batch_training(configs=confs, n_classes=used_n_classes, name=parent_folder):
    # Loop to execute all Configurations
    # Create .csv and .txt files with all Runs of a batch
    # e.g. /batch_sizes/..._batch_training.txt
    results_list = []
    errors = []
    for conf_name in configs:
        conf_folder = f"{name}/{conf_name}"
        conf = configs[conf_name]
        runs = len(conf['params'])

        # Result array for avg. accuracy + OF for each Run
        # (runs, len(n_classes), 2 (acc + OF))
        classes = len(n_classes)
        runs_results = numpy.zeros((runs, classes, 2))
        # Execute each run consisting of
        # name, params, init(optional)
        for run in range(runs):
            # Reset Global Config before each run
            CONFIG.reset()
            if 'init' in conf.keys():
                conf['init'][run]()
            params = conf['params'][run]
            training_folder = f"{conf_folder}/conf_{conf['names'][run]}"
            arguments = default_options + ['--name', training_folder] + params
            if '--n_classes' not in params:
                arguments += ['--n_classes'] + n_classes
            try:
                n_classes_accs, n_classes_ofs = single_run(arguments)
            except Exception as e:
                msg = f"Error occurred during Training for conf '{conf_name}', run '{conf['names'][run]}':\n" + str(e)
                file_write(os.path.join(results_folder, training_folder, 'error_log.txt'), msg)
                print(msg, file=sys.stderr)
                errors.append(msg)
                continue
            # Store run results (Accuracies/Overfittings)
            for n_class in range(classes):
                runs_results[run, n_class, 0] = n_classes_accs[n_class]
                runs_results[run, n_class, 1] = n_classes_ofs[n_class]
            if train_ss:
                single_run(
                    train_ss_options + [
                        f"{results_folder}/{training_folder}{training_results_folder}/"] +
                    ['--n_classes'] + n_classes)
            if train_ss & live_sim:
                train_ss_folder = f"{results_folder}/{training_folder}{training_results_folder}{training_ss_results_folder}"
                # Live sim on all /training_ss/*
                for x in os.listdir(train_ss_folder):
                    single_run(
                        live_sim_options + [os.path.join(train_ss, x)] +
                        ['--n_classes'] + n_classes)
        if 'after' in conf.keys():
            conf['after']()
        results_list.append(runs_results.copy())
        # Prepare results for Pandas
        runs_results = runs_results.reshape((runs, classes * 2), order='F')
        columns = []
        for n_class in n_classes:
            columns.append(f"{n_class}class Acc")
        for n_class in n_classes:
            columns.append(f"{n_class}class OF")
        df = pd.DataFrame(data=runs_results, index=conf['names'], columns=columns)
        # Write results into .csv and .txt
        classes_str = ",".join(n_classes)
        df.to_csv(f"{results_folder}/{conf_folder}/{classes_str}_batch_training_results.csv")
        save_dataframe(df,
                       os.path.join(f"{results_folder}/{conf_folder}", f'{classes_str}_{conf_name}_batch_training.txt'))
        print_pretty_table(df)
    return results_list, errors


if __name__ == '__main__':
    run_batch_training()
