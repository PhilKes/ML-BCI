# Project'sath
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))

sys.path.append(ROOT)
to_path = lambda x: os.path.join(ROOT, x)

results_folder = to_path('results')
training_results_folder = '/training'
benchmark_results_folder = '/benchmark'
live_sim_results_folder = '/live_sim'
training_ss_results_folder = '/training_ss'

trained_model_name = "trained_model.pt"
trained_ss_model_name = "trained_ss_model.pt"
chs_names_txt = "ch_names.txt"
# Folder where MNE downloads Physionet Dataset to
# on initial Run MNE needs to download the Dataset

datasets_folder = 'D:/EEG_Datasets'
