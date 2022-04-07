"""
Script to showcase usage of all available modes via Terminal
    1. Cross Validation Training (-train)
    2. Subject-specific Training (-train_ss)
    3. Live Simulation Run (-live_sim)
    4. Benchmarking (-benchmark)
"""
from subprocess import call

from app.data.datasets.bcic.bcic_dataset import BCIC
from app.data.datasets.lsmr21.lmsr_21_dataset import LSMR21
from app.data.datasets.phys.phys_dataset import PHYS
from app.paths import to_path, results_folder

excluded_subject = 1
n_classes = ['2', '3', '4']
datasets = [BCIC.short_name, PHYS.short_name, LSMR21.short_name]

# Get Relative path to 'main.py' in Project Root Directory
main_py_relative = to_path('main.py')

for ds in datasets:
    train_name = f"example/{ds}"
    """
    Execute global 2/3/4-class Cross Validation Training of PHYS Dataset with 3 Epochs per Fold
    and with Subject 1 excluded from the Training
    Results can be found in ../results/{train_name}/training:
    - n_class_training.txt containing achieved Accuracies on the Test Dataset
    - config.txt with all important Parameters for the Training process
    - n_class_trained_model.pt containg the weights/bias values of the trained model
    - n_class_training.npz containing all important parameters and results of the Training process
    - ch_names.txt containing list of used EEG Channel names
    . *.png Plots of achieved Accuracies, Epoch Losses, etc.
    """
    call(["python3", main_py_relative,
          "-train",
          "--n_classes"] + n_classes +
         ["--dataset", f"{ds}",
          "--epochs", "3",
          "--name", f"{train_name}",
          "--excluded", f"{excluded_subject}"])

    """
    Execute subject-specific Training on previously globally trained Model
    with excluded Subject from global Training run
    (can manually set Subject to be trained on with --subject)
    Results can be found in ../results/{train_name}/training/training_ss/S001:
    - n_class_training.txt containing achieved Accuracies on the Test Dataset
    - config.txt with all important Parameters for the Training process
    - n_class_trained_model.pt containg the weights/bias values of the trained model
    - n_class_training.npz containing all important parameters and results of the Training process
    - ch_names.txt containing list of used EEG Channel names
    """

    call(["python3", main_py_relative,
          "-train_ss",
          "--n_classes"] + n_classes +
         ["--model", f"{results_folder}/{train_name}/training"])

    """
    Executes Live Simulation predictions on subject-specific trained Model
    Results can be found in ../results/{train_name}/training/training_ss/S001/live_sim:
    - *.png Plots of Live Simulation run with predictions for every sample in the Subject's Run
    - n_class_predictions.npz with the actual_labels + predicted_labels arrays
    """
    call(["python3", main_py_relative,
          "-live_sim",
          "--n_classes"] + n_classes +
         ["--model", f"{results_folder}/{train_name}/training/training_ss/S00{excluded_subject}"])
    """
    Execute Benchmarking on globally trained model
    Results can be found in ../results/{train_name}/training/benchmark/:
    - n_class_benchmark.txt of Inference Performance (default without using TensorRT/Floating Point16)
    """
    call(["python3", main_py_relative,
          "-benchmark",
          "--n_classes"] + n_classes +
         ["--model", f"{results_folder}/{train_name}/training/"])
