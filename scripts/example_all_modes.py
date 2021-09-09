"""
Script to showcase usage of all available modes
"""
from subprocess import call

from data.datasets.phys.phys_dataset import PHYS

train_name = "example_train"
excluded_subject = 1
n_class = 2

# Execute global 2-class Cross Validation Training of PHYS Dataset with 3 Epochs per Fold
# and with Subject 1 excluded from the Training
# Results can be found in ../results/{train_name}/training:
# - n_class_training.txt containing achieved Accuracies on the Test Dataset
# - config.txt with all important Parameters for the Training process
# - n_class_trained_model.pt containg the weights/bias values of the trained model
# - n_class_training.npz containing all important parameters and results of the Training process
# - ch_names.txt containing list of used EEG Channel names
# . *.png Plots of achieved Accuracies, Epoch Losses, etc.
call(["python3", "../main.py",
      "-train",
      "--n_classes", f"{n_class}",
      "--dataset", f"{PHYS.short_name}",
      "--epochs", "3",
      "--name", f"{train_name}",
      "--excluded", f"{excluded_subject}"])

# Execute subject-specific Training on previously globally trained Model
# with excluded Subject from global Training run
# (can manually set Subject to be trained on with --subject)
# Results can be found in ../results/{train_name}/training/training_ss/S001:
# - n_class_training.txt containing achieved Accuracies on the Test Dataset
# - config.txt with all important Parameters for the Training process
# - n_class_trained_model.pt containg the weights/bias values of the trained model
# - n_class_training.npz containing all important parameters and results of the Training process
# - ch_names.txt containing list of used EEG Channel names
call(["python3", "../main.py",
      "-train_ss",
      "--n_classes", f"{n_class}",
      "--model", f"../results/{train_name}/training"])

# Executes Live Simulation predictions on subject-specific trained Model
# Results can be found in ../results/{train_name}/training/training_ss/S001/live_sim:
# - *.png Plots of Live Simulation run with predictions for every sample in the Subject's Run
# - n_class_predictions.npz with the actual_labels + predicted_labels arrays
call(["python3", "../main.py",
      "-live_sim",
      "--n_classes", f"{n_class}",
      "--model", f"../results/{train_name}/training/training_ss/S00{excluded_subject}"])
