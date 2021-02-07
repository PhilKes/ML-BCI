# Global Default Settings
import torchvision

PLATFORM = "PC"  # used for results folder names
CUDA = True
PLOTS = True
VERBOSE = False

results_folder = './results'
datasets_folder = './datasets/'

BASELINE_CORRECTION = False

# Learning Rate Settings
LR = dict(
    start=0.01,
    milestones=[20, 50],
    gamma=0.1
)

# Training Settings
EPOCHS = 100
BATCH_SIZE = 16
SPLITS = 5
# only 3-class classifciation for now
N_CLASSES = [3,4]
# 2 sec before + 4 sec event + 2 sec after
EEG_TMIN = -2
EEG_TMAX = 6

# Caluclate with python_test.py get_mean_std():
channel_means = [-4.4257e-06, -3.6615e-06, -3.5425e-06, -3.1105e-06, -1.9982e-06,
                 -3.3686e-06, -4.0484e-06, -3.2589e-06, -1.2037e-06, -3.1303e-06,
                 -1.7123e-06, -1.3769e-06, -3.8620e-06, -3.8488e-06, -3.8019e-06,
                 -3.4305e-06, -2.2203e-06, -3.4104e-06, -2.4583e-06, -2.7768e-06,
                 -2.1735e-06, -3.1017e-05, -2.8601e-05, -3.1928e-05, -2.5437e-05,
                 -1.9414e-05, -1.3425e-05, -1.7509e-05, -2.3842e-05, -9.7920e-06,
                 -1.0631e-05, -7.9275e-06, -7.3165e-06, -6.1011e-06, -6.9056e-06,
                 -7.8441e-06, -7.9372e-06, -8.7261e-06, -2.7639e-06, -2.2479e-06,
                 -1.4207e-07, -8.4886e-08, -2.4083e-06, -9.0723e-07, -8.6527e-08,
                 -8.9375e-07, -2.7776e-07, -1.2364e-07, -3.0605e-06, -2.8032e-06,
                 -3.5766e-06, -3.1065e-06, -4.2012e-06, -3.6984e-06, -4.8093e-06,
                 -3.0346e-06, -3.5540e-06, -4.0859e-06, -2.8972e-06, -5.5720e-06,
                 -4.2342e-06, -2.3905e-06, -3.9677e-06, -3.4984e-06]
channel_stds = [6.7977e-05, 6.7956e-05, 6.7134e-05, 6.3398e-05, 6.3251e-05, 6.6822e-05,
                6.4781e-05, 6.4687e-05, 6.1571e-05, 5.9905e-05, 5.6210e-05, 5.7690e-05,
                6.0474e-05, 5.8444e-05, 5.7063e-05, 5.8189e-05, 5.7179e-05, 5.7279e-05,
                5.3735e-05, 5.5777e-05, 5.5726e-05, 1.2370e-04, 1.2028e-04, 1.2554e-04,
                1.1895e-04, 9.8800e-05, 8.8457e-05, 9.7427e-05, 1.1697e-04, 8.5738e-05,
                8.4617e-05, 7.5524e-05, 7.5537e-05, 7.1688e-05, 7.6756e-05, 7.6226e-05,
                8.2670e-05, 8.3560e-05, 6.6245e-05, 6.4996e-05, 6.0052e-05, 5.6108e-05,
                5.8593e-05, 6.8983e-05, 5.8227e-05, 6.1241e-05, 6.2516e-05, 6.0535e-05,
                5.8508e-05, 6.1415e-05, 5.3972e-05, 5.4090e-05, 5.4028e-05, 5.5977e-05,
                5.6267e-05, 6.1128e-05, 6.0736e-05, 6.0434e-05, 5.3701e-05, 5.7876e-05,
                6.0989e-05, 6.0698e-05, 5.9087e-05, 5.0683e-05]
# Calculated using torch.mean on whole Dataset
torch_mean = -0.0258
torch_std = 1.0560

means = -6.1795e-06
stds = 7.1783e-05
# TRANSFORM = torchvision.transforms.Normalize(channel_means, channel_stds)
TRANSFORM = lambda x: x * 1e4
