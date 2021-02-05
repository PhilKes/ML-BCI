# Global Default Settings
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
N_CLASSES = [4]
# 2 sec before + 4 sec event + 2 sec after
EEG_TMIN = -2
EEG_TMAX = 6
