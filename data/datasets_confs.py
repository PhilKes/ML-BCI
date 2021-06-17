from data.bcic_data_loading import bcic_load_subjects_data, bcic_create_loaders_from_splits
from data.bcic_dataset import BCIC_name, BCIC_ALL_SUBJECTS, BCIC_cv_folds, BCIC_CONFIG, BCIC_CHANNELS, BCIC_short_name
from data.data_loading import load_subjects_data, phys_create_loaders_from_splits
from data.physionet_dataset import PHYS_name, PHYS_ALL_SUBJECTS, PHYS_cv_folds, PHYS_CONFIG, PHYS_CHANNELS, \
    PHYS_short_name

# String keys for Dataset Dictionaries
NAME = 'name'
AVAILABLE_SUBJECTS = 'available_subjects'
LOAD_SUBJECTS = 'load_subjects'
LOADERS_FROM_SPLITS = 'loaders_from_splits'
FOLDS = 'folds'
EEG_CONF = 'eeg_config'
CHANNELS = 'channels'

# Dictionaries contain config data + methods to load the dataset
PHYS_DICT = {
    NAME: PHYS_name,
    AVAILABLE_SUBJECTS: PHYS_ALL_SUBJECTS,
    LOAD_SUBJECTS: load_subjects_data,
    LOADERS_FROM_SPLITS: phys_create_loaders_from_splits,
    FOLDS: PHYS_cv_folds,
    EEG_CONF: PHYS_CONFIG,
    CHANNELS: PHYS_CHANNELS
}

BCIC_DICT = {
    NAME: BCIC_name,
    AVAILABLE_SUBJECTS: BCIC_ALL_SUBJECTS,
    LOAD_SUBJECTS: bcic_load_subjects_data,
    LOADERS_FROM_SPLITS: bcic_create_loaders_from_splits,
    FOLDS: BCIC_cv_folds,
    EEG_CONF: BCIC_CONFIG,
    CHANNELS: BCIC_CHANNELS
}

DS_DICTS = {
    PHYS_short_name: PHYS_DICT,
    BCIC_short_name: BCIC_DICT
}
