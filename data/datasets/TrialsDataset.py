from typing import Any, List, Tuple

import torch
from torch.utils.data import Dataset
import numpy as np


class TrialsDataset(Dataset):
    """
    Class: TrialsDataset(Dataset)

    Abstact Superclass for Tensor Datasets for EEG Trials Data

      Description:
      Dataset class which is based on torch.utils.data.Dataset. This type of class is
      required for creating a pytorch dataloader.
      Methods __len__ and __get_item__ must be implemented.
    """
    subjects: List[int]
    n_class: int
    device: Any
    equal_trials: bool
    preloaded_data: np.ndarray
    preloaded_labels: np.ndarray
    ch_names: List[str]

    # Either a number or a list of numbers
    trials_per_subject: Any

    def __init__(self, subjects: List[int], n_class: int, device, preloaded_tuple: Tuple[np.ndarray],
                 ch_names=[], equal_trials=True):
        """
        Method: constructor
        Parameters:
            subjects: list of subjects
        """
        self.subjects = subjects
        self.n_classes = n_class
        self.device = device
        self.equal_trials = equal_trials
        self.ch_names = ch_names
        if preloaded_tuple is not None:
            self.preloaded_data = preloaded_tuple[0]
            self.preloaded_labels = preloaded_tuple[1]

    def __len__(self):
        """
        Length of Dataset (trials)
        If trials_per_subject is a list return sum
        else return length of subjects * trials_per_subject
        :return: Amount of Trials in Dataset
        """
        if isinstance(self.trials_per_subject, list):
            ds_len = 0
            for subject_idx in range(len(self.subjects)):
                ds_len = ds_len + self.trials_per_subject[subject_idx]
            # print("ds_len = ", ds_len)
            return ds_len
        return len(self.subjects) * self.trials_per_subject

    def load_trial(self, trial):
        """
        Determines corresponding Subject of trial and loads subject's data+labels
        :return: trial data (X) and trial label (y)
        """
        raise NotImplementedError('This method is not implemented!')

    def __getitem__(self, trial):
        """
        Returns a single trial as Tensor with Labels
        :return: Tensor(Data), Labels
        """
        X, y = self.load_trial(trial)
        # Shape of 1 Batch (list of multiple __getitem__() calls):
        # [samples (BATCH_SIZE), 1 , Channels (len(ch_names), Timepoints (641)]
        X = torch.as_tensor(X[None, ...], device=self.device, dtype=torch.float32)
        # X = TRANSFORM(X)
        return X, y
