from typing import Any, List, Tuple

import torch
from torch.utils.data import Dataset
import numpy as np

from app.config import CONFIG


class TrialsDataset(Dataset):
    """
    Class: TrialsDataset(Dataset)

    Abstact Superclass for Tensor Datasets for EEG Trials Data

      Description:
      Dataset class which is based on torch.utils.data.Dataset. This type of class is
      required for creating a pytorch dataloader.
      Methods __len__ and __get_item__ must be implemented.
    """
    # Local Subjects (only subjects of this Dataset)
    subjects: List[int]
    # All subjects that are used for the entire Training process, not only in this Dataset
    used_subjects: List[int]
    n_class: int
    device: Any
    equal_trials: bool
    preloaded_data: np.ndarray
    preloaded_labels: np.ndarray
    ch_names: List[str]

    # Either a number or a list of numbers
    trials_per_subject: Any

    def __init__(self, subjects: List[int], used_subjects: List[int], n_class: int,
                 preloaded_tuple: Tuple[np.ndarray, np.ndarray],
                 ch_names=[], equal_trials=True):
        """
        Method: constructor
        :param subjects: list of subjects
        :param used_subjects: All subjects whose data are included in preloaded_tuple
        :param preloaded_tuple: (preloaded_data,preloaded_labels) of entire used Dataset
        """
        self.subjects = subjects
        self.used_subjects = used_subjects
        self.n_class = n_class
        self.equal_trials = equal_trials
        self.ch_names = ch_names
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
            # logging.info("ds_len = %s", ds_len)
            return ds_len
        return len(self.subjects) * self.trials_per_subject

    def load_trial(self, trial: int) -> (np.ndarray, int):
        """
        Determines corresponding Subject of trial and loads subject's data+labels
        :return: trial data (X) and trial label (y)
        """
        # Calculate local_subject_idx + trial_idx from trial parameter
        # if Trials per Subject are not equal (e.g. BCIC, LSMR21)
        if isinstance(self.trials_per_subject, list):
            local_subject_idx = None
            trial_idx = None
            for s_idx, subject in enumerate(self.subjects):
                # Does trial come after all trials of all subjects until s_idx?
                s = sum(self.trials_per_subject[:s_idx + 1])
                if trial < s:
                    # Found correct subject
                    local_subject_idx = s_idx
                    # Trial index as subject-local
                    if s_idx > 0:
                        trials_before = sum(self.trials_per_subject[:s_idx])
                        trial_idx = trial - trials_before
                    else:
                        trial_idx = trial
                    break
        else:
            # Trials per subject are equal (e.g. PHYS)
            trial_idx = trial % self.trials_per_subject
            # determine required subject for trial
            local_subject_idx = int(trial / self.trials_per_subject)
        return self.get_global_trial(local_subject_idx, trial_idx)

    def get_global_trial(self, local_subject_idx, trial_idx) -> (np.ndarray, np.ndarray):
        """
        Returns specified Trial from preloaded_data (contains the entire Dataset)
        converts local Subject index inside the TrialsDataset into index of self.used_subjects
        :param local_subject_idx: Subject index locally in the TrialsDataset
        :param trial_idx: Index of Trial of the Subject
        :return: (Trial data array, Trial labels array)
        """
        global_subject_idx = self.used_subjects.index(self.subjects[local_subject_idx])
        data, label = self.preloaded_data[global_subject_idx][trial_idx], self.preloaded_labels[global_subject_idx][
            trial_idx]
        if label == -1:
            raise Exception(f"Invalid label found: Subject Idx {global_subject_idx} Trial Idx {trial_idx}")
        return data, label

    def __getitem__(self, trial):
        """
        Returns a single trial as Tensor with Labels
        :return: Tensor(Data), Labels
        """
        X, y = self.load_trial(trial)
        # Shape of 1 Batch (list of multiple __getitem__() calls):
        # [samples (BATCH_SIZE), 1 , Channels (len(ch_names), Samples]
        X = torch.as_tensor(X[None, ...], device=CONFIG.DEVICE, dtype=torch.float32)
        # X = TRANSFORM(X)
        return X, y
