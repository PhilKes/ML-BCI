from typing import List, Dict, Any, Callable

from torch.utils.data import RandomSampler, DataLoader

from config import BATCH_SIZE, global_config
import numpy as np

from util.misc import print_subjects_ranges


class MIDataLoader:
    """
    Abstract Superclass for any Motor Imagery Dataset Loader
    If a new Dataset is added, a class implementing all here
    declared static methods + attributes has to be created
    see e.g. BCIC_DataLoader or PHYS_DataLoader
    """
    name: str
    name_short: str
    available_subjects: List[int]
    folds: int
    channels: List[int]
    eeg_config: Dict
    # Constructor of Dataset specific TrialsDataset subclass
    ds_class: Callable
    # Sample the trials in random order (see MIDataLoader.create_loader_from_subjects)
    sampler: Callable = RandomSampler

    # Returns Loaders of Training + Test Datasets from index splits
    # for n_class classification
    # also returns loader_valid containing validation_subjects for loss calculation if validation_subjects are present
    @classmethod
    def create_loaders_from_splits(cls, splits, validation_subjects: List[int], n_class: int, device,
                                   preloaded_data: np.ndarray = None, preloaded_labels: np.ndarray = None,
                                   bs: int = BATCH_SIZE, ch_names: List[str] = [],
                                   equal_trials: bool = True, used_subjects: List[int] = []):
        """
        Function: create_loaders_from_splits(...)

        Input parameters:
          splits: tuple of two arrays. First one contains the subject ids used for training,
                  second one contains subject ids used for testing

        Description:
          Returns Loaders of Training + Test Datasets from index splits
          for n_class classification. Optionally returns Validation Loader containing
          validation_subjects subject for loss calculation
        """
        subjects_train_idxs, subjects_test_idxs = splits
        subjects_train = [used_subjects[idx] for idx in subjects_train_idxs]
        subjects_test = [used_subjects[idx] for idx in subjects_test_idxs]
        print_subjects_ranges(subjects_train, subjects_test)
        # Only pass preloaded data for subjects, not ALL_SUBJECTS
        # have to get correct idxs for subjects
        # subjects_idxs= [ALL_SUBJECTS.index(i) for i in subjects]
        loader_valid = None
        if len(validation_subjects) > 0:
            subjects_valid_idxs = [used_subjects.index(i) for i in validation_subjects]
            loader_valid = cls.create_loader_from_subjects(validation_subjects, used_subjects, n_class, device,
                                                           preloaded_data,
                                                           preloaded_labels,
                                                           bs, ch_names, equal_trials)
        # TODO Numpy fancy slicing (indexing with list of subject_idxs)
        #  creates copy of array -> much higher memory usage every fold
        s_t = preloaded_data[subjects_train_idxs]
        # print("s_t is View of preloaded_data: ",s_t.base is preloaded_data)
        loader_train = cls.create_loader_from_subjects(subjects_train, used_subjects, n_class, device,
                                                       preloaded_data, preloaded_labels,
                                                       bs, ch_names, equal_trials)
        loader_test = cls.create_loader_from_subjects(subjects_test, used_subjects, n_class, device,
                                                      preloaded_data,
                                                      preloaded_labels,
                                                      bs, ch_names, equal_trials)
        return loader_train, loader_test, loader_valid

    # Creates DataLoader with Random Sampling from subject list
    @classmethod
    def create_loader_from_subjects(cls, subjects, used_subjects, n_class, device, preloaded_data, preloaded_labels,
                                    bs=BATCH_SIZE, ch_names=[], equal_trials=True) -> DataLoader:
        """
        Create Loaders for given subjects
        :return: Loader
        """
        trials_ds = cls.ds_class(subjects, used_subjects, n_class, device,
                                 preloaded_tuple=(preloaded_data, preloaded_labels),
                                 ch_names=ch_names, equal_trials=equal_trials)
        sampler = None if cls.sampler is None else cls.sampler(trials_ds)
        return DataLoader(trials_ds, bs, sampler=sampler, pin_memory=False)

    @classmethod
    def load_subjects_data(cls, subjects: List[int], n_class: int, ch_names: List[str] = [], equal_trials: bool = True,
                           normalize: bool = False, ignored_runs: List[int] = []):
        """
        Loads and returns n_class data of subjects
        Can lead to high memory usage depending on the Dataset
        :param subjects: Subjects to be loaded
        :param equal_trials: Load equal amount of Trials for each class?
        :param ignored_runs: Runs to ignore (MNE with PHYS)
        :return: preloaded_data, preloaded_labels of specified subjects
        """
        raise NotImplementedError('This method is not implemented!')

    @classmethod
    def create_n_class_loaders_from_subject(cls, used_subject: int, n_class: int, n_test_runs: List[int],
                                            batch_size: int, ch_names: List[str], device):
        """
        Create Train/Test Loaders for a single subject
        :param n_test_runs: Which runs are used for the Test Loader
        :return: Train Loader, Test Loader
        """
        raise NotImplementedError('This method is not implemented!')

    @classmethod
    def create_preloaded_loader(cls, subjects: List[int], n_class: int, ch_names: List[str], batch_size: int, device,
                                equal_trials: bool):
        """
        Create Loader with preloaded data for given subjects
        :return: Loader of subjects' data
        """
        print(f"Preloading Subjects [{subjects[0]}-{subjects[-1]}] Data in memory")
        preloaded_data, preloaded_labels = cls.load_subjects_data(subjects, n_class, ch_names,
                                                                  equal_trials=equal_trials)
        return cls.create_loader_from_subjects(subjects, n_class, device, preloaded_data,
                                               preloaded_labels, batch_size, equal_trials=equal_trials)

    @classmethod
    def mne_load_subject_raw(cls, subject: int, runs: List[int], ch_names: List[str] = [], notch: bool = False,
                             fmin=global_config.FREQ_FILTER_HIGHPASS, fmax=global_config.FREQ_FILTER_LOWPASS):
        raise NotImplementedError('This method is not implemented!')
