from config import BATCH_SIZE, global_config


class MI_DataLoader:
    """
    Abstract Superclass for any Motor Imagery Dataset Loader
    If a new Dataset is added, a class implementing all here
    declared static methods + attributes has to be created
    see e.g. BCIC_DataLoader or PHYS_DataLoader
    """
    name = None
    name_short = None
    available_subjects = None
    folds = None
    channels = None
    eeg_config = None

    @staticmethod
    def load_subjects_data(subjects, n_class, ch_names=[], equal_trials=True,
                           normalize=False, ignored_runs=[]):
        """
        Loads and returns n_class data of subjects
        :param subjects: Subjects to be loaded
        :param equal_trials: Load equal amount of Trials for each class?
        :param ignored_runs: Runs to ignore (MNE with PHYS)
        :return: preloaded_data, preloaded_labels of specified subjects
        """
        raise NotImplementedError('This method is not implemented!')

    @staticmethod
    def create_loaders_from_splits(splits, validation_subjects, n_class, device, preloaded_data=None,
                                   preloaded_labels=None, bs=BATCH_SIZE, ch_names=[],
                                   equal_trials=True, used_subjects=[]):
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
        raise NotImplementedError('This method is not implemented!')

    @staticmethod
    def create_n_class_loaders_from_subject(used_subject, n_class, n_test_runs, batch_size,
                                            ch_names, device):
        """
        Create Train/Test Loaders for a single subject
        :param n_test_runs: Which runs are used for the Test Loader
        :return: Train Loader, Test Loader
        """
        raise NotImplementedError('This method is not implemented!')

    @staticmethod
    def create_preloaded_loader(subjects, n_class, ch_names, batch_size, device,
                                equal_trials):
        """
        Create Loader with preloaded data for given subjects
        :return: Loader of subjects' data
        """
        raise NotImplementedError('This method is not implemented!')

    @staticmethod
    def mne_load_subject_raw(subject, runs, ch_names=[], notch=False,
                             fmin=global_config.FREQ_FILTER_HIGHPASS, fmax=global_config.FREQ_FILTER_LOWPASS):
        raise NotImplementedError('This method is not implemented!')
