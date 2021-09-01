from typing import List

from config import EEGConfig


class DSConstants(object):
    """
    Class for global default Constants of a Dataset
    Should be initialized once in a Subclass (e.g. BCICConstants)
    and never modified elsewhere in the code
    """

    def __init__(self):
        self.name: str = ''
        self.short_name: str = ''
        self.ALL_SUBJECTS: List[int] = []
        self.runs: List[int] = []
        self.trials_per_class_per_sr: int = -1
        self.cv_folds: int = -1
        self.CHANNELS: List[str] = []
        self.CONFIG: EEGConfig = EEGConfig()
        # Maximum possible Trial TMAX/TMIN
        self.TRIAL_TMAX = -1
        self.TRIAL_TMIN = -1
        self.REST_PHASES = []
