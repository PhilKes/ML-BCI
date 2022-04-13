from typing import List, Union

from app.data.datasets.bcic.bcic_dataset import BCIC
from app.data.datasets.lsmr21.lmsr_21_dataset import LSMR21
from app.data.datasets.phys.phys_dataset import PHYS


class RunParams:
    """
    All Params for the ML-BCI Command-Line-Interface
    """
    def __init__(self):
        self.n_classes: List[int] = [3]
        self.name: Union[str, None] = None
        self.tag: Union[str, None] = None
        self.device: str = 'gpu'
        self.batch_size: int = 16
        self.model: Union[str, None] = None
        self.subject: Union[int, None] = None
        self.trials_slices: int = 1
        self.tmin: Union[float, None] = None
        self.tmax: Union[float, None] = None
        self.dataset: str = PHYS.short_name
        self.epochs: int = 100
        self.ch_names: Union[List[str], None] = None
        self.ch_motorimg: Union[str, None] = None
        self.all_trials: bool = False
        self.early_stop: bool = False
        self.excluded: List[int] = []
        self.only_fold: Union[int, None] = None
        self.subjects_cs: int = 10
        self.trt: bool = False
        self.fp16: bool = False
        self.iters: int = 1
        self.all: bool = True
        self.equal_trials: bool = True

        self.available_datasets = [PHYS.short_name, BCIC.short_name, LSMR21.short_name]
        self.available_channels = PHYS.CHANNELS
        self.available_subjects = PHYS.ALL_SUBJECTS
        self.available_n_classes = [2, 3, 4]



DEFAULT_PARAMS = RunParams()
