from typing import Dict, Type

from data.MIDataLoader import MIDataLoader
from data.datasets.bcic.bcic_data_loading import BCICDataLoader
from data.datasets.bcic.bcic_dataset import BCIC
from data.datasets.lsmr21.lmsr_21_dataset import LSMR21
from data.datasets.lsmr21.lsmr21_data_loading import LSMR21DataLoader
from data.datasets.openBCI.obci_data_loading import OpenBCIDataLoader
from data.datasets.openBCI.obci_dataset import OpenBCI
from data.datasets.phys.phys_data_loading import PHYSDataLoader
from data.datasets.phys.phys_dataset import PHYS

# Available Datasets as Dictionary (short_name: MI_DataLoader class)
DATASETS: Dict[str, Type[MIDataLoader]] = {
    BCIC.short_name: BCICDataLoader,
    PHYS.short_name: PHYSDataLoader,
    LSMR21.short_name: LSMR21DataLoader,
    OpenBCI.short_name: OpenBCIDataLoader
}
