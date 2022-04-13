from typing import Dict, Type


# Available Datasets as Dictionary (short_name: MI_DataLoader class)
from app.data.MIDataLoader import MIDataLoader
from app.data.datasets.bcic.bcic_data_loading import BCICDataLoader
from app.data.datasets.bcic.bcic_dataset import BCIC
from app.data.datasets.lsmr21.lmsr_21_dataset import LSMR21
from app.data.datasets.lsmr21.lsmr21_data_loading import LSMR21DataLoader
from app.data.datasets.phys.phys_data_loading import PHYSDataLoader
from app.data.datasets.phys.phys_dataset import PHYS

DATASETS: Dict[str, MIDataLoader] = {
    BCIC.short_name: BCICDataLoader,
    PHYS.short_name: PHYSDataLoader,
    LSMR21.short_name: LSMR21DataLoader
}
