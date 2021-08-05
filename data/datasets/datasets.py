from data.datasets.bcic.bcic_data_loading import BCICDataLoader
from data.datasets.lsmr21.lsmr21_data_loading import LSMR21DataLoader
from data.datasets.phys.phys_data_loading import PHYSDataLoader

# Available Datasets as Dictionary (short_name: MI_DataLoader class)
DATASETS = {
    PHYSDataLoader.name_short: PHYSDataLoader,
    BCICDataLoader.name_short: BCICDataLoader,
    LSMR21DataLoader.name_short: LSMR21DataLoader
}
