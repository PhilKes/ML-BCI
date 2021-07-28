from data.datasets.bcic.bcic_data_loading import BCICDataloader
from data.datasets.lsmr21.lsmr21_data_loading import LSMR21DataLoader
from data.datasets.phys.phys_data_loading import PHYSDataLoader
from data.datasets.openBCI.openBCI_data_loading import OpenBCIDataLoader

# Available Datasets as Dictionary (short_name: MI_DataLoader class)
DATASETS = {
    PHYSDataLoader.name_short: PHYSDataLoader,
    BCICDataloader.name_short: BCICDataloader,
    LSMR21DataLoader.name_short: LSMR21DataLoader,
    OpenBCIDataLoader.name_short: OpenBCIDataLoader

}
