from data.datasets.bcic.bcic_data_loading import BCIC_Dataloader
from data.datasets.phys.phys_data_loading import PHYS_DataLoader

# Available Datasets
DS_DICT = {
    PHYS_DataLoader.name_short: PHYS_DataLoader,
    BCIC_Dataloader.name_short: BCIC_Dataloader
}
