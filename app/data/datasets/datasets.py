import logging
from typing import Dict, Type

# Available Datasets as Dictionary (short_name: MI_DataLoader class)
from PyQt5.QtCore import QThread

from app.data.MIDataLoader import MIDataLoader
from app.data.datasets.bcic.bcic_data_loading import BCICDataLoader
from app.data.datasets.bcic.bcic_dataset import BCIC
from app.data.datasets.lsmr21.lmsr_21_dataset import LSMR21
from app.data.datasets.lsmr21.lsmr21_data_loading import LSMR21DataLoader
from app.data.datasets.phys.phys_data_loading import PHYSDataLoader
from app.data.datasets.phys.phys_dataset import PHYS
from app.scripts.lsmr21_download_convert import main

DATASETS: Dict[str, MIDataLoader] = {
    BCIC.short_name: BCICDataLoader,
    PHYS.short_name: PHYSDataLoader,
    LSMR21.short_name: LSMR21DataLoader
}


def download_dataset(ds: str, qthread: QThread = None):
    logging.info(f"Download '{ds}' Dataset...")
    if ds == PHYS.short_name:
        # Download Dataset for all subjects with n_class=4
        PHYSDataLoader.load_subjects_data(PHYS.ALL_SUBJECTS, 4,qthread=qthread)
    elif ds == BCIC.short_name:
        BCICDataLoader.download_dataset(qthread=qthread)
    elif ds == LSMR21.short_name:
        main(['--download', '--origin-path', LSMR21DataLoader.numpy_dataset_path, '--dest_path',
              LSMR21DataLoader.numpy_dataset_path],qthread=qthread)
    logging.info(f"Completed Download of '{ds}' Dataset")
