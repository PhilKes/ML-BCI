from sys import stdout

from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtWidgets import QWidget
from tqdm import tqdm

from app.util.custom_tqdm import CustomTqdm


class __ProgressWrapper__(QObject):
    """
    Custom QObject with pyqtSignals to emitted from tqdm Progress-Bars
    """
    file = stdout

    progress = pyqtSignal(int)
    max_val = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)

    def tqdm(self, iterable, **kwargs) -> tqdm:
        """
        Instantiate new tqdm with pyqtsignals
        """
        return CustomTqdm(iterable, self.file, self.progress, self.max_val, **kwargs)


# Singleton
ProgressWrapper = __ProgressWrapper__()
# global tqdm constructor
TqdmProgressBar = ProgressWrapper.tqdm
