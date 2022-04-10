from sys import stdout
from typing import Union

from PyQt5.QtCore import pyqtSignal, QObject
from tqdm import tqdm


class CustomTqdm(tqdm):
    """
    tqdm that emits to pyqtsignals
    """
    progress: pyqtSignal
    max_val: pyqtSignal

    def __init__(self, iterable, file, progress, max_val, **kwargs):
        super().__init__(iterable, **kwargs, file=file)
        self.progress = progress
        self.max_val = max_val
        # Set maximum Progress value
        self.max_val.emit(self.total)

    def update(self, n=1):
        val = super().update(n)
        # Update ProgressBar value
        if self.total is not None:
            self.progress.emit(self.n)

        return val