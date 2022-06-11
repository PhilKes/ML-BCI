from PyQt5.QtCore import pyqtSlot, pyqtSignal
from PyQt5.QtWidgets import QProgressBar


class ProgressBar(QProgressBar):
    """
    Custom QProgressBar with pyqtSlots
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFormat("%v/%m")
        self.setDisabled(True)

    @pyqtSlot(int)
    def update_progress(self, progress):
        self.setValue(progress)

    @pyqtSlot(int)
    def init_task(self, max_val: int):
        self.setDisabled(False)
        self.setMaximum(max_val)
        self.setMinimum(0)
        self.setValue(0)

