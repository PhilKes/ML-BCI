from PyQt5 import QtCore
from PyQt5.QtCore import QThread
from PyQt5.QtWidgets import QWidget, QApplication


def run_task(main_window: QWidget, func, *args):
    class Thread(QtCore.QThread):
        def __init__(self, parent=None):
            QtCore.QThread.__init__(self, parent)

        def run(self):
            func(*args, qthread=main_window.task_thread)

    task = Thread(main_window)
    task.setTerminationEnabled(True)
    task.started.connect(main_window.start_task)
    task.finished.connect(main_window.stop_task)

    main_window.task_thread = task
    main_window.task_thread.start()


def is_thread_interrupted(qthread: QThread):
    """
    If qthread is not None check if Thread was interrupted/stopped, else return True
    """
    if qthread is not None:
        QApplication.processEvents()
        return qthread.isInterruptionRequested() or not qthread.isRunning()
    return False
