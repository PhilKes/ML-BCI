import logging
from contextlib import contextmanager, ExitStack
from functools import wraps

from PyQt5 import QtCore
from PyQt5.QtCore import QEventLoop, pyqtSlot, QThread
from PyQt5.QtWidgets import QWidget, QApplication

import app.ui.main_window


def run_task(main_window: QWidget, func, *args):
    result, exception = None, None

    class Thread(QtCore.QThread):
        def __init__(self, parent=None):
            QtCore.QThread.__init__(self, parent)

        def run(self):
            nonlocal result, exception
            result = func(*args, qthread=main_window.task_thread)


    task = Thread(main_window)
    task.setTerminationEnabled(True)
    task.started.connect(main_window.start_task)
    task.finished.connect(main_window.stop_task)

    main_window.task_thread = task
    main_window.task_thread.start()
    if exception is not None:
        raise exception

    return result


def is_thread_running(qthread: QThread):
    """
    If qthread is not None check if Thread was interrupted/stopped, else return True
    """
    if qthread is not None:
        QApplication.processEvents()
        return qthread.isInterruptionRequested() or not qthread.isRunning()
    return True


def long_operation(window_title=" ", label_text="Processing...", disable=True, is_slot=True):
    """
    Shows an infinite progress bar and does the actual work inside a QThread. This keeps the GUI responsive while
    showing that some operation is currently running.

    :param window_title: Window title for the progress bar
    :param label_text: Text for the progress bar
    :param disable:  This temporarily disables the parent QWidget while the operation is going on. Only has effect
                     if is_qt_method = True
    :param is_qt_method: If set to true, the decorated function must be a QWidget class method. It is then used
                         as the parent of the progress bar window.
    :param is_slot: This decorator additionally makes the decorated function a pyqtSlot. If this is not wanted or
                    needed, set to False.
    :return: function decorator
    """

    def wrapper(func):
        if is_slot:
            func = pyqtSlot()(func)

        @wraps(func)
        def decorator(*args, **kwargs):
            main_window: app.ui.main_window.MainWindow = args[0]
            result, exception = None, None
            loop = QEventLoop()

            class Thread(QtCore.QThread):
                def __init__(self, parent=None):
                    QtCore.QThread.__init__(self, parent)

                def run(self):
                    nonlocal result, exception
                    try:
                        result = func(*args[:-1], **kwargs)
                    except Exception as e:
                        exception = e

            task = Thread(main_window)
            task.setTerminationEnabled(True)
            task.started.connect(main_window.start_task)
            task.finished.connect(loop.exit)
            task.finished.connect(main_window.stop_task)

            nonlocal disable
            disable = disable and main_window is not None
            main_window.task_thread = task
            main_window.task_thread.start()
            loop.exec()
            if exception is not None:
                raise exception

            return result

        return decorator

    return wrapper


@contextmanager
def disabled(qobjs, state=True, enable=True):
    """
    Temporarily enables/disables the passed QWidget.

    :param qobj: QWidget to disable
    :param state: Target "disable" state (True = temporarily disabled)
    :param enable: Completely disables what this does.
    :param except_objs: These objects are set to the opposite state.
    :return: None
    """
    if not enable:
        yield
        return
    original_states = []
    for qobj in qobjs:
        original_states.append(not qobj.isEnabled())
        qobj.setDisabled(state)
    with ExitStack() as stack:
        for qobj in qobjs:
            stack.enter_context(disabled([qobj], state=not state))
        yield
    for qobj, original_state in zip(qobjs, original_states):
        qobj.setDisabled(original_state)
