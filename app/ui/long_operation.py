import logging
from contextlib import contextmanager, ExitStack
from functools import wraps

from PyQt5 import QtCore
from PyQt5.QtCore import QEventLoop, pyqtSlot

import app.ui.gui


def long_operation(window_title=" ", label_text="Processing...", disable=True, is_qt_method=True, is_slot=True):
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
            logging.info("ARGS %s", args)
            qobj: app.ui.gui.MainWindow = args[0] if is_qt_method else None
            result, exception = None, None
            loop = QEventLoop()

            class Thread(QtCore.QThread):
                def run(self):
                    nonlocal result, exception
                    try:
                        result = func(*args[:-1], **kwargs)
                    except Exception as e:
                        exception = e

            task = Thread()
            task.started.connect(qobj.start_task)
            task.finished.connect(loop.exit)
            task.finished.connect(qobj.stop_task)

            nonlocal disable
            disable = disable and qobj is not None

            task.start()
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
