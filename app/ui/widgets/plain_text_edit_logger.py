import logging
import re

from PyQt5 import QtGui, QtCore
from PyQt5.QtWidgets import QPlainTextEdit

from app.util.misc import str_replace


class PlainTextEditLogger(logging.Handler, QtCore.QObject):
    appendPlainText = QtCore.pyqtSignal(str)

    def __init__(self, widget : QPlainTextEdit):
        logging.Handler.__init__(self)
        QtCore.QObject.__init__(self)
        self.widget = widget
        self.widget.setReadOnly(True)
        self.appendPlainText.connect(self.widget.appendPlainText)
        self.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)s %(message)s',datefmt='%Y-%m-%d %H:%M:%S'))

    def emit(self, record):
        msg = self.format(record)
        self.appendPlainText.emit(msg)

    def write(self, msg):
        formatted_msg = msg
        amt_blocks = re.search('\%\|([0-9]+) ', msg)
        if amt_blocks is not None and len(amt_blocks.regs) > 0:
            formatted_msg = str_replace(msg, '#', amt_blocks.regs[0][0]+2)
        self.appendPlainText.emit(formatted_msg)
        self.widget.moveCursor(QtGui.QTextCursor.End)
