import logging

from PyQt5 import QtGui
from PyQt5.QtWidgets import QPlainTextEdit


class PlainTextEditLogger(logging.StreamHandler):
    def __init__(self, parent):
        super(PlainTextEditLogger, self).__init__()

        self.widget = QPlainTextEdit(parent)
        self.widget.setReadOnly(True)

    def emit(self, record):
        msg = self.format(record)
        self.widget.appendPlainText(msg)
        self.widget.moveCursor(QtGui.QTextCursor.End)

    def write(self, msg):
        formatted_msg = msg
        # amt_blocks = re.search('\%\|([0-9]+) ', msg)
        # if amt_blocks is not None and len(amt_blocks.regs) > 0:
        #     formatted_msg = str_replace(msg, '#', amt_blocks.regs[0][0]+2)

        self.widget.appendPlainText(formatted_msg)
        self.widget.moveCursor(QtGui.QTextCursor.End)
