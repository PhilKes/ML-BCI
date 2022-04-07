import argparse
import logging
import sys
import traceback

from app.main import single_run
from app.util.cmd_parser import create_parser, parse_and_check


def new_excepthook(type, value, tb):
    # by default, Qt does not seem to output any errors, this prevents that
    traceback.print_exception(type, value, tb)


sys.excepthook = new_excepthook


def main(argv=sys.argv[1:]):
    parser = create_parser()
    args = parse_and_check(parser, argv, check=False)
    init()

    if args.no_gui:
        # Run CLI
        logging.info("Running CLI")
        single_run()
    else:
        # Init and Show GUI
        logging.info("Running GUI")
        from PyQt5.QtWidgets import QApplication
        from app.gui import MainWindow
        qapp = QApplication(sys.argv)
        gui = MainWindow(None)
        gui.show()
        sys.exit(qapp.exec_())


def init():
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    root_logger.addHandler(handler)


if __name__ == '__main__':
    main()
