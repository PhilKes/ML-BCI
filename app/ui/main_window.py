import logging
from typing import Union

from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtCore import QThread
from PyQt5.QtWidgets import QMainWindow, QMessageBox, QStyle
from superqt import QLabeledRangeSlider
from superqt.sliders._labeled import LabelPosition

import app.cli.cli
from app.data.MIDataLoader import MIDataLoader
from app.data.datasets.datasets import DATASETS, download_dataset
from app.defaults import DEFAULT_PARAMS, RunParams
from app.ui.app import Ui_MainWindow
import app.ui.long_operation as LongOperation
from app.ui.widgets.multi_combo_box import MultiComboBox
from app.ui.widgets.plain_text_edit_logger import PlainTextEditLogger
from app.ui.widgets.progress_bar import ProgressBar
from app.util.progress_wrapper import ProgressWrapper


class MainWindow(QMainWindow, Ui_MainWindow):

    def __init__(self, app, parent=None):
        super().__init__(parent)
        self.setWindowIcon(QtGui.QIcon('../../ml-bci-logo.png'))
        self.setWindowTitle('ML-BCI')
        self.dataset: MIDataLoader = DATASETS[DEFAULT_PARAMS.available_datasets[0]]
        self.setupUi(self)
        self.init_ui()
        self.load_default_values()
        self.app = app
        self.task_active = False
        self.task_thread: Union[None, QThread] = None
        # self.btn_thebutton.pressed.connect(self.button_press)

    def init_ui(self):
        # Dataset Selection
        self.datasetInput.addItems(DEFAULT_PARAMS.available_datasets)
        self.datasetInput.currentIndexChanged.connect(self.dataset_changed)

        # N-Classes Selection
        nclassesMultiSelect = MultiComboBox()
        nclassesMultiSelect.setObjectName("nclassesInput")
        nclassesMultiSelect.addItems(list(map(str, DEFAULT_PARAMS.available_n_classes)),
                                     DEFAULT_PARAMS.available_n_classes)
        self.tabGridLayout.replaceWidget(self.nclassesInput, nclassesMultiSelect)
        self.nclassesInput.deleteLater()
        self.nclassesInput = nclassesMultiSelect

        # Channel Selection
        channelMultiSelect = MultiComboBox()
        channelMultiSelect.setObjectName("channelsInput")
        channelMultiSelect.addItems(DEFAULT_PARAMS.available_channels)
        self.tabGridLayout.replaceWidget(self.channelsInput, channelMultiSelect)
        self.channelsInput.deleteLater()
        self.channelsInput = channelMultiSelect

        # Excluded Subjects Selection
        excludedMultiSelect = MultiComboBox()
        excludedMultiSelect.setObjectName("excludedInput")
        excludedMultiSelect.addItems(list(map(str, self.dataset.CONSTANTS.ALL_SUBJECTS)),
                                     self.dataset.CONSTANTS.ALL_SUBJECTS)
        self.tabGridLayout.replaceWidget(self.excludedInput, excludedMultiSelect)
        self.excludedInput.deleteLater()
        self.excludedInput = excludedMultiSelect

        # Run Button
        self.runButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.runButton.clicked.connect(lambda x: self.run_button_click())

        # Logger
        # Set up logging to use your widget as a handler
        log_handler = PlainTextEditLogger(self.logTextEdit)
        logging.getLogger().addHandler(log_handler)

        # Progress Bar in statusbar
        self.progressBar = ProgressBar(self.statusbar)
        self.statusbar.addWidget(self.progressBar)

        ProgressWrapper.setParent(self)
        # Do not log tqdm except through ProgressBar
        ProgressWrapper.file = None
        ProgressWrapper.progress.connect(self.progressBar.update_progress)
        ProgressWrapper.max_val.connect(self.progressBar.init_task)

        # TMIN;TMAX Slider
        tminTmaxSlider = QLabeledRangeSlider()
        # tminTmaxSlider.setFixedSize(self.intervalSlider.size())
        tminTmaxSlider.setMinimum(self.dataset.CONSTANTS.TRIAL_TMIN)
        tminTmaxSlider.setMaximum(self.dataset.CONSTANTS.TRIAL_TMAX)
        tminTmaxSlider.setParent(self)
        tminTmaxSlider.setOrientation(QtCore.Qt.Horizontal)
        tminTmaxSlider.setValue([self.dataset.CONSTANTS.CONFIG.TMIN, self.dataset.CONSTANTS.CONFIG.TMAX])
        tminTmaxSlider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        tminTmaxSlider.setTickInterval(0.5)
        tminTmaxSlider.setHandleLabelPosition(LabelPosition.LabelsLeft)
        self.gridLayout.replaceWidget(self.intervalSlider, tminTmaxSlider)
        self.intervalSlider.deleteLater()
        self.intervalSlider = tminTmaxSlider

        # Setup Dataset Menu ACtions
        self.actionPHYS.triggered.connect(lambda: self.dataset_action('PHYS'))
        self.actionBCIC.triggered.connect(lambda: self.dataset_action('BCIC'))
        self.actionLSMR21.triggered.connect(lambda: self.dataset_action('LSMR21'))

        pass

    def load_default_values(self):
        """
        Fill UI Inputs with default values
        """
        self.datasetInput.setCurrentIndex(0)
        self.nameInput.setText(DEFAULT_PARAMS.name)
        self.datasetInput.setCurrentText(DEFAULT_PARAMS.dataset)
        self.nclassesInput.selectItemsByData(DEFAULT_PARAMS.n_classes)
        self.epochsInput.setValue(DEFAULT_PARAMS.epochs)
        self.batchsizeInput.setValue(DEFAULT_PARAMS.batch_size)
        self.tagInput.setText(DEFAULT_PARAMS.tag)
        self.onlyfoldInput.setCurrentText(str(DEFAULT_PARAMS.only_fold) if DEFAULT_PARAMS.only_fold is not None else '')
        self.equaltrialsInput.setChecked(DEFAULT_PARAMS.equal_trials)
        self.earlystopInput.setChecked(DEFAULT_PARAMS.early_stop)

        self.update_dataset_config()
        pass

    def current_config(self):
        """
        Get RunParams from current Input values
        """
        run_params = RunParams()
        run_params.name = self.nameInput.text()
        run_params.dataset = self.datasetInput.currentText()
        run_params.n_classes = self.nclassesInput.selectedItemsData()
        run_params.epochs = self.epochsInput.value()
        run_params.batch_size = self.batchsizeInput.value()
        run_params.ch_names = self.channelsInput.selectedItemsTexts()
        run_params.tag = self.tagInput.text()
        run_params.excluded = self.excludedInput.selectedItemsData()
        run_params.only_fold = self.onlyfoldInput.currentText()
        run_params.equal_trials = self.equaltrialsInput.isChecked()
        run_params.early_stop = self.earlystopInput.isChecked()
        run_params.tmin = self.intervalSlider.value()[0]
        run_params.tmax = self.intervalSlider.value()[1]
        return run_params

    def current_cli_args(self):
        """
        Returns RunParams from current Input values as ready-to-use CLI args
        """
        run_params = self.current_config()
        cli_params = []
        for idx, param in enumerate(vars(run_params)):
            param_val = getattr(run_params, param)
            if param_val == getattr(DEFAULT_PARAMS, param) or (isinstance(param_val, str) and len(param_val) == 0):
                continue
            cli_params.append(f"--{param}")
            if not isinstance(param_val, bool):
                if isinstance(param_val, list):
                    cli_params += list(map(str, param_val))
                else:
                    cli_params.append(str(param_val))
        return cli_params

    def button_press(self):
        text = self.operation()
        QMessageBox.information(self, "Message Box", text)

    def run_button_click(self):
        if self.task_active:
            if self.task_thread is not None:
                self.task_thread.requestInterruption()
                # self.task_thread.terminate()
            pass
        else:
            self.run()
        pass

    # @long_operation("Run")
    def run(self):
        cli_args = ['-train'] + self.current_cli_args()
        logging.info("RUN %s", cli_args)
        LongOperation.run_task(self, app.cli.cli.single_run, cli_args)

    def start_task(self):
        self.task_active = True
        self.runButton.setText("Stop")
        self.runButton.setIcon(self.style().standardIcon(QStyle.SP_MediaStop))
        self.mainWidget.setDisabled(True)

    def stop_task(self):
        self.task_active = False
        self.task_thread.deleteLater()
        self.task_thread = None
        self.runButton.setText("Run")
        self.runButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.mainWidget.setDisabled(False)
        self.progressBar.reset()

    def dataset_changed(self, idx):
        print("Changed Dataset to", self.datasetInput.currentText(), idx)
        self.dataset = DATASETS[self.datasetInput.currentText()]
        self.update_dataset_config()

    def update_dataset_config(self):
        # Update available Channels
        self.channelsInput.clear()
        self.channelsInput.addItems(self.dataset.CONSTANTS.CHANNELS)
        self.channelsInput.selectItemsByTexts(self.dataset.CONSTANTS.CHANNELS)
        self.excludedInput.clear()
        self.excludedInput.addItems(list(map(str, self.dataset.CONSTANTS.ALL_SUBJECTS)),
                                    self.dataset.CONSTANTS.ALL_SUBJECTS)
        self.excludedInput.setCurrentIndex(-1)

        # Update Time Interval
        self.intervalSlider.setMinimum(self.dataset.CONSTANTS.TRIAL_TMIN)
        self.intervalSlider.setMaximum(self.dataset.CONSTANTS.TRIAL_TMAX)
        self.intervalSlider.setValue([self.dataset.CONSTANTS.CONFIG.TMIN, self.dataset.CONSTANTS.CONFIG.TMAX])

        # self.tminLabel.setText(str(self.intervalSlider.minimum()))
        # self.tmaxLabel.setText(str(self.intervalSlider.maximum()))


    def dataset_action(self, ds: str):
        LongOperation.run_task(self, download_dataset, ds)