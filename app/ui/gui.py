import logging

from PyQt5.QtWidgets import QMainWindow, QMessageBox

from app.defaults import DEFAULT_PARAMS, RunParams
from app.cli import single_run
from app.ui.long_operation import long_operation
from app.ui.app import Ui_MainWindow
from app.ui.widgets.multi_combo_box import MultiComboBox
from app.ui.widgets.progress_bar import ProgressBar
from app.ui.widgets.q_edit_logger import QPlainTextEditLogger
from app.util.progress_wrapper import TqdmProgressBar, ProgressWrapper


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, app, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.init_ui()
        self.load_default_values()
        self.app = app
        # self.btn_thebutton.pressed.connect(self.button_press)

    def init_ui(self):
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
        excludedMultiSelect.addItems(list(map(str, DEFAULT_PARAMS.available_subjects)),
                                     DEFAULT_PARAMS.available_subjects)
        self.tabGridLayout.replaceWidget(self.excludedInput, excludedMultiSelect)
        self.excludedInput.deleteLater()
        self.excludedInput = excludedMultiSelect

        # Run Button
        self.runButton.clicked.connect(lambda x: self.run('BLOB'))

        # Logger
        # Set up logging to use your widget as a handler
        log_handler = QPlainTextEditLogger(self.logTab)
        log_handler_widget = log_handler.widget
        log_handler_widget.setFixedSize(self.logListView.size())
        logging.getLogger().addHandler(log_handler)
        self.logListView.deleteLater()
        self.logListView = log_handler.widget

        # Progress Bar in statusbar
        self.progressBar = ProgressBar(self.statusbar)
        self.statusbar.addWidget(self.progressBar)

        ProgressWrapper.setParent(self)
        # Do not log tqdm except through ProgressBar
        ProgressWrapper.file = None
        ProgressWrapper.progress.connect(self.progressBar.update_progress)
        ProgressWrapper.max_val.connect(self.progressBar.init_task)

        pass

    def load_default_values(self):
        """
        Fill UI Inputs with default values
        """
        self.nameInput.setText(DEFAULT_PARAMS.name)
        self.datasetInput.setCurrentText(DEFAULT_PARAMS.dataset)
        self.nclassesInput.selectItemsByData(DEFAULT_PARAMS.n_classes)
        self.epochsInput.setValue(DEFAULT_PARAMS.epochs)
        self.batchsizeInput.setValue(DEFAULT_PARAMS.batch_size)
        self.channelsInput.selectItemsByText(DEFAULT_PARAMS.available_channels)
        self.tagInput.setText(DEFAULT_PARAMS.tag)
        self.excludedInput.selectItemsByText(DEFAULT_PARAMS.excluded)
        self.onlyfoldInput.setCurrentText(str(DEFAULT_PARAMS.only_fold) if DEFAULT_PARAMS.only_fold is not None else '')
        self.equaltrialsInput.setChecked(DEFAULT_PARAMS.equal_trials)
        self.earlystopInput.setChecked(DEFAULT_PARAMS.early_stop)
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
        return run_params

    def current_cli_args(self):
        """
        Returns RunParams from current Input values as ready-to-use CLI args
        """
        run_params = self.current_config()
        cli_params = []
        for idx, param in enumerate(vars(run_params)):
            param_val = getattr(run_params, param)
            if param_val == getattr(DEFAULT_PARAMS, param) or len(param_val) == 0:
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

    @long_operation("Run")
    def run(self):
        cli_args = ['-train'] + self.current_cli_args()
        logging.info("RUN %s", cli_args)
        single_run(cli_args)
        pass

    @long_operation("Calculation")
    def operation(self):
        return self.app.calculation(3)

    def start_task(self):
        self.mainWidget.setDisabled(True)

    def stop_task(self):
        self.mainWidget.setDisabled(False)
