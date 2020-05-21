from PyQt5.QtWidgets import (QPushButton, QHBoxLayout, QVBoxLayout,
                             QGroupBox, QLineEdit, QCheckBox, QLabel,
                             QComboBox, QListWidget, QListWidgetItem,
                             QDialog, QDialogButtonBox, QFileDialog,
                             QRadioButton, QButtonGroup, QGridLayout,
                             QTextEdit, QTableWidget, QTableWidgetItem,
                             QHeaderView, QWidget)
from PyQt5.QtGui import QWindow, QBrush, QColor
from PyQt5.QtCore import Qt

from .useful_widgets import LineEditReadOnly, global_gui_parameters, global_gui_variables
from .form_base_widget import FormBaseWidget


class FitMapsWidget(FormBaseWidget):

    def __init__(self):
        super().__init__()
        self.initialize()

    def initialize(self):

        v_spacing = global_gui_parameters["vertical_spacing_in_tabs"]

        self._setup_settings()
        self._setup_start_fitting()
        self._setup_save_results()

        vbox = QVBoxLayout()
        vbox.addWidget(self.group_settings)
        vbox.addSpacing(v_spacing)

        vbox.addWidget(self.pb_start_map_fitting)
        vbox.addSpacing(v_spacing)

        vbox.addWidget(self.group_save_results)
        self.setLayout(vbox)

    def _setup_settings(self):
        self.group_settings = QGroupBox("Options")

        self.cb_save_plots = QCheckBox("Save spectra for pixels in the selected region")
        self.le_start_row = QLineEdit("0")
        self.le_start_col = QLineEdit("0")
        self.le_end_row = QLineEdit("0")
        self.le_end_col = QLineEdit("0")
        self.cb_interpolate_with_x_y = QCheckBox("Interpolate with (x,y) coordinates")

        self.group_save_plots = QGroupBox("Save spectra for pixels in the selected region")
        self.group_save_plots.setCheckable(True)
        self.group_save_plots.setChecked(False)

        vbox = QVBoxLayout()
        grid = QGridLayout()
        grid.addWidget(QLabel("Start row:"), 0, 0)
        grid.addWidget(self.le_start_row, 0, 1)
        grid.addWidget(QLabel("column:"), 0, 2)
        grid.addWidget(self.le_start_col, 0, 3)
        grid.addWidget(QLabel("End row(*):"), 1, 0)
        grid.addWidget(self.le_end_row, 1, 1)
        grid.addWidget(QLabel("column:"), 1, 2)
        grid.addWidget(self.le_end_col, 1, 3)
        vbox.addLayout(grid)

        hbox = QHBoxLayout()
        hbox.addStretch(1)
        hbox.addWidget(QLabel("* Point is not included in the selection"))
        hbox.addStretch(1)
        vbox.addLayout(hbox)
        self.group_save_plots.setLayout(vbox)

        vbox = QVBoxLayout()
        vbox.addWidget(self.group_save_plots)
        vbox.addSpacing(5)
        vbox.addWidget(self.cb_interpolate_with_x_y)

        self.group_settings.setLayout(vbox)

    def _setup_start_fitting(self):
        self.pb_start_map_fitting = QPushButton("Start XRF Map Fitting")

    def _setup_save_results(self):
        self.group_save_results = QGroupBox("Save Results")

        self.pb_save_q_standard = QPushButton("Save Q. Standard")
        self.pb_save_to_db = QPushButton("Save to Database (Databroker)")
        self.pb_save_to_tiff = QPushButton("Save to TIFF")
        self.pb_save_to_txt = QPushButton("Save to TXT")
        self.pb_output_setup = QPushButton("Output Setup ...")

        grid = QGridLayout()
        grid.addWidget(self.pb_save_to_db, 0, 0, 1, 2)
        grid.addWidget(self.pb_save_q_standard, 1, 0)
        grid.addWidget(self.pb_save_to_tiff, 1, 1)
        grid.addWidget(self.pb_output_setup, 2, 0)
        grid.addWidget(self.pb_save_to_txt, 2, 1)

        self.group_save_results.setLayout(grid)