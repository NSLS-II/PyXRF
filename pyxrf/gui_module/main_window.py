import webbrowser
from datetime import datetime

from PyQt5.QtWidgets import (QMainWindow, QMessageBox, QLabel, QAction,
                             QDialog, QVBoxLayout, QDialogButtonBox, QHBoxLayout)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt

from .central_widget import TwoPanelWidget
from .useful_widgets import global_gui_variables
from .wd_model import WndManageEmissionLines
from .wd_fit_maps import WndComputeRoiMaps, WndLoadQuantitativeCalibration
from .wd_plots_xrf_maps import WndImageWizard

import pyxrf

_main_window_geometry = {
    "initial_height": 700,
    "initial_width": 1200,
    "min_height": 700,
    "min_width": 1200,
}


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        global_gui_variables["ref_main_window"] = self
        self.wnd_manage_emission_lines = WndManageEmissionLines()
        self.wnd_compute_roi_maps = WndComputeRoiMaps()
        self.wnd_image_wizard = WndImageWizard()
        self.wnd_load_quantitative_calibration = WndLoadQuantitativeCalibration()
        # Indicates that the window was closed (used mostly for testing)
        self._is_closed = False

        self.initialize()

    def initialize(self):

        self.resize(_main_window_geometry["initial_width"],
                    _main_window_geometry["initial_height"])

        self.setMinimumWidth(_main_window_geometry["min_width"])
        self.setMinimumHeight(_main_window_geometry["min_height"])

        self.setWindowTitle("PyXRF window title")

        # Status bar
        self.statusLabel = QLabel()
        self.statusBar().addWidget(self.statusLabel)
        self.statusLabelDefaultText = \
            "Load data by selecting one of the options in 'Load Data' menu ..."
        self.statusLabel.setText(self.statusLabelDefaultText)

        # Define some actions
        action_open_file = QAction("&Open data file", self)
        action_open_file.setStatusTip('Load data from file')

        action_load_from_db = QAction("Load data from &database", self)
        action_load_from_db.setStatusTip('Load data from database')

        action_load_params = QAction("Load &processing parameters", self)
        action_load_params.setStatusTip('Load processing parameters from file')

        action_refine_params = QAction("Re&fine processing parameters", self)
        action_refine_params.setStatusTip(
            "Refine processing parameters by fitting total spectrum")

        action_gen_xrf_map = QAction("&Compute XRF map", self)
        action_gen_xrf_map.setStatusTip(
            "Compute XRF map by individually fitting of spectra for each pixel")

        action_show_matplotlib_toolbar = QAction("Show &Matplotlib toolbar", self)
        action_show_matplotlib_toolbar.setCheckable(True)
        action_show_matplotlib_toolbar.setChecked(True)

        action_online_docs = QAction("Online &documentation", self)
        action_online_docs.setStatusTip("Open online documentation in browser")
        action_online_docs.triggered.connect(self.action_online_docs_triggered)

        action_about = QAction("&About PyXRF", self)
        action_about.setStatusTip("Show information about this program")
        action_about.triggered.connect(self.action_about_triggered)

        # Main menu
        menubar = self.menuBar()
        loadData = menubar.addMenu('&Load Data')
        loadData.addAction(action_open_file)
        loadData.addAction(action_load_from_db)
        loadData.addSeparator()
        loadData.addAction(action_load_params)

        proc = menubar.addMenu('&Processing')
        proc.addAction(action_refine_params)
        proc.addAction(action_gen_xrf_map)

        view = menubar.addMenu('&View')
        view.addAction(action_show_matplotlib_toolbar)

        help = menubar.addMenu('&Help')
        help.addAction(action_online_docs)
        help.addAction(action_about)

        central_widget = TwoPanelWidget()
        self.setCentralWidget(central_widget)

    def closeEvent(self, event):

        mb_close = QMessageBox(QMessageBox.Question, "Exit",
                               "Are you sure you want to EXIT the program?",
                               QMessageBox.Yes | QMessageBox.No,
                               parent=self)
        mb_close.setDefaultButton(QMessageBox.No)

        if mb_close.exec() == QMessageBox.Yes:
            event.accept()

            self.wnd_manage_emission_lines.close()
            self.wnd_compute_roi_maps.close()
            self.wnd_image_wizard.close()
            self.wnd_load_quantitative_calibration.close()

            # This flag is used for CI tests
            self._is_closed = True
        else:
            event.ignore()

    def action_online_docs_triggered(self):
        """
        Display online documentation: open the URL in the default browser.
        """
        doc_url = "http://nsls-ii.github.io/PyXRF/"
        try:
            webbrowser.open(doc_url, autoraise=True)
        except Exception:
            print(f"Error occurred while opening URL '{doc_url}' in the default browser")

    def action_about_triggered(self):
        """
        Display 'About' dialog box
        """
        dlg = DialogAbout()
        dlg.exec()


class DialogAbout(QDialog):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("About PyXRF")
        self.setFixedSize(500, 500)

        text_name = "PyXRF"
        text_description = "X-Ray Fluorescence Analysis Tool"

        text_ver = f"Version: {pyxrf.__version__}"
        text_latest_ver = "Latest stable version:"

        text_credit = "Credits:"
        text_credit_org = ("Data Acquisition, Management and Analysis Group\n"
                           "National Synchrontron Light Source II\n"
                           "Brookhaven National Laboratory")

        text_copyright = f"\u00A92015\u2014{datetime.now().year}"\
                         " Brookhaven National Laboratory"

        label_name = QLabel(text_name)
        label_name.setStyleSheet('QLabel {font-weight: bold; font-size: 32px}')

        label_description = QLabel(text_description)
        label_description.setStyleSheet('QLabel {font-style: italic; font-size: 18px}')

        label_ver = QLabel(text_ver)
        label_latest_ver = QLabel(text_latest_ver)
        label_credit = QLabel(text_credit)
        label_org = QLabel(text_credit_org)
        label_copyright = QLabel(text_copyright)

        button_box = QDialogButtonBox(QDialogButtonBox.Close)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        vbox = QVBoxLayout()

        vbox.addStretch(1)

        hbox = QHBoxLayout()
        hbox.addStretch(1)
        hbox.addWidget(label_name)
        hbox.addStretch(1)
        vbox.addLayout(hbox)

        hbox = QHBoxLayout()
        hbox.addStretch(1)
        hbox.addWidget(label_description)
        hbox.addStretch(1)
        vbox.addLayout(hbox)

        vbox.addStretch(1)

        hbox = QHBoxLayout()
        hbox.addSpacing(30)
        hbox.addWidget(label_ver)
        hbox.addStretch(1)
        vbox.addLayout(hbox)

        hbox = QHBoxLayout()
        hbox.addSpacing(30)
        hbox.addWidget(label_latest_ver)
        hbox.addStretch(1)
        vbox.addLayout(hbox)

        vbox.addStretch(1)

        hbox = QHBoxLayout()
        hbox.addSpacing(30)
        hbox.addWidget(label_credit, 0, Qt.AlignTop)
        hbox.addWidget(label_org, 0, Qt.AlignTop)
        hbox.addStretch(1)
        vbox.addLayout(hbox)

        vbox.addSpacing(20)

        hbox = QHBoxLayout()
        hbox.addSpacing(30)
        hbox.addWidget(label_copyright)
        hbox.addStretch(1)
        vbox.addLayout(hbox)

        vbox.addSpacing(20)

        vbox.addWidget(button_box)

        self.setLayout(vbox)
