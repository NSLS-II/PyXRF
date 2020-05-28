from PyQt5.QtWidgets import (QMainWindow, QMessageBox, QLabel, QAction)
# from PyQt5.QtCore import Qt

from .central_widget import TwoPanelWidget
from .useful_widgets import global_gui_variables
from .wd_model import WndManageEmissionLines
from .wd_fit_maps import WndComputeRoiMaps, WndLoadQuantitativeCalibration
from .wd_plots_xrf_maps import WndImageWizard

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

        action_show_about = QAction("&About", self)
        action_show_about.setStatusTip("Show information about this program")

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
        help.addAction(action_show_about)

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
