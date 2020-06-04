import webbrowser
from datetime import datetime

from PyQt5.QtWidgets import (QMainWindow, QMessageBox, QLabel, QAction,
                             QDialog, QVBoxLayout, QDialogButtonBox, QHBoxLayout,
                             QProgressBar)
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

        self.central_widget = TwoPanelWidget()
        self.setCentralWidget(self.central_widget)

        # Status bar
        self.statusLabel = QLabel()
        self.statusBar().addWidget(self.statusLabel)
        self.statusProgressBar = QProgressBar()
        self.statusProgressBar.setFixedWidth(200)
        self.statusBar().addPermanentWidget(self.statusProgressBar)

        self.statusLabelDefaultText = \
            "Load data by selecting one of the options in 'Load Data' menu ..."
        self.statusLabel.setText(self.statusLabelDefaultText)

        # 'Scan Data' menu item
        action_read_file = QAction("&Read File...", self)
        action_read_file.setStatusTip('Load data from HDF5 file')
        action_read_file.triggered.connect(
            self.central_widget.left_panel.load_data_widget.pb_file.clicked)

        action_load_run = QAction("&Load Run...", self)
        action_load_run.setStatusTip('Load data from database (Databroker)')
        action_load_run.triggered.connect(
            self.central_widget.left_panel.load_data_widget.pb_dbase.clicked)

        action_view_metadata = QAction("View Metadata...", self)
        action_view_metadata.setStatusTip('View metadata for loaded run')
        action_view_metadata.triggered.connect(
            self.central_widget.left_panel.load_data_widget.pb_view_metadata.clicked)

        # Main menu
        menubar = self.menuBar()
        loadData = menubar.addMenu('Scan &Data')
        loadData.addAction(action_read_file)
        loadData.addAction(action_load_run)
        loadData.addSeparator()
        loadData.addAction(action_view_metadata)

        # 'Fitting Model' menu item
        action_lines_find_automatically = QAction("Find &Automatically...", self)
        action_lines_find_automatically.setStatusTip(
            "Automatically find emission lines in total spectrum")
        action_lines_find_automatically.triggered.connect(
            self.central_widget.left_panel.model_widget.pb_find_elines.clicked)

        action_lines_load_from_file = QAction("Load From &File...", self)
        action_lines_load_from_file.setStatusTip(
            "Load processing parameters, including selected emission lines, from JSON file")
        action_lines_load_from_file.triggered.connect(
            self.central_widget.left_panel.model_widget.pb_load_elines.clicked)

        action_lines_load_quant_standard = QAction("Load &Quantitative Standards...", self)
        action_lines_load_quant_standard.setStatusTip(
            "Load quantitative standard. The emission lines from the standard are automatically selected")
        action_lines_load_quant_standard.triggered.connect(
            self.central_widget.left_panel.model_widget.pb_load_qstandard.clicked)

        action_add_remove_emission_lines = QAction("&Add/Remove Emission Lines...", self)
        action_add_remove_emission_lines.setStatusTip(
            "Manually add and remove emission lines")
        action_add_remove_emission_lines.triggered.connect(
            self.central_widget.left_panel.model_widget.pb_manage_emission_lines.clicked)

        action_save_model_params = QAction("&Save Model Parameters...", self)
        action_save_model_params.setStatusTip(
            "Save model parameters to JSON file")
        action_save_model_params.triggered.connect(
            self.central_widget.left_panel.model_widget.pb_save_elines.clicked)

        action_add_remove_emission_lines = QAction("Start Model &Fitting", self)
        action_add_remove_emission_lines.setStatusTip(
            "Run computations: start fitting for total spectrum")
        action_add_remove_emission_lines.triggered.connect(
            self.central_widget.left_panel.model_widget.pb_start_fitting.clicked)

        fittingModel = menubar.addMenu('Scan &Data')
        emissionLines = fittingModel.addMenu("&Emission Lines")
        emissionLines.addAction(action_lines_find_automatically)
        emissionLines.addAction(action_lines_load_from_file)
        emissionLines.addAction(action_lines_load_quant_standard)
        fittingModel.addAction(action_add_remove_emission_lines)
        fittingModel.addSeparator()
        fittingModel.addAction(action_save_model_params)
        fittingModel.addSeparator()
        fittingModel.addAction(action_add_remove_emission_lines)

        # "XRF Maps" menu item
        action_start_xrf_map_fitting = QAction("Start XRF Map &Fitting", self)
        action_start_xrf_map_fitting.setStatusTip(
            "Run computations: start fitting for XRF maps")
        action_start_xrf_map_fitting.triggered.connect(
            self.central_widget.left_panel.fit_maps_widget.pb_start_map_fitting.clicked)

        action_compute_rois = QAction("Compute &ROIs...", self)
        action_compute_rois.setStatusTip(
            "Compute XRF Maps based on spectral ROIs")
        action_compute_rois.triggered.connect(
            self.central_widget.left_panel.fit_maps_widget.pb_compute_roi_maps.clicked)

        action_load_quant_calibration = QAction("&Load Quantitative Calibration...", self)
        action_load_quant_calibration.setStatusTip(
            "Load quantitative calibration from JSON file. Calibration is used for scaling of XRF Maps")
        action_load_quant_calibration.triggered.connect(
            self.central_widget.left_panel.fit_maps_widget.pb_load_quant_calib.clicked)

        action_save_quant_calibration = QAction("&Save Quantitative Calibration...", self)
        action_save_quant_calibration.setStatusTip(
            "Save Quantitative Calibration based on XRF map of the standard sample")
        action_save_quant_calibration.triggered.connect(
            self.central_widget.left_panel.fit_maps_widget.pb_save_q_calibration.clicked)

        action_export_to_tiff_and_txt = QAction("&Export to TIFF and TXT...", self)
        action_export_to_tiff_and_txt.setStatusTip(
            "Export XRF Maps as TIFF and/or TXT files")
        action_export_to_tiff_and_txt.triggered.connect(
            self.central_widget.left_panel.fit_maps_widget.pb_export_to_tiff_and_txt.clicked)

        xrfMaps = menubar.addMenu('XRF &Maps')
        xrfMaps.addAction(action_start_xrf_map_fitting)
        xrfMaps.addAction(action_compute_rois)
        xrfMaps.addSeparator()
        xrfMaps.addAction(action_load_quant_calibration)
        xrfMaps.addSeparator()
        xrfMaps.addAction(action_save_quant_calibration)
        xrfMaps.addAction(action_export_to_tiff_and_txt)

        # "View" menu item
        action_show_matplotlib_toolbar = QAction("Show &Matplotlib toolbar", self)
        action_show_matplotlib_toolbar.setCheckable(True)
        action_show_matplotlib_toolbar.setChecked(True)
        action_show_matplotlib_toolbar.setStatusTip(
            "Show Matplotlib Toolbar on the plots")

        view = menubar.addMenu('&View')
        view.addAction(action_show_matplotlib_toolbar)

        # "Help" menu item
        action_online_docs = QAction("Online &Documentation", self)
        action_online_docs.setStatusTip("Open online documentation in the default browser")
        action_online_docs.triggered.connect(self.action_online_docs_triggered)

        action_about = QAction("&About PyXRF", self)
        action_about.setStatusTip("Show information about this program")
        action_about.triggered.connect(self.action_about_triggered)

        help = menubar.addMenu('&Help')
        help.addAction(action_online_docs)
        help.addSeparator()
        help.addAction(action_about)

    def update_widget_state(self):
        self.central_widget.update_widget_state()
        # Update the state of the menu bar
        state = not global_gui_variables["gui_state"]["running_computations"]
        self.menuBar().setEnabled(state)

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
