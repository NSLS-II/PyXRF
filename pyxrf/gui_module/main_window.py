import webbrowser
from datetime import datetime

from qtpy.QtWidgets import (
    QMainWindow,
    QMessageBox,
    QLabel,
    QAction,
    QDialog,
    QVBoxLayout,
    QDialogButtonBox,
    QHBoxLayout,
    QProgressBar,
)
from qtpy.QtCore import Qt, Slot, Signal
from qtpy.QtGui import QGuiApplication, QCursor

from .central_widget import TwoPanelWidget
from .useful_widgets import global_gui_variables
from .wnd_manage_emission_lines import WndManageEmissionLines
from .wnd_compute_roi_maps import WndComputeRoiMaps
from .wnd_load_quant_calibration import WndLoadQuantitativeCalibration
from .wnd_image_wizard import WndImageWizard
from .wnd_general_settings_for_fitting import WndGeneralFittingSettings
from .wnd_detailed_fitting_params import WndDetailedFittingParamsLines, WndDetailedFittingParamsShared

import pyxrf

import logging

logger = logging.getLogger(__name__)

_main_window_geometry = {
    "initial_height": 700,
    "initial_width": 1200,
    "min_height": 700,
    "min_width": 1200,
}


class MainWindow(QMainWindow):

    signal_fitting_parameters_changed = Signal()

    def __init__(self, *, gpc):
        """
        Parameters
        ----------
        gpc: object
            reference to a class that holds references to processing classes.
        """
        super().__init__()

        self._cursor_set = False  # Indicates if temporary 'wait' cursor is set

        self.gpc = gpc
        self.gui_vars = global_gui_variables
        self.gui_vars["ref_main_window"] = self

        self.wnd_manage_emission_lines = WndManageEmissionLines(gpc=self.gpc, gui_vars=self.gui_vars)
        self.wnd_compute_roi_maps = WndComputeRoiMaps(gpc=self.gpc, gui_vars=self.gui_vars)
        self.wnd_image_wizard = WndImageWizard(gpc=self.gpc, gui_vars=self.gui_vars)
        self.wnd_load_quantitative_calibration = WndLoadQuantitativeCalibration(
            gpc=self.gpc, gui_vars=self.gui_vars
        )
        self.wnd_general_fitting_settings = WndGeneralFittingSettings(gpc=self.gpc, gui_vars=self.gui_vars)
        self.wnd_fitting_parameters_shared = WndDetailedFittingParamsShared(gpc=self.gpc, gui_vars=self.gui_vars)
        self.wnd_fitting_parameters_lines = WndDetailedFittingParamsLines(gpc=self.gpc, gui_vars=self.gui_vars)
        # Indicates that the window was closed (used mostly for testing)
        self._is_closed = False

        global_gui_variables["gui_state"]["databroker_available"] = self.gpc.is_databroker_available()

        self.initialize()

        self.central_widget.left_panel.load_data_widget.update_main_window_title.connect(self.update_window_title)

        # Set the callback for update forms with fitting parameters
        def update_fitting_parameter_forms():
            self.signal_fitting_parameters_changed.emit()

        gpc.add_parameters_changed_cb(update_fitting_parameter_forms)
        gpc.param_model.parameters_changed()

    def initialize(self):

        self.resize(_main_window_geometry["initial_width"], _main_window_geometry["initial_height"])

        self.setMinimumWidth(_main_window_geometry["min_width"])
        self.setMinimumHeight(_main_window_geometry["min_height"])

        self.setWindowTitle(self.gpc.get_window_title())

        self.central_widget = TwoPanelWidget(gpc=self.gpc, gui_vars=self.gui_vars)
        self.setCentralWidget(self.central_widget)

        # Status bar
        self.statusLabel = QLabel()
        self.statusBar().addWidget(self.statusLabel)
        self.statusProgressBar = QProgressBar()
        self.statusProgressBar.setFixedWidth(200)
        self.statusBar().addPermanentWidget(self.statusProgressBar)

        self.statusLabelDefaultText = "No data is loaded"
        self.statusLabel.setText(self.statusLabelDefaultText)

        # 'Scan Data' menu item
        self.action_read_file = QAction("&Read File...", self)
        self.action_read_file.setStatusTip("Load data from HDF5 file")
        self.action_read_file.triggered.connect(self.central_widget.left_panel.load_data_widget.pb_file.clicked)

        self.action_load_run = QAction("&Load Run...", self)
        self.action_load_run.setEnabled(self.gui_vars["gui_state"]["databroker_available"])
        self.action_load_run.setStatusTip("Load data from database (Databroker)")
        self.action_load_run.triggered.connect(self.central_widget.left_panel.load_data_widget.pb_dbase.clicked)

        self.action_view_metadata = QAction("View Metadata...", self)
        self.action_view_metadata.setEnabled(self.gpc.is_scan_metadata_available())
        self.action_view_metadata.setStatusTip("View metadata for loaded run")
        self.action_view_metadata.triggered.connect(
            self.central_widget.left_panel.load_data_widget.pb_view_metadata.clicked
        )

        # Main menu
        menubar = self.menuBar()
        # Disable native menu bar (it doesn't work on MacOS 10.15 with PyQt<=5.11)
        #   It may work with later versions of PyQt when they become available.
        menubar.setNativeMenuBar(False)
        loadData = menubar.addMenu("Scan &Data")
        loadData.addAction(self.action_read_file)
        loadData.addAction(self.action_load_run)
        loadData.addSeparator()
        loadData.addAction(self.action_view_metadata)

        # 'Fitting Model' menu item
        self.action_lines_find_automatically = QAction("Find &Automatically...", self)
        self.action_lines_find_automatically.setStatusTip("Automatically find emission lines in total spectrum")
        self.action_lines_find_automatically.triggered.connect(
            self.central_widget.left_panel.model_widget.pb_find_elines.clicked
        )

        self.action_lines_load_from_file = QAction("Load From &File...", self)
        self.action_lines_load_from_file.setStatusTip(
            "Load processing parameters, including selected emission lines, from JSON file"
        )
        self.action_lines_load_from_file.triggered.connect(
            self.central_widget.left_panel.model_widget.pb_load_elines.clicked
        )

        self.action_lines_load_quant_standard = QAction("Load &Quantitative Standards...", self)
        self.action_lines_load_quant_standard.setStatusTip(
            "Load quantitative standard. The emission lines from the standard are automatically selected"
        )
        self.action_lines_load_quant_standard.triggered.connect(
            self.central_widget.left_panel.model_widget.pb_load_qstandard.clicked
        )

        self.action_add_remove_emission_lines = QAction("&Add/Remove Emission Lines...", self)
        self.action_add_remove_emission_lines.setStatusTip("Manually add and remove emission lines")
        self.action_add_remove_emission_lines.triggered.connect(
            self.central_widget.left_panel.model_widget.pb_manage_emission_lines.clicked
        )

        self.action_save_model_params = QAction("&Save Model Parameters...", self)
        self.action_save_model_params.setStatusTip("Save model parameters to JSON file")
        self.action_save_model_params.triggered.connect(
            self.central_widget.left_panel.model_widget.pb_save_elines.clicked
        )

        self.action_add_remove_emission_lines = QAction("Start Model &Fitting", self)
        self.action_add_remove_emission_lines.setStatusTip("Run computations: start fitting for total spectrum")
        self.action_add_remove_emission_lines.triggered.connect(
            self.central_widget.left_panel.model_widget.pb_start_fitting.clicked
        )

        fittingModel = menubar.addMenu("Fitting &Model")
        emissionLines = fittingModel.addMenu("&Emission Lines")
        emissionLines.addAction(self.action_lines_find_automatically)
        emissionLines.addAction(self.action_lines_load_from_file)
        emissionLines.addAction(self.action_lines_load_quant_standard)
        fittingModel.addAction(self.action_add_remove_emission_lines)
        fittingModel.addSeparator()
        fittingModel.addAction(self.action_save_model_params)
        fittingModel.addSeparator()
        fittingModel.addAction(self.action_add_remove_emission_lines)

        # "XRF Maps" menu item
        self.action_start_xrf_map_fitting = QAction("Start XRF Map &Fitting", self)
        self.action_start_xrf_map_fitting.setStatusTip("Run computations: start fitting for XRF maps")
        self.action_start_xrf_map_fitting.triggered.connect(
            self.central_widget.left_panel.fit_maps_widget.pb_start_map_fitting.clicked
        )

        self.action_compute_rois = QAction("Compute &ROIs...", self)
        self.action_compute_rois.setStatusTip("Compute XRF Maps based on spectral ROIs")
        self.action_compute_rois.triggered.connect(
            self.central_widget.left_panel.fit_maps_widget.pb_compute_roi_maps.clicked
        )

        self.action_load_quant_calibration = QAction("&Load Quantitative Calibration...", self)
        self.action_load_quant_calibration.setStatusTip(
            "Load quantitative calibration from JSON file. Calibration is used for scaling of XRF Maps"
        )
        self.action_load_quant_calibration.triggered.connect(
            self.central_widget.left_panel.fit_maps_widget.pb_load_quant_calib.clicked
        )

        self.action_save_quant_calibration = QAction("&Save Quantitative Calibration...", self)
        self.action_save_quant_calibration.setStatusTip(
            "Save Quantitative Calibration based on XRF map of the standard sample"
        )
        self.action_save_quant_calibration.triggered.connect(
            self.central_widget.left_panel.fit_maps_widget.pb_save_q_calibration.clicked
        )

        self.action_export_to_tiff_and_txt = QAction("&Export to TIFF and TXT...", self)
        self.action_export_to_tiff_and_txt.setStatusTip("Export XRF Maps as TIFF and/or TXT files")
        self.action_export_to_tiff_and_txt.triggered.connect(
            self.central_widget.left_panel.fit_maps_widget.pb_export_to_tiff_and_txt.clicked
        )

        xrfMaps = menubar.addMenu("XRF &Maps")
        xrfMaps.addAction(self.action_start_xrf_map_fitting)
        xrfMaps.addAction(self.action_compute_rois)
        xrfMaps.addSeparator()
        xrfMaps.addAction(self.action_load_quant_calibration)
        xrfMaps.addSeparator()
        xrfMaps.addAction(self.action_save_quant_calibration)
        xrfMaps.addAction(self.action_export_to_tiff_and_txt)

        # "View" menu item
        self.action_show_matplotlib_toolbar = QAction("Show &Matplotlib Toolbar", self)
        self.action_show_matplotlib_toolbar.setCheckable(True)
        self.action_show_matplotlib_toolbar.setChecked(True)
        self.action_show_matplotlib_toolbar.setStatusTip("Show Matplotlib Toolbar on the plots")
        self.action_show_matplotlib_toolbar.toggled.connect(self.action_show_matplotlib_toolbar_toggled)

        self.action_show_widget_tooltips = QAction("Show &Tooltips", self)
        self.action_show_widget_tooltips.setCheckable(True)
        self.action_show_widget_tooltips.setChecked(self.gui_vars["show_tooltip"])
        self.action_show_widget_tooltips.setStatusTip("Show widget tooltips")
        self.action_show_widget_tooltips.toggled.connect(self.action_show_widget_tooltips_toggled)

        options = menubar.addMenu("&Options")
        options.addAction(self.action_show_widget_tooltips)
        options.addAction(self.action_show_matplotlib_toolbar)

        # "Help" menu item
        self.action_online_docs = QAction("Online &Documentation", self)
        self.action_online_docs.setStatusTip("Open online documentation in the default browser")
        self.action_online_docs.triggered.connect(self.action_online_docs_triggered)

        self.action_about = QAction("&About PyXRF", self)
        self.action_about.setStatusTip("Show information about this program")
        self.action_about.triggered.connect(self.action_about_triggered)

        help = menubar.addMenu("&Help")
        help.addAction(self.action_online_docs)
        help.addSeparator()
        help.addAction(self.action_about)

        self.update_widget_state()

        # Connect signals
        self.central_widget.left_panel.load_data_widget.update_preview_map_range.connect(
            self.central_widget.right_panel.tab_preview_plots.preview_plot_count.update_map_range
        )

        # Before loading a new file or run
        self.central_widget.left_panel.load_data_widget.signal_loading_new_run.connect(
            self.central_widget.right_panel.slot_activate_tab_preview
        )

        # Open a new file or run
        self.central_widget.left_panel.load_data_widget.signal_new_run_loaded.connect(
            self.central_widget.left_panel.slot_activate_load_data_tab
        )
        self.central_widget.left_panel.load_data_widget.signal_new_run_loaded.connect(
            self.central_widget.right_panel.slot_activate_tab_preview
        )
        self.central_widget.left_panel.load_data_widget.signal_new_run_loaded.connect(self.slot_new_run_loaded)
        self.central_widget.left_panel.load_data_widget.signal_new_run_loaded.connect(
            self.wnd_image_wizard.slot_update_table
        )
        self.central_widget.left_panel.load_data_widget.signal_new_run_loaded.connect(
            self.central_widget.right_panel.tab_plot_xrf_maps.slot_update_dataset_info
        )
        self.central_widget.left_panel.load_data_widget.signal_new_run_loaded.connect(
            self.central_widget.right_panel.tab_plot_rgb_maps.slot_update_dataset_info
        )
        self.central_widget.left_panel.load_data_widget.signal_new_run_loaded.connect(
            self.wnd_load_quantitative_calibration.update_all_data
        )
        self.central_widget.left_panel.load_data_widget.signal_new_run_loaded.connect(
            self.central_widget.left_panel.fit_maps_widget.slot_update_for_new_loaded_run
        )

        # New model is loaded or processing parameters (incident energy) was changed
        self.central_widget.left_panel.model_widget.signal_incident_energy_or_range_changed.connect(
            self.central_widget.right_panel.tab_preview_plots.preview_plot_spectrum.redraw_preview_plot
        )
        self.central_widget.left_panel.model_widget.signal_incident_energy_or_range_changed.connect(
            self.wnd_manage_emission_lines.update_widget_data
        )
        self.central_widget.left_panel.model_widget.signal_incident_energy_or_range_changed.connect(
            self.central_widget.right_panel.tab_plot_fitting_model.redraw_plot_fit
        )

        self.central_widget.left_panel.model_widget.signal_model_loaded.connect(
            self.central_widget.right_panel.tab_preview_plots.preview_plot_spectrum.redraw_preview_plot
        )
        self.central_widget.left_panel.model_widget.signal_model_loaded.connect(
            self.central_widget.right_panel.slot_activate_tab_fitting_model
        )
        self.central_widget.left_panel.model_widget.signal_model_loaded.connect(
            self.central_widget.right_panel.tab_plot_fitting_model.update_controls
        )
        self.central_widget.left_panel.model_widget.signal_model_loaded.connect(
            self.wnd_manage_emission_lines.update_widget_data
        )

        # XRF Maps dataset changed
        self.central_widget.right_panel.tab_plot_xrf_maps.signal_maps_dataset_selection_changed.connect(
            self.wnd_image_wizard.slot_update_table
        )

        self.central_widget.right_panel.tab_plot_xrf_maps.signal_maps_dataset_selection_changed.connect(
            self.central_widget.right_panel.tab_plot_rgb_maps.combo_select_dataset_update_current_index
        )
        self.central_widget.right_panel.tab_plot_rgb_maps.signal_rgb_maps_dataset_selection_changed.connect(
            self.central_widget.right_panel.tab_plot_xrf_maps.combo_select_dataset_update_current_index
        )

        self.central_widget.right_panel.tab_plot_xrf_maps.signal_maps_norm_changed.connect(
            self.wnd_image_wizard.slot_update_ranges
        )

        # Quantitative calibration changed
        self.wnd_load_quantitative_calibration.signal_quantitative_calibration_changed.connect(
            self.central_widget.right_panel.tab_plot_rgb_maps.slot_update_ranges
        )
        self.wnd_load_quantitative_calibration.signal_quantitative_calibration_changed.connect(
            self.wnd_image_wizard.slot_update_ranges
        )

        # Selected element is changed (tools for emission line selection)
        self.wnd_manage_emission_lines.signal_selected_element_changed.connect(
            self.central_widget.right_panel.tab_plot_fitting_model.slot_selection_item_changed
        )
        self.central_widget.right_panel.tab_plot_fitting_model.signal_selected_element_changed.connect(
            self.wnd_manage_emission_lines.slot_selection_item_changed
        )
        self.central_widget.right_panel.tab_plot_fitting_model.signal_add_line.connect(
            self.wnd_manage_emission_lines.pb_add_eline_clicked
        )
        self.central_widget.right_panel.tab_plot_fitting_model.signal_remove_line.connect(
            self.wnd_manage_emission_lines.pb_remove_eline_clicked
        )
        self.wnd_manage_emission_lines.signal_update_element_selection_list.connect(
            self.central_widget.right_panel.tab_plot_fitting_model.slot_update_eline_selection_list
        )
        self.wnd_manage_emission_lines.signal_update_add_remove_btn_state.connect(
            self.central_widget.right_panel.tab_plot_fitting_model.slot_update_add_remove_btn_state
        )

        self.wnd_manage_emission_lines.signal_selected_element_changed.connect(
            self.central_widget.left_panel.model_widget.slot_selection_item_changed
        )
        self.central_widget.right_panel.tab_plot_fitting_model.signal_selected_element_changed.connect(
            self.central_widget.left_panel.model_widget.slot_selection_item_changed
        )

        # Total spectrum fitting completed
        self.central_widget.left_panel.model_widget.signal_total_spectrum_fitting_completed.connect(
            self.wnd_manage_emission_lines.update_eline_table
        )
        # Total spectrum invalidated
        self.wnd_manage_emission_lines.signal_parameters_changed.connect(
            self.central_widget.left_panel.model_widget.update_fit_status
        )
        # New dataset loaded or different channel selected. Compute fit parameters.
        self.central_widget.left_panel.load_data_widget.signal_data_channel_changed.connect(
            self.central_widget.left_panel.model_widget.clear_fit_status
        )
        self.central_widget.left_panel.load_data_widget.signal_new_run_loaded.connect(
            self.central_widget.left_panel.model_widget.clear_fit_status
        )
        self.central_widget.left_panel.model_widget.signal_incident_energy_or_range_changed.connect(
            self.central_widget.left_panel.model_widget.clear_fit_status
        )

        # Update map datasets (Fitted maps)
        self.central_widget.left_panel.fit_maps_widget.signal_map_fitting_complete.connect(
            self.central_widget.right_panel.tab_plot_xrf_maps.slot_update_dataset_info
        )
        self.central_widget.left_panel.fit_maps_widget.signal_map_fitting_complete.connect(
            self.central_widget.right_panel.tab_plot_rgb_maps.slot_update_dataset_info
        )
        self.central_widget.left_panel.fit_maps_widget.signal_activate_tab_xrf_maps.connect(
            self.central_widget.right_panel.slot_activate_tab_xrf_maps
        )

        # Update map datasets (ROI maps)
        self.wnd_compute_roi_maps.signal_roi_computation_complete.connect(
            self.central_widget.right_panel.tab_plot_xrf_maps.slot_update_dataset_info
        )
        self.wnd_compute_roi_maps.signal_roi_computation_complete.connect(
            self.central_widget.right_panel.tab_plot_rgb_maps.slot_update_dataset_info
        )
        self.wnd_compute_roi_maps.signal_activate_tab_xrf_maps.connect(
            self.central_widget.right_panel.slot_activate_tab_xrf_maps
        )

        self.signal_fitting_parameters_changed.connect(self.wnd_general_fitting_settings.update_form_data)
        self.signal_fitting_parameters_changed.connect(self.wnd_fitting_parameters_shared.update_form_data)
        self.signal_fitting_parameters_changed.connect(self.wnd_fitting_parameters_lines.update_form_data)
        self.signal_fitting_parameters_changed.connect(self.wnd_manage_emission_lines.update_widget_data)

    @Slot()
    @Slot(str)
    def update_widget_state(self, condition=None):
        # Update the state of the menu bar
        state = not self.gui_vars["gui_state"]["running_computations"]
        self.menuBar().setEnabled(state)

        state_computations = self.gui_vars["gui_state"]["running_computations"]
        if state_computations:
            if not self._cursor_set:
                QGuiApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
                self._cursor_set = True
        else:
            if self._cursor_set:
                QGuiApplication.restoreOverrideCursor()
                self._cursor_set = False

        # Forward to children
        self.central_widget.update_widget_state(condition)

        # Forward the updates to open windows
        self.wnd_manage_emission_lines.update_widget_state(condition)
        self.wnd_compute_roi_maps.update_widget_state(condition)
        self.wnd_image_wizard.update_widget_state(condition)
        self.wnd_load_quantitative_calibration.update_widget_state(condition)
        self.wnd_general_fitting_settings.update_widget_state(condition)
        self.wnd_fitting_parameters_shared.update_widget_state(condition)
        self.wnd_fitting_parameters_lines.update_widget_state(condition)

    def closeEvent(self, event):
        mb_close = QMessageBox(
            QMessageBox.Question,
            "Exit",
            "Are you sure you want to EXIT the program?",
            QMessageBox.Yes | QMessageBox.No,
            parent=self,
        )
        mb_close.setDefaultButton(QMessageBox.No)

        if mb_close.exec() == QMessageBox.Yes:
            event.accept()

            self.wnd_manage_emission_lines.close()
            self.wnd_compute_roi_maps.close()
            self.wnd_image_wizard.close()
            self.wnd_load_quantitative_calibration.close()
            self.wnd_general_fitting_settings.close()
            self.wnd_fitting_parameters_shared.close()
            self.wnd_fitting_parameters_lines.close()

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
        except Exception as ex:
            logger.error(f"Error occurred while opening URL '{doc_url}' in the default browser")
            msg = f"Failed to Open Online Documentation. \n  Exception: {str(ex)}"
            msgbox = QMessageBox(QMessageBox.Critical, "Error", msg, QMessageBox.Ok, parent=self)
            msgbox.exec()

    def action_about_triggered(self):
        """
        Display 'About' dialog box
        """
        dlg = DialogAbout()
        dlg.exec()

    def action_show_matplotlib_toolbar_toggled(self, state):
        """
        Turn tooltips on or off
        """
        self.gui_vars["show_matplotlib_toolbar"] = state
        self.update_widget_state()

    def action_show_widget_tooltips_toggled(self, state):
        """
        Turn tooltips on or off
        """
        self.gui_vars["show_tooltip"] = state
        self.update_widget_state("tooltips")

    @Slot()
    def update_window_title(self):
        self.setWindowTitle(self.gpc.get_window_title())

    @Slot(bool)
    def slot_new_run_loaded(self, success):
        if success:
            # Update status bar
            file_name = self.gpc.get_loaded_file_name()
            run_id, run_uid = "", ""
            if self.gpc.is_scan_metadata_available():
                try:
                    run_id = self.gpc.get_metadata_scan_id()
                except Exception:
                    pass
                try:
                    run_uid = self.gpc.get_metadata_scan_uid()
                except Exception:
                    pass
            else:
                if self.gpc.get_current_run_id() >= 0:
                    run_id = self.gpc.get_current_run_id()
            s = ""
            if run_id:
                s += f"ID: {run_id}  "
            if run_uid:
                if run_id:
                    s += f"({run_uid})  "
                else:
                    s += f"UID: {run_uid}  "
            if file_name:
                s += f"File: '{file_name}'"
            self._set_status_bar_text(s)

            # Activate/deactivate "View Metadata ..." menu item
            if self.gpc.is_scan_metadata_available():
                self.action_view_metadata.setEnabled(True)
            else:
                self.action_view_metadata.setEnabled(False)

        else:
            self._set_status_bar_text()

    def _set_status_bar_text(self, text=None):
        if text is None:
            text = self.statusLabelDefaultText
        self.statusLabel.setText(text)


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
        text_credit_org = (
            "Data Acquisition, Management and Analysis Group\n"
            "National Synchrontron Light Source II\n"
            "Brookhaven National Laboratory"
        )

        text_copyright = f"\u00A92015\u2014{datetime.now().year} Brookhaven National Laboratory"

        label_name = QLabel(text_name)
        label_name.setStyleSheet("QLabel {font-weight: bold; font-size: 32px}")

        label_description = QLabel(text_description)
        label_description.setStyleSheet("QLabel {font-style: italic; font-size: 18px}")

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
