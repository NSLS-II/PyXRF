import os
import textwrap

from qtpy.QtWidgets import (QPushButton, QHBoxLayout, QVBoxLayout, QGroupBox,
                            QCheckBox, QLabel, QComboBox, QDialog, QDialogButtonBox, QFileDialog,
                            QRadioButton, QButtonGroup, QGridLayout, QTextEdit, QTableWidget,
                            QTableWidgetItem, QHeaderView, QWidget, QScrollArea,
                            QTabWidget, QFrame, QMessageBox)
from qtpy.QtGui import QBrush, QColor, QDoubleValidator
from qtpy.QtCore import Qt, Slot, Signal, QThreadPool, QRunnable

from .useful_widgets import (LineEditReadOnly, global_gui_parameters, get_background_css,
                             PushButtonMinimumWidth, SecondaryWindow, set_tooltip, LineEditExtended,
                             PushButtonNamed, CheckBoxNamed, RangeManager,
                             IntValidatorStrict, DoubleValidatorStrict)
from .form_base_widget import FormBaseWidget

import logging
logger = logging.getLogger(__name__)


class FitMapsWidget(FormBaseWidget):

    # Signal that is sent (to main window) to update global state of the program
    update_global_state = Signal()
    computations_complete = Signal(object)

    signal_map_fitting_complete = Signal()
    signal_activate_tab_xrf_maps = Signal()

    def __init__(self, *, gpc, gui_vars):
        super().__init__()

        # Global processing classes
        self.gpc = gpc
        # Global GUI variables (used for control of GUI state)
        self.gui_vars = gui_vars

        # Reference to the main window. The main window will hold
        #   references to all non-modal windows that could be opened
        #   from multiple places in the program.
        self.ref_main_window = self.gui_vars["ref_main_window"]

        self.update_global_state.connect(self.ref_main_window.update_widget_state)

        self.initialize()

    def initialize(self):

        v_spacing = global_gui_parameters["vertical_spacing_in_tabs"]

        self._setup_settings()
        self._setup_start_fitting()
        self._setup_compute_roi_maps()
        self._setup_save_results()
        self._setup_quantitative_analysis()

        vbox = QVBoxLayout()
        vbox.addWidget(self.group_settings)
        vbox.addSpacing(v_spacing)

        vbox.addWidget(self.pb_start_map_fitting)
        vbox.addWidget(self.pb_compute_roi_maps)
        vbox.addSpacing(v_spacing)

        vbox.addWidget(self.group_save_results)

        vbox.addWidget(self.group_quant_analysis)

        self.setLayout(vbox)

        self._set_tooltips()

        # Timer is currently used to simulate processing
        self._timer = None
        self._timer_counter = 0

    def _setup_settings(self):
        self.group_settings = QGroupBox("Options")

        self._dset_n_rows = 1
        self._dset_n_cols = 1
        self._area_row_min = 1
        self._area_row_max = 1
        self._area_col_min = 1
        self._area_col_max = 1
        self._validator_selected_area = IntValidatorStrict()

        self.le_start_row = LineEditExtended()
        self.le_start_row.textChanged.connect(self.le_start_row_text_changed)
        self.le_start_row.editingFinished.connect(self.le_start_row_editing_finished)
        self.le_start_col = LineEditExtended()
        self.le_start_col.textChanged.connect(self.le_start_col_text_changed)
        self.le_start_col.editingFinished.connect(self.le_start_col_editing_finished)
        self.le_end_row = LineEditExtended()
        self.le_end_row.textChanged.connect(self.le_end_row_text_changed)
        self.le_end_row.editingFinished.connect(self.le_end_row_editing_finished)
        self.le_end_col = LineEditExtended()
        self.le_end_col.textChanged.connect(self.le_end_col_text_changed)
        self.le_end_col.editingFinished.connect(self.le_end_col_editing_finished)

        self.group_save_plots = QGroupBox("Save spectra for pixels in the selected region")
        self.group_save_plots.setCheckable(True)
        self.group_save_plots.setChecked(False)
        self.group_save_plots.toggled.connect(self.group_save_plots_toggled)

        vbox = QVBoxLayout()
        grid = QGridLayout()
        grid.addWidget(QLabel("Start row:"), 0, 0)
        grid.addWidget(self.le_start_row, 0, 1)
        grid.addWidget(QLabel("column:"), 0, 2)
        grid.addWidget(self.le_start_col, 0, 3)
        grid.addWidget(QLabel("End row:"), 1, 0)
        grid.addWidget(self.le_end_row, 1, 1)
        grid.addWidget(QLabel("column:"), 1, 2)
        grid.addWidget(self.le_end_col, 1, 3)
        vbox.addLayout(grid)

        self.group_save_plots.setLayout(vbox)

        vbox = QVBoxLayout()
        vbox.addWidget(self.group_save_plots)

        self.group_settings.setLayout(vbox)

    def _setup_start_fitting(self):
        self.pb_start_map_fitting = QPushButton("Start XRF Map Fitting")
        self.pb_start_map_fitting.clicked.connect(self.pb_start_map_fitting_clicked)

    def _setup_compute_roi_maps(self):
        self.pb_compute_roi_maps = QPushButton("Compute XRF Maps Based on ROI ...")
        self.pb_compute_roi_maps.clicked.connect(self.pb_compute_roi_maps_clicked)

    def _setup_save_results(self):
        self.group_save_results = QGroupBox("Save Results")

        self.pb_save_to_db = QPushButton("Save to Database (Databroker) ...")
        self.pb_save_to_db.setEnabled(False)

        self.pb_save_q_calibration = QPushButton("Save Quantitative Calibration ...")
        self.pb_save_q_calibration.clicked.connect(self.pb_save_q_calibration_clicked)

        self.pb_export_to_tiff_and_txt = QPushButton("Export to TIFF and TXT ...")
        self.pb_export_to_tiff_and_txt.clicked.connect(self.pb_export_to_tiff_and_txt_clicked)

        grid = QGridLayout()
        grid.addWidget(self.pb_save_to_db, 0, 0, 1, 2)
        grid.addWidget(self.pb_save_q_calibration, 1, 0, 1, 2)
        grid.addWidget(self.pb_export_to_tiff_and_txt, 2, 0, 1, 2)

        self.group_save_results.setLayout(grid)

    def _setup_quantitative_analysis(self):
        self.group_quant_analysis = QGroupBox("Quantitative Analysis")

        self.pb_load_quant_calib = QPushButton("Load Quantitative Calibration ...")
        self.pb_load_quant_calib.clicked.connect(self.pb_load_quant_calib_clicked)

        vbox = QVBoxLayout()
        vbox.addWidget(self.pb_load_quant_calib)
        self.group_quant_analysis.setLayout(vbox)

    def _set_tooltips(self):
        set_tooltip(
            self.group_settings,
            "Raw spectra of individual pixels are saved as <b>.png</b> files for "
            "the selected region of the map."
            "The region is selected by specifying the <b>Start</b> and <b>End</b> coordinates "
            "(ranges of rows and columns) in pixels. The first and last rows and columns are "
            "included in the selection.")
        set_tooltip(
            self.le_start_row,
            "Number of the <b>first row</b> of the map to be included in the selection. "
            "The number must be less than the number entered into 'End row' box.")
        set_tooltip(
            self.le_start_col,
            "Number of the <b>first column</b> of the map to be included in the selection. "
            "The number must be less than the number entered into 'End column' box.")
        set_tooltip(
            self.le_end_row,
            "Number of the <b>last row</b> included in the selection. "
            "The number must be greater than the number entered into 'Start row' box.")
        set_tooltip(
            self.le_end_col,
            "Number of the <b>last column</b> included in the selection. "
            "The number must be greater than the number entered into 'Start column' box.")

        set_tooltip(
            self.pb_start_map_fitting,
            "Click to start <b>fitting of the XRF Maps</b>. The generated XRF Maps can be viewed "
            "in <b>'XRF Maps' tab</b>")

        set_tooltip(
            self.pb_compute_roi_maps,
            "Opens the window for setting up <b>spectral ROIs</b> and computating XRF Maps "
            "based on the ROIs")

        set_tooltip(self.pb_save_to_db,
                    "Save generated XRF Maps to a <b>database</b> via Databroker")

        set_tooltip(
            self.pb_save_q_calibration,
            "Opens a Dialog Box which allows to preview and save <b>Quantitative Calibration data</b>")
        set_tooltip(
            self.pb_export_to_tiff_and_txt,
            "Open a Dialog box which allows to export XRF Maps as <b>TIFF</b> and <b>TXT</b> files")

        set_tooltip(
            self.pb_load_quant_calib,
            "Open a window with GUI tools for loading and managing previously saved "
            "<b>Quantitative Calibration data</b> used for processing (normalization) "
            "of XRF Maps. The loaded calibration data is applied to XRF Maps if 'Quantitative' "
            "box is checked in 'XRF Maps' tab")

    def update_widget_state(self, condition=None):
        if condition == "tooltips":
            self._set_tooltips()

        state_file_loaded = self.gui_vars["gui_state"]["state_file_loaded"]
        state_model_exist = self.gui_vars["gui_state"]["state_model_exists"]
        state_xrf_map_exists = self.gui_vars["gui_state"]["state_xrf_map_exists"]

        self.group_settings.setEnabled(state_file_loaded & state_model_exist)
        self.pb_start_map_fitting.setEnabled(state_file_loaded & state_model_exist)
        self.pb_compute_roi_maps.setEnabled(state_file_loaded & state_model_exist)
        self.group_save_results.setEnabled(state_xrf_map_exists)
        self.group_quant_analysis.setEnabled(state_xrf_map_exists)

    def slot_update_for_new_loaded_run(self):
        self.gpc.set_enable_save_spectra(False)
        selected_area = {"row_min": 1, "row_max": 1,
                         "col_min": 1, "col_max": 1}
        self.gpc.set_selection_area_save_spectra(selected_area)

        self._update_area_selection_controls()

    def pb_compute_roi_maps_clicked(self):
        # Position the window in relation ot the main window (only when called once)
        pos = self.ref_main_window.pos()
        self.ref_main_window.wnd_compute_roi_maps.position_once(pos.x(), pos.y())

        if not self.ref_main_window.wnd_compute_roi_maps.isVisible():
            self.ref_main_window.wnd_compute_roi_maps.show()
        self.ref_main_window.wnd_compute_roi_maps.activateWindow()

    def pb_save_q_calibration_clicked(self):
        msg = ""
        if not self.gpc.is_quant_standard_selected():
            # This is a safeguard. The button should be disabled if no standard is selected.
            msg += "No quantitative standard is selected. " \
                "Use 'Load Quantitative Standard ...' button in 'Model' tab"
        if not self.gpc.is_quant_standard_fitting_available():
            msg = msg + "\n" if msg else msg
            msg += "Select a dataset containing fitted XRF maps for the quantitative standard " \
                "in XRF Maps tab (dataset name must end with 'fit')."
        if msg:
            msgbox = QMessageBox(QMessageBox.Information, "Additional Steps Needed",
                                 msg, QMessageBox.Ok, parent=self)
            msgbox.exec()
            return

        try:
            params = self.gpc.get_displayed_quant_calib_parameters()
            distance_to_sample = params["distance_to_sample"]
            file_name = params["suggested_file_name"]
            preview = params["preview"]

            file_dir = self.gpc.get_current_working_directory()
            file_dir = os.path.expanduser(file_dir)
            file_path = os.path.join(file_dir, file_name)

            dlg = DialogSaveCalibration(file_path=file_path)
            dlg.distance_to_sample = distance_to_sample
            dlg.preview = preview
            res = dlg.exec()
            if res:
                self.gpc.save_quantitative_standard(dlg.file_path, dlg.overwrite_existing)
                logger.info(f"Quantitative calibration was saved to the file '{dlg.file_path}'")
            else:
                logger.info("Saving quantitative calibration was cancelled.")

            # We want to save distance to sample even if saving was cancelled
            self.gpc.save_distance_to_sample(dlg.distance_to_sample)
        except Exception as ex:
            msg = str(ex)
            msgbox = QMessageBox(QMessageBox.Critical, "Error",
                                 msg, QMessageBox.Ok, parent=self)
            msgbox.exec()

    def pb_export_to_tiff_and_txt_clicked(self):
        # TODO: Propagate full path to the saved file here
        dir_path = self.gpc.get_current_working_directory()
        dir_path = os.path.expanduser(dir_path)

        params = self.gpc.get_parameters_for_exporting_maps()

        dlg = DialogExportToTiffAndTxt(dir_path=dir_path)
        dlg.dset_list = params["dset_list"]
        dlg.dset_sel = params["dset_sel"]
        dlg.scaler_list = params["scaler_list"]
        dlg.scaler_sel = params["scaler_sel"]
        dlg.interpolate_on = params["interpolate_on"]
        dlg.quant_norm_on = params["quant_norm_on"]

        res = dlg.exec()
        if res:
            try:
                result_path = dlg.dir_path
                dataset_name = dlg.get_selected_dset_name()
                scaler_name = dlg.get_selected_scaler_name()
                interpolate_on = dlg.interpolate_on
                quant_norm_on = dlg.quant_norm_on
                file_formats = []
                if dlg.save_tiff:
                    file_formats.append("tiff")
                if dlg.save_txt:
                    file_formats.append("txt")
                self.gpc.export_xrf_maps(results_path=result_path,
                                         dataset_name=dataset_name,
                                         scaler_name=scaler_name,
                                         interpolate_on=interpolate_on,
                                         quant_norm_on=quant_norm_on,
                                         file_formats=file_formats)
                if file_formats:
                    formats_text = " and ".join([_.upper() for _ in file_formats])
                else:
                    formats_text = "No"
                msg = f"{formats_text} files were saved to the directory '{result_path}'"
                logger.info(msg)

                msgbox = QMessageBox(QMessageBox.Information, "Files Saved",
                                     msg, QMessageBox.Ok, parent=self)
                msgbox.exec()

            except Exception as ex:
                msg = str(ex)
                msgbox = QMessageBox(QMessageBox.Critical, "Error",
                                     msg, QMessageBox.Ok, parent=self)
                msgbox.exec()

    def pb_load_quant_calib_clicked(self):
        # Position the window in relation ot the main window (only when called once)
        pos = self.ref_main_window.pos()
        self.ref_main_window.wnd_load_quantitative_calibration.position_once(pos.x(), pos.y())

        if not self.ref_main_window.wnd_load_quantitative_calibration.isVisible():
            self.ref_main_window.wnd_load_quantitative_calibration.show()
        self.ref_main_window.wnd_load_quantitative_calibration.activateWindow()

    def pb_start_map_fitting_clicked(self):

        def cb():
            try:
                self.gpc.fit_individual_pixels()
                success, msg = True, ""
            except Exception as ex:
                success, msg = False, str(ex)

            return {"success": success, "msg": msg}

        self._compute_in_background(cb, self.slot_start_map_fitting_clicked)

    @Slot(object)
    def slot_start_map_fitting_clicked(self, result):
        self._recover_after_compute(self.slot_start_map_fitting_clicked)

        success = result["success"]
        if success:
            self.gui_vars["gui_state"]["state_xrf_map_exists"] = True
        else:
            msg = result["msg"]
            msgbox = QMessageBox(QMessageBox.Critical, "Failed to Fit Individual Pixel Spectra",
                                 msg, QMessageBox.Ok, parent=self)
            msgbox.exec()

        self.signal_map_fitting_complete.emit()
        self.update_global_state.emit()
        if success:
            self.signal_activate_tab_xrf_maps.emit()

    '''
    @Slot()
    def timerExpired(self):
        self._timer_counter += 1
        progress_bar = self.ref_main_window.statusProgressBar
        progress_bar.setValue(self._timer_counter)
        if self._timer_counter >= 100:
            self._timer.stop()
            self._timer.timeout.disconnect(self.timerExpired)
            self._timer = None
            progress_bar.setValue(0)
            status_bar = self.ref_main_window.statusBar()
            status_bar.showMessage("XRF Maps are generated. "
                                   "Results are presented in 'XRF Maps' tab.", 5000)
            self.gui_vars["gui_state"]["running_computations"] = False
            self.update_global_state.emit()
    '''

    def group_save_plots_toggled(self, state):
        self.gpc.set_enable_save_spectra(state)

    def le_start_row_text_changed(self, text):
        valid = self._validate_row_number(text) and int(text) <= self._area_row_max
        self.le_start_row.setValid(valid)

    def le_start_row_editing_finished(self):
        text = self.le_start_row.text()
        valid = self._validate_row_number(text) and int(text) <= self._area_row_max
        if valid:
            self._save_selected_area(row_min=int(text))
        else:
            self._show_selected_area()

    def le_end_row_text_changed(self, text):
        valid = self._validate_row_number(text) and int(text) >= self._area_row_min
        self.le_end_row.setValid(valid)

    def le_end_row_editing_finished(self):
        text = self.le_end_row.text()
        valid = self._validate_row_number(text) and int(text) >= self._area_row_min
        if valid:
            self._save_selected_area(row_max=int(text))
        else:
            self._show_selected_area()

    def le_start_col_text_changed(self, text):
        valid = self._validate_col_number(text) and int(text) <= self._area_col_max
        self.le_start_col.setValid(valid)

    def le_start_col_editing_finished(self):
        text = self.le_start_col.text()
        valid = self._validate_col_number(text) and int(text) <= self._area_col_max
        if valid:
            self._save_selected_area(col_min=int(text))
        else:
            self._show_selected_area()

    def le_end_col_text_changed(self, text):
        valid = self._validate_col_number(text) and int(text) >= self._area_col_min
        self.le_end_col.setValid(valid)

    def le_end_col_editing_finished(self):
        text = self.le_end_col.text()
        valid = self._validate_col_number(text) and int(text) >= self._area_col_min
        if valid:
            self._save_selected_area(col_max=int(text))
        else:
            self._show_selected_area()

    def _validate_row_number(self, value_str):
        if self._validator_selected_area.validate(value_str, 0)[0] != IntValidatorStrict.Acceptable:
            return False
        value = int(value_str)
        if 1 <= value <= self._dset_n_rows:
            return True
        else:
            return False

    def _validate_col_number(self, value_str):
        if self._validator_selected_area.validate(value_str, 0)[0] != IntValidatorStrict.Acceptable:
            return False
        value = int(value_str)
        if 1 <= value <= self._dset_n_cols:
            return True
        else:
            return False

    def _update_area_selection_controls(self):
        map_size = self.gpc.get_dataset_map_size()
        map_size = (1, 1) if map_size is None else map_size
        self._dset_n_rows, self._dset_n_cols = map_size

        self.group_save_plots.setChecked(self.gpc.get_enable_save_spectra())
        area = self.gpc.get_selection_area_save_spectra()
        self._area_row_min = area['row_min']
        self._area_row_max = area['row_max']
        self._area_col_min = area['col_min']
        self._area_col_max = area['col_max']
        self._show_selected_area()

    def _show_selected_area(self):
        self.le_start_row.setText(f"{self._area_row_min}")
        self.le_end_row.setText(f"{self._area_row_max}")
        self.le_start_col.setText(f"{self._area_col_min}")
        self.le_end_col.setText(f"{self._area_col_max}")

    def _save_selected_area(self, row_min=None, row_max=None, col_min=None, col_max=None):
        if row_min is not None:
            self._area_row_min = row_min
        if row_max is not None:
            self._area_row_max = row_max
        if col_min is not None:
            self._area_col_min = col_min
        if col_max is not None:
            self._area_col_max = col_max
        area = {"row_min": self._area_row_min,
                "row_max": self._area_row_max,
                "col_min": self._area_col_min,
                "col_max": self._area_col_max}
        self.gpc.set_selection_area_save_spectra(area)
        self._update_area_selection_controls()

    def _compute_in_background(self, func, slot, *args, **kwargs):
        """
        Run function `func` in a background thread. Send the signal
        `self.computations_complete` once computation is finished.

        Parameters
        ----------
        func: function
            Reference to a function that is supposed to be executed at the background.
            The function return value is passed as a signal parameter once computation is
            complete.
        slot: qtpy.QtCore.Slot or None
            Reference to a slot. If not None, then the signal `self.computation_complete`
            is connected to this slot.
        args, kwargs
            arguments of the function `func`.
        """
        signal_complete = self.computations_complete

        def func_to_run(func, *args, **kwargs):
            class LoadFile(QRunnable):
                def run(self):
                    result_dict = func(*args, **kwargs)
                    signal_complete.emit(result_dict)
            return LoadFile()

        if slot is not None:
            self.computations_complete.connect(slot)
        self.gui_vars["gui_state"]["running_computations"] = True
        self.update_global_state.emit()
        QThreadPool.globalInstance().start(func_to_run(func, *args, **kwargs))

    def _recover_after_compute(self, slot):
        """
        The function should be called after the signal `self.computations_complete` is
        received. The slot should be the same as the one used when calling
        `self.compute_in_background`.
        """
        if slot is not None:
            self.computations_complete.disconnect(slot)
        self.gui_vars["gui_state"]["running_computations"] = False
        self.update_global_state.emit()


class WndComputeRoiMaps(SecondaryWindow):

    # Signal that is sent (to main window) to update global state of the program
    update_global_state = Signal()
    computations_complete = Signal(object)

    signal_roi_computation_complete = Signal()
    signal_activate_tab_xrf_maps = Signal()

    def __init__(self, *, gpc, gui_vars):
        super().__init__()

        # Global processing classes
        self.gpc = gpc
        # Global GUI variables (used for control of GUI state)
        self.gui_vars = gui_vars

        # Reference to the main window. The main window will hold
        #   references to all non-modal windows that could be opened
        #   from multiple places in the program.
        self.ref_main_window = self.gui_vars["ref_main_window"]

        self.update_global_state.connect(self.ref_main_window.update_widget_state)

        self.initialize()

    def initialize(self):
        self.setWindowTitle("PyXRF: Compute XRF Maps Based on ROIs")

        self.setMinimumWidth(600)
        self.setMinimumHeight(300)
        self.resize(600, 600)

        header_vbox = self._setup_header()
        self._setup_table()
        footer_hbox = self._setup_footer()

        vbox = QVBoxLayout()
        vbox.addLayout(header_vbox)
        vbox.addWidget(self.table)
        vbox.addLayout(footer_hbox)

        self.setLayout(vbox)

        self._set_tooltips()

    def _setup_header(self):
        self.pb_clear = QPushButton("Clear")
        self.pb_clear.clicked.connect(self.pb_clear_clicked)
        self.pb_use_lines_for_fitting = QPushButton("Use Lines Selected For Fitting")
        self.pb_use_lines_for_fitting.clicked.connect(self.pb_use_lines_for_fitting_clicked)

        self.le_sel_emission_lines = LineEditExtended()
        self.le_sel_emission_lines.textChanged.connect(self.le_sel_emission_lines_text_changed)
        self.le_sel_emission_lines.editingFinished.connect(self.le_sel_emission_lines_editing_finished)

        sample_elements = ""
        self.le_sel_emission_lines.setText(sample_elements)

        vbox = QVBoxLayout()
        hbox = QHBoxLayout()
        hbox.addWidget(QLabel("Enter emission lines, e.g. Fe_K, Gd_L  "))
        hbox.addStretch(1)
        hbox.addWidget(self.pb_clear)
        hbox.addWidget(self.pb_use_lines_for_fitting)
        vbox.addLayout(hbox)
        vbox.addWidget(self.le_sel_emission_lines)

        return vbox

    def _setup_table(self):

        # Labels for horizontal header
        self.tbl_labels = ["Line", "E, keV", "ROI, keV", "Show", "Reset"]

        # The list of columns that stretch with the table
        self.tbl_cols_stretch = ("E, keV", "ROI, keV")

        # Table item representation if different from default
        self.tbl_format = {"E, keV": ".3f"}

        # Editable items (highlighted with lighter background)
        self.tbl_cols_editable = {"ROI, keV"}

        # Columns that contain Range Manager
        self.tbl_cols_range_manager = ("ROI, keV",)

        self.table = QTableWidget()
        self.table.setColumnCount(len(self.tbl_labels))
        self.table.setHorizontalHeaderLabels(self.tbl_labels)
        self.table.verticalHeader().hide()
        self.table.setSelectionMode(QTableWidget.NoSelection)

        header = self.table.horizontalHeader()
        for n, lbl in enumerate(self.tbl_labels):
            # Set stretching for the columns
            if lbl in self.tbl_cols_stretch:
                header.setSectionResizeMode(n, QHeaderView.Stretch)
            else:
                header.setSectionResizeMode(n, QHeaderView.ResizeToContents)

        self._table_contents = []
        self.cb_list = []
        self.range_manager_list = []
        self.pb_default_list = []
        self.fill_table(self._table_contents)

    def fill_table(self, table_contents):

        self.table.clearContents()
        self._table_contents = table_contents  # Save new table contents

        for item in self.range_manager_list:
            item.selection_changed.disconnect(self.range_manager_selection_changed)
        self.range_manager_list = []

        for cb in self.cb_list:
            cb.stateChanged.disconnect(self.cb_state_changed)
        self.cb_list = []

        for pb in self.pb_default_list:
            pb.clicked.connect(self.pb_default_clicked)
        self.pb_default_list = []

        self.table.setRowCount(len(table_contents))
        for nr, row in enumerate(table_contents):
            eline_name = row["eline"] + "a1"
            energy = row["energy_center"]
            energy_left = row["energy_left"]
            energy_right = row["energy_right"]
            range_displayed = row["range_displayed"]
            table_row = [eline_name, energy, (energy_left, energy_right)]
            for nc, entry in enumerate(table_row):

                label = self.tbl_labels[nc]

                # Set alternating background colors for the table rows
                #   Make background for editable items a little brighter
                brightness = 240 if label in self.tbl_cols_editable else 220
                if nr % 2:
                    rgb_bckg = (255, brightness, brightness)
                else:
                    rgb_bckg = (brightness, 255, brightness)

                if self.tbl_labels[nc] not in self.tbl_cols_range_manager:
                    if self.tbl_labels[nc] in self.tbl_format:
                        fmt = self.tbl_format[self.tbl_labels[nc]]
                        s = ("{:" + fmt + "}").format(entry)
                    else:
                        s = f"{entry}"

                    item = QTableWidgetItem(s)
                    if nc > 0:
                        item.setTextAlignment(Qt.AlignCenter)
                    else:
                        item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)

                    # Set all columns not editable (unless needed)
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)

                    # Note, that there is no way to set style sheet for QTableWidgetItem
                    item.setBackground(QBrush(QColor(*rgb_bckg)))

                    self.table.setItem(nr, nc, item)
                else:
                    spin_name = f"{nr}"
                    item = RangeManager(name=spin_name, add_sliders=False, selection_to_range_min=0.0001)
                    item.set_range(0.0, 100.0)  # The range is greater than needed (in keV)
                    item.set_selection(value_low=entry[0], value_high=entry[1])
                    item.setTextColor((0, 0, 0))  # In case of dark theme
                    item.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)

                    self.range_manager_list.append(item)

                    item.selection_changed.connect(self.range_manager_selection_changed)

                    color = (rgb_bckg[0], rgb_bckg[1], rgb_bckg[2])
                    item.setBackground(color)

                    self.table.setCellWidget(nr, nc, item)

            brightness = 220
            if nr % 2:
                rgb_bckg = (255, brightness, brightness)
            else:
                rgb_bckg = (brightness, 255, brightness)

            item = QWidget()
            cb = CheckBoxNamed(name=f"{nr}")
            cb.setChecked(Qt.Checked if range_displayed else Qt.Unchecked)
            self.cb_list.append(cb)
            cb.stateChanged.connect(self.cb_state_changed)

            item_hbox = QHBoxLayout(item)
            item_hbox.addWidget(cb)
            item_hbox.setAlignment(Qt.AlignCenter)
            item_hbox.setContentsMargins(0, 0, 0, 0)
            color_css = f"rgb({rgb_bckg[0]}, {rgb_bckg[1]}, {rgb_bckg[2]})"
            item.setStyleSheet(f"QWidget {{ background-color: {color_css}; }};; "
                               f"QCheckBox {{ background-color: white }}")
            self.table.setCellWidget(nr, nc + 1, item)

            item = PushButtonNamed("Reset", name=f"{nr}")
            item.clicked.connect(self.pb_default_clicked)
            self.pb_default_list.append(item)
            rgb_bckg = [_ - 35 if (_ < 255) else _ for _ in rgb_bckg]
            color_css = f"rgb({rgb_bckg[0]}, {rgb_bckg[1]}, {rgb_bckg[2]})"
            item.setStyleSheet(f"QPushButton {{ background-color: {color_css}; }}")
            self.table.setCellWidget(nr, nc + 2, item)

    def _setup_footer(self):

        self.cb_subtract_baseline = QCheckBox("Subtract baseline")
        self.cb_subtract_baseline.setChecked(
            Qt.Checked if self.gpc.get_roi_subtract_background() else Qt.Unchecked)
        self.cb_subtract_baseline.toggled.connect(self.cb_subtract_baseline_toggled)

        self.pb_compute_roi = QPushButton("Compute ROIs")
        self.pb_compute_roi.clicked.connect(self.pb_compute_roi_clicked)

        hbox = QHBoxLayout()
        hbox.addWidget(self.cb_subtract_baseline)
        hbox.addStretch(1)
        hbox.addWidget(self.pb_compute_roi)

        return hbox

    def _set_tooltips(self):
        set_tooltip(self.pb_clear,
                    "<b>Clear</b> the list")
        set_tooltip(self.pb_use_lines_for_fitting,
                    "Copy the contents of <b>the list of emission lines selected for "
                    "fitting</b> to the list of ROIs")
        set_tooltip(self.le_sel_emission_lines,
                    "The list of <b>emission lines</b> selected for ROI computation.")

        set_tooltip(self.table, "The list of ROIs")

        set_tooltip(self.cb_subtract_baseline,
                    "<b>Subtract baseline</b> from the pixel spectra before computing ROIs. "
                    "Subtracting baseline slows down computations and usually have no benefit. "
                    "In most cases it should remain <b>unchecked</b>.")
        set_tooltip(self.pb_compute_roi,
                    "<b>Run</b> computations of the ROIs. The resulting <b>ROI</b> dataset "
                    "may be viewed in <b>XRF Maps</b> tab.")

    def update_widget_state(self, condition=None):
        # Update the state of the menu bar
        state = not self.gui_vars["gui_state"]["running_computations"]
        self.setEnabled(state)

        # Hide the window if required by the program state
        state_file_loaded = self.gui_vars["gui_state"]["state_file_loaded"]
        state_model_exist = self.gui_vars["gui_state"]["state_model_exists"]
        if not state_file_loaded or not state_model_exist:
            self.hide()

        if condition == "tooltips":
            self._set_tooltips()

    def pb_clear_clicked(self):
        self.gpc.clear_roi_element_list()
        self._update_displayed_element_list()
        self._validate_element_list()

    def pb_use_lines_for_fitting_clicked(self):
        self.gpc.load_roi_element_list_from_selected()
        self._update_displayed_element_list()
        self._validate_element_list()

    def le_sel_emission_lines_text_changed(self, text):
        self._validate_element_list(text)

    def le_sel_emission_lines_editing_finished(self):
        text = self.le_sel_emission_lines.text()
        if self._validate_element_list(text):
            self.gpc.set_roi_selected_element_list(text)
            self._update_table()
        else:
            element_list = self.gpc.get_roi_selected_element_list()
            self.le_sel_emission_lines.setText(element_list)

    def cb_subtract_baseline_toggled(self, state):
        self.gpc.set_roi_subtract_background(bool(state))

    def cb_state_changed(self, name, state):
        try:
            nr = int(name)  # Row number
            checked = state == Qt.Checked

            eline = self._table_contents[nr]["eline"]
            self._table_contents[nr]["range_displayed"] = checked
            self.gpc.show_roi(eline, checked)
        except Exception as ex:
            logger.error(f"Failed to process selection change. Exception occurred: {ex}.")

    def _find_spin_box(self, name):
        for item in self.spin_list:
            if item.getName() == name:
                return item
        return None

    def spin_value_changed(self, name, value):
        try:
            nr, side = name.split(",")
            nr = int(nr)
            keys = {"left": "energy_left", "right": "energy_right"}
            side = keys[side]
            eline = self._table_contents[nr]["eline"]
            if self._table_contents[nr][side] == value:
                return
            if side == "energy_left":  # Left boundary
                if value < self._table_contents[nr]["energy_right"]:
                    self._table_contents[nr][side] = value
            else:  # Right boundary
                if value > self._table_contents[nr]["energy_left"]:
                    self._table_contents[nr][side] = value

            # Update plot
            left, right = self._table_contents[nr]["energy_left"], self._table_contents[nr]["energy_right"]
            self.gpc.change_roi(eline, left, right)
        except Exception as ex:
            logger.error(f"Failed to change the ROI. Exception occurred: {ex}.")

    def range_manager_selection_changed(self, left, right, name):
        try:
            nr = int(name)
            eline = self._table_contents[nr]["eline"]
            self.gpc.change_roi(eline, left, right)
        except Exception as ex:
            logger.error(f"Failed to change the ROI. Exception occurred: {ex}.")

    def pb_default_clicked(self, name):
        try:
            nr = int(name)
            eline = self._table_contents[nr]["eline"]
            left = self._table_contents[nr]["energy_left_default"]
            right = self._table_contents[nr]["energy_right_default"]
            self.range_manager_list[nr].set_selection(value_low=left, value_high=right)
            self.gpc.change_roi(eline, left, right)
        except Exception as ex:
            logger.error(f"Failed to change the ROI. Exception occurred: {ex}.")

    def pb_compute_roi_clicked(self):

        def cb():
            try:
                self.gpc.compute_rois()
                success, msg = True, ""
            except Exception as ex:
                success, msg = False, str(ex)

            return {"success": success, "msg": msg}

        self._compute_in_background(cb, self.slot_compute_roi_clicked)

    @Slot(object)
    def slot_compute_roi_clicked(self, result):
        self._recover_after_compute(self.slot_compute_roi_clicked)

        success = result["success"]
        if success:
            self.gui_vars["gui_state"]["state_xrf_map_exists"] = True
        else:
            msg = result["msg"]
            msgbox = QMessageBox(QMessageBox.Critical, "Failed to Compute ROIs",
                                 msg, QMessageBox.Ok, parent=self)
            msgbox.exec()

        self.signal_roi_computation_complete.emit()
        self.update_global_state.emit()
        if success:
            self.signal_activate_tab_xrf_maps.emit()

    def _update_displayed_element_list(self):
        element_list = self.gpc.get_roi_selected_element_list()
        self.le_sel_emission_lines.setText(element_list)
        self._validate_element_list()
        self._update_table()

    def _update_table(self):
        table_contents = self.gpc.get_roi_settings()
        self.fill_table(table_contents)

    def _validate_element_list(self, text=None):
        if text is None:
            text = self.le_sel_emission_lines.text()
        el_list = text.split(",")
        el_list = [_.strip() for _ in el_list]
        if el_list == [""]:
            el_list = []
        valid = bool(len(el_list))
        for eline in el_list:
            if self.gpc.get_eline_name_category(eline) != "eline":
                valid = False

        self.le_sel_emission_lines.setValid(valid)
        self.pb_compute_roi.setEnabled(valid)

        return valid

    def _compute_in_background(self, func, slot, *args, **kwargs):
        """
        Run function `func` in a background thread. Send the signal
        `self.computations_complete` once computation is finished.

        Parameters
        ----------
        func: function
            Reference to a function that is supposed to be executed at the background.
            The function return value is passed as a signal parameter once computation is
            complete.
        slot: qtpy.QtCore.Slot or None
            Reference to a slot. If not None, then the signal `self.computation_complete`
            is connected to this slot.
        args, kwargs
            arguments of the function `func`.
        """
        signal_complete = self.computations_complete

        def func_to_run(func, *args, **kwargs):
            class LoadFile(QRunnable):
                def run(self):
                    result_dict = func(*args, **kwargs)
                    signal_complete.emit(result_dict)
            return LoadFile()

        if slot is not None:
            self.computations_complete.connect(slot)
        self.gui_vars["gui_state"]["running_computations"] = True
        self.update_global_state.emit()
        QThreadPool.globalInstance().start(func_to_run(func, *args, **kwargs))

    def _recover_after_compute(self, slot):
        """
        The function should be called after the signal `self.computations_complete` is
        received. The slot should be the same as the one used when calling
        `self.compute_in_background`.
        """
        if slot is not None:
            self.computations_complete.disconnect(slot)
        self.gui_vars["gui_state"]["running_computations"] = False
        self.update_global_state.emit()


class WndLoadQuantitativeCalibration(SecondaryWindow):

    signal_quantitative_calibration_changed = Signal()

    def __init__(self, *, gpc, gui_vars):
        super().__init__()

        # Global processing classes
        self.gpc = gpc
        # Global GUI variables (used for control of GUI state)
        self.gui_vars = gui_vars

        self.initialize()

    def initialize(self):
        self.table_header_display_names = False

        self.setWindowTitle("PyXRF: Load Quantitative Calibration")
        self.setMinimumWidth(750)
        self.setMinimumHeight(400)
        self.resize(750, 600)

        self.pb_load_calib = QPushButton("Load Calibration ...")
        self.pb_load_calib.clicked.connect(self.pb_load_calib_clicked)

        self._changes_exist = False
        self._auto_update = True
        self.cb_auto_update = QCheckBox("Auto")
        self.cb_auto_update.setCheckState(Qt.Checked if self._auto_update else Qt.Unchecked)
        self.cb_auto_update.stateChanged.connect(self.cb_auto_update_state_changed)

        self.pb_update_plots = QPushButton("Update Plots")
        self.pb_update_plots.clicked.connect(self.pb_update_plots_clicked)

        self.grp_current_scan = QGroupBox("Parameters of Currently Processed Scan")

        self._distance_to_sample = 0.0
        self.le_distance_to_sample = LineEditExtended()
        le_dist_validator = QDoubleValidator()
        le_dist_validator.setBottom(0)
        self.le_distance_to_sample.setValidator(le_dist_validator)
        self._set_distance_to_sample()
        self.le_distance_to_sample.editingFinished.connect(self.le_distance_to_sample_editing_finished)
        self.le_distance_to_sample.focusOut.connect(self.le_distance_to_sample_focus_out)

        hbox = QHBoxLayout()
        hbox.addWidget(QLabel("Distance-to-sample:"))
        hbox.addWidget(self.le_distance_to_sample)
        hbox.addStretch(1)
        self.grp_current_scan.setLayout(hbox)

        self.eline_rb_exclusive = []  # Holds the list of groups of exclusive radio buttons
        self._setup_tab_widget()

        vbox = QVBoxLayout()

        hbox = QHBoxLayout()
        hbox.addWidget(self.pb_load_calib)
        hbox.addStretch(1)
        hbox.addWidget(self.cb_auto_update)
        hbox.addWidget(self.pb_update_plots)
        vbox.addLayout(hbox)

        vbox.addWidget(self.tab_widget)

        vbox.addWidget(self.grp_current_scan)

        self.setLayout(vbox)

        # Display data
        self.update_all_data()

        self._set_tooltips()

    def _setup_tab_widget(self):

        self.tab_widget = QTabWidget()
        self.loaded_standards = QWidget()
        # self.display_loaded_standards()
        self.scroll = QScrollArea()
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setWidget(self.loaded_standards)
        self.tab_widget.addTab(self.scroll, "Loaded Standards")

        self.combo_set_table_header = QComboBox()
        self.combo_set_table_header.addItems(["Standard Serial #", "Standard Name"])
        self.combo_set_table_header.currentIndexChanged.connect(
            self.combo_set_table_header_index_changed)

        vbox = QVBoxLayout()
        vbox.addSpacing(5)
        hbox = QHBoxLayout()
        hbox.addWidget(QLabel("Display in table header:"))
        hbox.addWidget(self.combo_set_table_header)
        hbox.addStretch(1)
        vbox.addLayout(hbox)
        self.table = QTableWidget()
        self.table.verticalHeader().hide()
        self.table.setSelectionMode(QTableWidget.NoSelection)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setMinimumSectionSize(150)
        vbox.addWidget(self.table)

        frame = QFrame()
        vbox.setContentsMargins(0, 0, 0, 0)
        frame.setLayout(vbox)

        self.tab_widget.addTab(frame, "Selected Emission Lines")

    def display_loaded_standards(self):
        calib_data = self.gpc.get_quant_calibration_data()
        calib_settings = self.gpc.get_quant_calibration_settings()

        # Create the new widget (this deletes the old widget)
        self.loaded_standards = QWidget()
        self.loaded_standards.setMinimumWidth(700)

        # Also delete references to all components
        self.frames_calib_data = []
        self.pbs_view = []
        self.pbs_remove = []

        # All 'View' buttons are added to the group in order to be connected to the same slot
        self.group_view = QButtonGroup()
        self.group_view.setExclusive(False)
        self.group_view.buttonClicked.connect(self.pb_view_clicked)
        # The same for the 'Remove' buttons
        self.group_remove = QButtonGroup()
        self.group_remove.setExclusive(False)
        self.group_remove.buttonClicked.connect(self.pb_remove_clicked)

        vbox = QVBoxLayout()

        for cdata, csettings in zip(calib_data, calib_settings):
            frame = QFrame()
            frame.setFrameStyle(QFrame.StyledPanel)
            frame.setStyleSheet(get_background_css((200, 255, 200), widget="QFrame"))

            _vbox = QVBoxLayout()

            name = cdata['name']  # Standard name (can be arbitrary string
            # If name is long, then print it in a separate line
            _name_is_long = len(name) > 30

            pb_view = QPushButton("View ...")
            self.group_view.addButton(pb_view)
            pb_remove = QPushButton("Remove")
            self.group_remove.addButton(pb_remove)

            # Row 1: serial, name
            serial = cdata['serial']
            _hbox = QHBoxLayout()
            _hbox.addWidget(QLabel(f"<b>Standard</b> #{serial}"))
            if not _name_is_long:
                _hbox.addWidget(QLabel(f"'{name}'"))
            _hbox.addStretch(1)
            _hbox.addWidget(pb_view)
            _hbox.addWidget(pb_remove)
            _vbox.addLayout(_hbox)

            # Optional row
            if _name_is_long:
                # Wrap name if it is extemely long
                name = textwrap.fill(name, width=80)
                _hbox = QHBoxLayout()
                _hbox.addWidget(QLabel("<b>Name:</b> "), 0, Qt.AlignTop)
                _hbox.addWidget(QLabel(name), 0, Qt.AlignTop)
                _hbox.addStretch(1)
                _vbox.addLayout(_hbox)

            # Row 2: description
            description = textwrap.fill(cdata['description'], width=80)
            _hbox = QHBoxLayout()
            _hbox.addWidget(QLabel("<b>Description:</b>"), 0, Qt.AlignTop)
            _hbox.addWidget(QLabel(f"{description}"), 0, Qt.AlignTop)
            _hbox.addStretch(1)
            _vbox.addLayout(_hbox)

            # Row 3:
            incident_energy = cdata['incident_energy']
            scaler = cdata['scaler_name']
            detector_channel = cdata['detector_channel']
            distance_to_sample = cdata['distance_to_sample']
            _hbox = QHBoxLayout()
            _hbox.addWidget(QLabel(f"<b>Incident energy, keV:</b> {incident_energy}"))
            _hbox.addWidget(QLabel(f"  <b>Scaler:</b> {scaler}"))
            _hbox.addWidget(QLabel(f"  <b>Detector channel:</b> {detector_channel}"))
            _hbox.addWidget(QLabel(f"  <b>Distance-to-sample:</b> {distance_to_sample}"))
            _hbox.addStretch(1)
            _vbox.addLayout(_hbox)

            # Row 4: file name
            fln = textwrap.fill(csettings['file_path'], width=80)
            _hbox = QHBoxLayout()
            _hbox.addWidget(QLabel("<b>Source file:</b>"), 0, Qt.AlignTop)
            _hbox.addWidget(QLabel(fln), 0, Qt.AlignTop)
            _hbox.addStretch(1)
            _vbox.addLayout(_hbox)

            frame.setLayout(_vbox)

            # Now the group box is added to the upper level layout
            vbox.addWidget(frame)
            vbox.addSpacing(5)
            self.frames_calib_data.append(frame)
            self.pbs_view.append(pb_view)
            self.pbs_remove.append(pb_remove)

        # Add the layout to the widget
        self.loaded_standards.setLayout(vbox)
        # ... and put the widget inside the scroll area. This will update the
        # contents of the scroll area.
        self.scroll.setWidget(self.loaded_standards)

    def display_table_header(self):
        calib_data = self.gpc.get_quant_calibration_data()
        header_by_name = self.table_header_display_names

        tbl_labels = ["Lines"]
        for n, cdata in enumerate(calib_data):
            if header_by_name:
                txt = cdata['name']
            else:
                txt = cdata['serial']
            txt = textwrap.fill(txt, width=20)
            tbl_labels.append(txt)

        self.table.setHorizontalHeaderLabels(tbl_labels)

    def display_standard_selection_table(self):
        calib_data = self.gpc.get_quant_calibration_data()
        self._quant_file_paths = self.gpc.get_quant_calibration_file_path_list()

        brightness = 220
        table_colors = [(255, brightness, brightness), (brightness, 255, brightness)]

        # Disconnect all radio button signals before clearing the table
        for bgroup in self.eline_rb_exclusive:
            bgroup.buttonToggled.disconnect(self.rb_selection_toggled)

        # This list will hold radio button groups for horizontal rows
        #   Those are exclusive groups. They are not going to be
        #   used directly, but they must be kept alive in order
        #   for the radiobuttons to work properly. Most of the groups
        #   will contain only 1 radiobutton, which will always remain checked.
        self.eline_rb_exclusive = []
        # The following list will contain the list of radio buttons for each
        #   row. If there is no radiobutton in a position, then the element is
        #   set to None.
        # N rows: the number of emission lines, N cols: the number of standards
        self.eline_rb_lists = []

        self.table.clear()

        if not calib_data:
            self.table.setRowCount(0)
            self.table.setColumnCount(0)
        else:
            # Create the sorted list of available element lines
            line_set = set()
            for cdata in calib_data:
                ks = list(cdata['element_lines'].keys())
                line_set.update(list(ks))
            self.eline_list = list(line_set)
            self.eline_list.sort()

            for n in range(len(self.eline_list)):
                self.eline_rb_exclusive.append(QButtonGroup())
                self.eline_rb_lists.append([None] * len(calib_data))

            self.table.setColumnCount(len(calib_data) + 1)
            self.table.setRowCount(len(self.eline_list))
            self.display_table_header()

            for n, eline in enumerate(self.eline_list):

                rgb = table_colors[n % 2]

                item = QTableWidgetItem(eline)
                item.setTextAlignment(Qt.AlignCenter)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                item.setBackground(QBrush(QColor(*rgb)))
                self.table.setItem(n, 0, item)

                for ns, cdata in enumerate(calib_data):
                    q_file_path = self._quant_file_paths[ns]  # Used to identify standard
                    if eline in cdata['element_lines']:
                        rb = QRadioButton()
                        if self.gpc.get_quant_calibration_is_eline_selected(eline, q_file_path):
                            rb.setChecked(True)

                        self.eline_rb_lists[n][ns] = rb
                        # self.eline_rb_by_standard[ns].addButton(rb)
                        self.eline_rb_exclusive[n].addButton(rb)

                        item = QWidget()
                        item_hbox = QHBoxLayout(item)
                        item_hbox.addWidget(rb)
                        item_hbox.setAlignment(Qt.AlignCenter)
                        item_hbox.setContentsMargins(0, 0, 0, 0)

                        item.setStyleSheet(get_background_css(rgb))

                        # Generate tooltip
                        density = cdata['element_lines'][eline]['density']
                        fluorescence = cdata['element_lines'][eline]['fluorescence']
                        ttip = (f"Fluorescence (F): {fluorescence:12g}\n"
                                f"Density (D): {density:12g}\n")
                        # Avoid very small values of density (probably zero)
                        if abs(density) > 1e-30:
                            ttip += f"F/D: {fluorescence/density:12g}"

                        item.setToolTip(ttip)

                        self.table.setCellWidget(n, ns + 1, item)
                    else:
                        # There is no radio button, but we still need to fill the cell
                        item = QTableWidgetItem("")
                        item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                        item.setBackground(QBrush(QColor(*rgb)))
                        self.table.setItem(n, ns + 1, item)

            # Now the table is set (specifically radio buttons).
            # So we can connect the button groups with the event processing function
            for bgroup in self.eline_rb_exclusive:
                bgroup.buttonToggled.connect(self.rb_selection_toggled)

    @Slot()
    def update_all_data(self):
        self.display_loaded_standards()
        self.display_standard_selection_table()
        self._set_distance_to_sample()

    def _set_distance_to_sample(self):
        """Set 'le_distance_to_sample` without updating maps"""
        distance_to_sample = self.gpc.get_quant_calibration_distance_to_sample()
        if distance_to_sample is None:
            distance_to_sample = 0.0
        self._distance_to_sample = distance_to_sample
        self._set_le_distance_to_sample(distance_to_sample)

    def _set_tooltips(self):
        set_tooltip(self.pb_load_calib, "Load <b>calibration data</b> from JSON file.")
        set_tooltip(self.cb_auto_update, "Automatically <b>update the plots</b> when changes are made. "
                                         "If unchecked, then button <b>Update Plots</b> must be pressed "
                                         "to update the plots. Automatic update is often undesirable "
                                         "when large maps are displayed and multiple changes to parameters "
                                         "are made.")
        set_tooltip(self.pb_update_plots,
                    "<b>Update plots</b> based on currently selected parameters.")
        set_tooltip(self.le_distance_to_sample,
                    "Distance between <b>the sample and the detector</b>. The ratio between of the distances "
                    "during calibration and measurement is used to scale computed concentrations. "
                    "If distance-to-sample is 0 for calibration or measurement, then no scaling is performed.")
        set_tooltip(self.combo_set_table_header,
                    "Use <b>Serial Number</b> or <b>Name</b> of the calibration standard "
                    "in the header of the table")
        set_tooltip(self.table,
                    "Use Radio Buttons to select the <b>source of calibration data</b> for each emission line. "
                    "This feature is needed if multiple loaded calibration files have data on the same "
                    "emission line.")

    def update_widget_state(self, condition=None):
        # Update the state of the menu bar
        state = not self.gui_vars["gui_state"]["running_computations"]
        self.setEnabled(state)

        # Hide the window if required by the program state
        state_xrf_map_exists = self.gui_vars["gui_state"]["state_xrf_map_exists"]
        if not state_xrf_map_exists:
            self.hide()

        if condition == "tooltips":
            self._set_tooltips()

    def cb_auto_update_state_changed(self, state):
        self._auto_update = state
        self.pb_update_plots.setEnabled(not state)
        # If changes were made, apply the changes while switching to 'auto' mode
        if state and self._changes_exist:
            self._update_maps_auto()

    def pb_update_plots_clicked(self):
        self._update_maps()

    def pb_load_calib_clicked(self):
        current_dir = self.gpc.get_current_working_directory()
        file_name = QFileDialog.getOpenFileName(self, "Select File with Quantitative Calibration Data",
                                                current_dir,
                                                "JSON (*.json);; All (*)")
        file_name = file_name[0]
        if file_name:
            logger.debug(f"Loading quantitative calibration from file: '{file_name}'")
            self.gpc.load_quantitative_calibration_data(file_name)
            self.update_all_data()
            self._update_maps_auto()

    def pb_view_clicked(self, button):
        try:
            n_standard = self.pbs_view.index(button)
            calib_settings = self.gpc.get_quant_calibration_settings()
            file_path = calib_settings[n_standard]['file_path']
            calib_preview = self.gpc.get_quant_calibration_text_preview(file_path)
            dlg = DialogViewCalibStandard(None,
                                          file_path=file_path,
                                          calib_preview=calib_preview)
            dlg.exec()
        except ValueError:
            logger.error("'View' button was pressed, but not found in the list of buttons")

    def pb_remove_clicked(self, button):
        try:
            n_standard = self.pbs_remove.index(button)
            calib_settings = self.gpc.get_quant_calibration_settings()
            file_path = calib_settings[n_standard]['file_path']
            self.gpc.quant_calibration_remove_entry(file_path)
            self.update_all_data()
            self._update_maps_auto()
        except ValueError:
            logger.error("'Remove' button was pressed, but not found in the list of buttons")

    def rb_selection_toggled(self, button, checked):
        if checked:
            # Find the button in 2D list 'self.eline_rb_lists'
            button_found = False
            for nr, rb_list in enumerate(self.eline_rb_lists):
                try:
                    nc = rb_list.index(button)
                    button_found = True
                    break
                except ValueError:
                    pass

            if button_found:
                eline = self.eline_list[nr]
                n_standard = nc
                file_path = self._quant_file_paths[n_standard]
                self.gpc.set_quant_calibration_select_eline(eline, file_path)
                self._update_maps_auto()
            else:
                # This should never happen
                logger.error("Selection radio button was pressed, but not found in the list")

    def combo_set_table_header_index_changed(self, index):
        self.table_header_display_names = bool(index)
        self.display_table_header()

    def le_distance_to_sample_editing_finished(self):
        distance_to_sample = float(self.le_distance_to_sample.text())
        if distance_to_sample != self._distance_to_sample:
            self._distance_to_sample = distance_to_sample
            self.gpc.set_quant_calibration_distance_to_sample(distance_to_sample)
            self._update_maps_auto()

    def le_distance_to_sample_focus_out(self):
        try:
            float(self.le_distance_to_sample.text())
        except ValueError:
            # If the text can not be interpreted to float, then replace the text with the old value
            self._set_le_distance_to_sample(self._distance_to_sample)

    def _set_le_distance_to_sample(self, distance_to_sample):
        self.le_distance_to_sample.setText(f"{distance_to_sample:.12g}")

    def _update_maps_auto(self):
        """Update maps only if 'auto' update is ON. Used as a 'filter'
        to prevent extra plot updates."""
        self._changes_exist = True
        if self._auto_update:
            self._update_maps()

    def _update_maps(self):
        """Upload the selections (limit table) and update plot"""
        self._changes_exist = False
        self._redraw_maps()
        # Emit signal only after the maps are redrawn. This should change
        #   ranges in the respective controls for the plots
        self.signal_quantitative_calibration_changed.emit()

    def _redraw_maps(self):
        # We don't emit any signals here, but we don't really need to.
        logger.debug("Redrawing RGB XRF Maps")
        self.gpc.compute_map_ranges()
        self.gpc.redraw_maps()
        self.gpc.compute_rgb_map_ranges()
        self.gpc.redraw_rgb_maps()


class DialogViewCalibStandard(QDialog):

    def __init__(self, parent=None, *, file_path="", calib_preview=""):

        super().__init__(parent)

        self.setWindowTitle("View Calibration Standard")

        self.setMinimumSize(300, 400)
        self.resize(700, 700)

        # Displayed data (must be set before the dialog is shown
        self.file_path = file_path
        self.calib_preview = calib_preview

        vbox = QVBoxLayout()
        hbox = QHBoxLayout()
        hbox.addWidget(QLabel("<b>Source file:</b> "), 0, Qt.AlignTop)
        file_path = textwrap.fill(self.file_path, width=80)
        hbox.addWidget(QLabel(file_path), 0, Qt.AlignTop)
        hbox.addStretch(1)
        vbox.addLayout(hbox)

        te = QTextEdit()
        te.setReadOnly(True)
        te.setText(self.calib_preview)
        vbox.addWidget(te)

        button_box = QDialogButtonBox(QDialogButtonBox.Close)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        vbox.addWidget(button_box)

        self.setLayout(vbox)


class DialogSaveCalibration(QDialog):

    def __init__(self, parent=None, *, file_path=None):

        super().__init__(parent)

        self.__file_path = ""
        self.__distance_to_sample = 0.0
        self.__overwrite_existing = False
        self.__preview = ("", {})  # str - information, dict - warnings

        self.setWindowTitle("Save Quantitative Calibration")
        self.setMinimumHeight(600)
        self.setMinimumWidth(600)
        self.resize(600, 600)

        self.text_edit = QTextEdit()
        set_tooltip(self.text_edit,
                    "Preview the <b>quantitative calibration data</b> to be saved. The displayed "
                    "warnings will not be saved, but need to be addressed in order to keep "
                    "data integrity. The parameter <b>distance-to-sample</b> is optional, "
                    "but desirable. If <b>distance-to-sample</b> is zero then no scaling will be "
                    "applied to data to compensate for changing distance.")
        self.text_edit.setReadOnly(True)

        self.le_file_path = LineEditReadOnly()
        set_tooltip(self.le_file_path,
                    "Full path to the file used to <b>save the calibration data</b>. The path "
                    "can be changed in file selection dialog box.")
        self.pb_file_path = PushButtonMinimumWidth("..")
        set_tooltip(self.pb_file_path,
                    "Change <b>file path</b> for saving the calibration data.")
        self.pb_file_path.clicked.connect(self.pb_file_path_clicked)
        self.pb_file_path.setDefault(False)
        self.pb_file_path.setAutoDefault(False)

        self.le_distance_to_sample = LineEditExtended()
        self.le_distance_to_sample.textChanged.connect(self.le_distance_to_sample_text_changed)
        self.le_distance_to_sample.editingFinished.connect(self.le_distance_to_sample_editing_finished)
        self._le_distance_to_sample_validator = DoubleValidatorStrict()
        self._le_distance_to_sample_validator.setBottom(0.0)
        set_tooltip(self.le_distance_to_sample,
                    "<b>Distance</b> between the detector and the sample during calibration. If the value "
                    "is 0, then no scaling is applied to data to correct the data if distance-to-sample "
                    "is changed between calibration and measurement.")

        self.cb_overwrite = QCheckBox("Overwrite Existing")
        self.cb_overwrite.stateChanged.connect(self.cb_overwrite_state_changed)
        set_tooltip(self.cb_overwrite,
                    "Overwrite the <b>existing</b> file. This is a safety feature implemented to protect "
                    "valuable results from accidental deletion.")

        vbox = QVBoxLayout()

        hbox = QHBoxLayout()
        hbox.addWidget(QLabel("The following data will be saved to JSON file:"))
        hbox.addStretch(1)
        vbox.addLayout(hbox)

        vbox.addWidget(self.text_edit)

        hbox = QHBoxLayout()
        hbox.addWidget(QLabel("Path: "))
        hbox.addWidget(self.pb_file_path)
        hbox.addWidget(self.le_file_path)
        vbox.addLayout(hbox)

        hbox = QHBoxLayout()
        hbox.addWidget(QLabel("Distance-to-sample:"))
        hbox.addWidget(self.le_distance_to_sample)
        hbox.addStretch(1)
        hbox.addWidget(self.cb_overwrite)
        vbox.addLayout(hbox)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.pb_ok = button_box.button(QDialogButtonBox.Ok)
        self.pb_ok.setDefault(False)
        self.pb_ok.setAutoDefault(False)
        self.pb_cancel = button_box.button(QDialogButtonBox.Cancel)
        self.pb_cancel.setDefault(True)
        self.pb_cancel.setAutoDefault(True)

        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        vbox.addWidget(button_box)

        self.setLayout(vbox)

        self._show_distance_to_sample()
        self._show_preview()
        self._show_overwrite_existing()

        # Set and display file path
        if file_path is not None:
            self.file_path = file_path

    @property
    def file_path(self):
        return self.__file_path

    @file_path.setter
    def file_path(self, file_path):
        file_path = os.path.expanduser(file_path)
        self.__file_path = file_path
        self.le_file_path.setText(file_path)

    @property
    def distance_to_sample(self):
        return self.__distance_to_sample

    @distance_to_sample.setter
    def distance_to_sample(self, distance_to_sample):
        self.__distance_to_sample = distance_to_sample
        self._show_distance_to_sample()

    @property
    def overwrite_existing(self):
        return self.__overwrite_existing

    @overwrite_existing.setter
    def overwrite_existing(self, overwrite_existing):
        self.__overwrite_existing = overwrite_existing
        self._show_overwrite_existing()

    @property
    def preview(self):
        return self.__preview

    @preview.setter
    def preview(self, preview):
        self.__preview = preview
        self._show_preview()

    def pb_file_path_clicked(self):
        file_path = QFileDialog.getSaveFileName(self, "Select File to Save Quantitative Calibration",
                                                self.file_path,
                                                "JSON (*.json);; All (*)",
                                                options=QFileDialog.DontConfirmOverwrite)
        file_path = file_path[0]
        if file_path:
            self.file_path = file_path

    def le_distance_to_sample_text_changed(self, text):
        valid = self._le_distance_to_sample_validator.validate(text, 0)[0] == QDoubleValidator.Acceptable
        self.le_distance_to_sample.setValid(valid)
        self.pb_ok.setEnabled(valid)

    def le_distance_to_sample_editing_finished(self):
        text = self.le_distance_to_sample.text()
        if self._le_distance_to_sample_validator.validate(text, 0)[0] == QDoubleValidator.Acceptable:
            self.__distance_to_sample = float(text)
        self._show_distance_to_sample()
        self._show_preview()  # Show/hide warning on zero distance-to-sample value

    def cb_overwrite_state_changed(self, state):
        state = state == Qt.Checked
        self.__overwrite_existing = state

    def _show_distance_to_sample(self):
        self.le_distance_to_sample.setText(f"{self.__distance_to_sample:.10g}")

    def _show_overwrite_existing(self):
        self.cb_overwrite.setChecked(Qt.Checked if self.__overwrite_existing
                                     else Qt.Unchecked)

    def _show_preview(self):
        text = ""
        # First print warnings
        for key, value in self.__preview[1].items():
            if "distance" in key and self.__distance_to_sample > 0:
                continue
            text += value + "\n"
        # Additional space if there are any warinings
        if len(text):
            text += "\n"
        # Then add the main block of text
        text += self.__preview[0]
        self.text_edit.setText(text)


class DialogExportToTiffAndTxt(QDialog):

    def __init__(self, parent=None, *, dir_path=""):

        super().__init__(parent)

        self.__dir_path = ""
        self.__save_tiff = True
        self.__save_txt = False
        self.__interpolate_on = False
        self.__quant_norm_on = False
        self.__dset_list = []
        self.__dset_sel = 0
        self.__scaler_list = []
        self.__scaler_sel = 0

        self.setWindowTitle("Export XRF Maps to TIFF and/or TXT Files")
        self.setMinimumHeight(600)
        self.setMinimumWidth(600)
        self.resize(600, 600)

        self.te_saved_files = QTextEdit()
        set_tooltip(self.te_saved_files,
                    "The list of <b>data file groups</b> about to be created.")
        self.te_saved_files.setReadOnly(True)

        self.combo_select_dataset = QComboBox()
        self.combo_select_dataset.currentIndexChanged.connect(
            self.combo_select_dataset_current_index_changed)
        self._fill_dataset_combo()
        set_tooltip(self.combo_select_dataset,
                    "Select <b>dataset</b>. Initially, the selection matches the dataset activated "
                    "in <b>XRF Maps</b> tab, but the selection may be changed if different dataset "
                    "needs to be saved.")

        self.combo_normalization = QComboBox()
        self.combo_normalization.currentIndexChanged.connect(
            self.combo_normalization_current_index_changed)
        self._fill_scaler_combo()
        set_tooltip(self.combo_normalization,
                    "Select <b>scaler</b> used for data normalization. Initially, the selection matches "
                    "the scaler activated in <b>XRF Maps</b> tab, but the selection may be changed "
                    "if needed")

        self.cb_interpolate = QCheckBox("Interpolate to uniform grid")
        self.cb_interpolate.setChecked(Qt.Checked if self.__interpolate_on else Qt.Unchecked)
        self.cb_interpolate.stateChanged.connect(self.cb_interpolate_state_changed)
        set_tooltip(self.cb_interpolate,
                    "Interpolate pixel coordinates to <b>uniform grid</b>. The initial choice is "
                    "copied from <b>XRF Maps</b> tab.")

        self.cb_quantitative = QCheckBox("Quantitative normalization")
        self.cb_quantitative.setChecked(Qt.Checked if self.__quant_norm_on else Qt.Unchecked)
        self.cb_quantitative.stateChanged.connect(self.cb_quantitative_state_changed)
        set_tooltip(self.cb_quantitative,
                    "Apply <b>quantitative normalization</b> before saving the maps. "
                    "The initial choice is copied from <b>XRF Maps</b> tab.")

        self.group_settings = QGroupBox("Settings (selections from XRF Maps tab)")
        grid = QGridLayout()
        grid.addWidget(self.combo_select_dataset, 0, 0)
        grid.addWidget(self.combo_normalization, 0, 1)
        grid.addWidget(self.cb_interpolate, 1, 0)
        grid.addWidget(self.cb_quantitative, 1, 1)
        self.group_settings.setLayout(grid)

        self.le_dir_path = LineEditReadOnly()
        set_tooltip(self.le_dir_path,
                    "<b>Root directory</b> for saving TIFF and TXT files. The files will be saved "
                    "in subdirectories inside the root directory.")
        self.pb_dir_path = PushButtonMinimumWidth("..")
        set_tooltip(self.pb_dir_path,
                    "Change to <b>root directory</b> for TIFF and TXT files.")
        self.pb_dir_path.clicked.connect(self.pb_dir_path_clicked)
        self.pb_dir_path.setDefault(False)
        self.pb_dir_path.setAutoDefault(False)

        self.cb_save_tiff = QCheckBox("Save TIFF")
        set_tooltip(self.cb_save_tiff,
                    "Save XRF Maps as <b>TIFF</b> files.")
        self.cb_save_tiff.setChecked(Qt.Checked if self.__save_tiff else Qt.Unchecked)
        self.cb_save_tiff.stateChanged.connect(self.cb_save_tiff_state_changed)
        self.cb_save_txt = QCheckBox("Save TXT")
        self.cb_save_txt.setChecked(Qt.Checked if self.__save_txt else Qt.Unchecked)
        set_tooltip(self.cb_save_txt,
                    "Save XRF Maps as <b>TXT</b> files.")
        self.cb_save_txt.stateChanged.connect(self.cb_save_txt_state_changed)

        vbox = QVBoxLayout()

        hbox = QHBoxLayout()
        hbox.addWidget(self.group_settings)
        hbox.addStretch(1)
        vbox.addLayout(hbox)

        hbox = QHBoxLayout()
        hbox.addWidget(QLabel("The following groups of files will be created:"))
        hbox.addStretch(1)
        vbox.addLayout(hbox)
        vbox.addWidget(self.te_saved_files)

        hbox = QHBoxLayout()
        hbox.addWidget(QLabel("Directory: "))
        hbox.addWidget(self.pb_dir_path)
        hbox.addWidget(self.le_dir_path)
        vbox.addLayout(hbox)

        self.button_box = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        self.pb_save = self.button_box.button(QDialogButtonBox.Save)
        self.pb_save.setDefault(False)
        self.pb_save.setAutoDefault(False)
        self.pb_cancel = self.button_box.button(QDialogButtonBox.Cancel)
        self.pb_cancel.setDefault(True)
        self.pb_cancel.setAutoDefault(True)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        hbox = QHBoxLayout()
        hbox.addStretch(1)
        hbox.addWidget(self.cb_save_tiff)
        hbox.addWidget(self.cb_save_txt)
        hbox.addStretch(1)
        hbox.addWidget(self.button_box)
        vbox.addLayout(hbox)

        self.setLayout(vbox)

        # Set and display file path
        dir_path = os.path.expanduser(dir_path)
        self.dir_path = dir_path

        self._update_pb_save()
        self._update_saved_file_groups()

    @property
    def dir_path(self):
        return self.__dir_path

    @dir_path.setter
    def dir_path(self, dir_path):
        self.__dir_path = dir_path
        self.le_dir_path.setText(dir_path)

    @property
    def save_tiff(self):
        return self.__save_tiff

    @save_tiff.setter
    def save_tiff(self, save_tiff):
        self.__save_tiff = save_tiff
        self.cb_save_tiff.setChecked(Qt.Checked if save_tiff else Qt.Unchecked)
        self._update_pb_save()

    @property
    def save_txt(self):
        return self.__save_txt

    @save_txt.setter
    def save_txt(self, save_txt):
        self.__save_txt = save_txt
        self.cb_save_txt.setChecked(Qt.Checked if save_txt else Qt.Unchecked)
        self._update_pb_save()

    @property
    def interpolate_on(self):
        return self.__interpolate_on

    @interpolate_on.setter
    def interpolate_on(self, interpolate_on):
        self.__interpolate_on = interpolate_on
        self.cb_interpolate.setChecked(Qt.Checked if interpolate_on else Qt.Unchecked)

    @property
    def quant_norm_on(self):
        return self.__quant_norm_on

    @quant_norm_on.setter
    def quant_norm_on(self, quant_norm_on):
        self.__quant_norm_on = quant_norm_on
        self.cb_quantitative.setChecked(Qt.Checked if quant_norm_on else Qt.Unchecked)

    @property
    def dset_list(self):
        return self.__dset_list

    @dset_list.setter
    def dset_list(self, dset_list):
        self.__dset_list = dset_list
        self._fill_dataset_combo()

    @property
    def dset_sel(self):
        return self.__dset_sel

    @dset_sel.setter
    def dset_sel(self, dset_sel):
        self.__dset_sel = dset_sel
        self.combo_select_dataset.setCurrentIndex(dset_sel - 1)

    @property
    def scaler_list(self):
        return self.__scaler_list

    @scaler_list.setter
    def scaler_list(self, scaler_list):
        self.__scaler_list = scaler_list
        self._fill_scaler_combo()

    @property
    def scaler_sel(self):
        return self.__scaler_sel

    @scaler_sel.setter
    def scaler_sel(self, scaler_sel):
        self.__scaler_sel = scaler_sel
        self.combo_normalization.setCurrentIndex(scaler_sel)

    def pb_dir_path_clicked(self):
        # Note: QFileDialog.ShowDirsOnly is not set on purpose, so that the dialog
        #   could be used to inspect directory contents. Files can not be selected anyway.
        dir_path = QFileDialog.getExistingDirectory(self, "Select Root Directory for TIFF and TXT Files",
                                                    self.dir_path,
                                                    options=QFileDialog.DontResolveSymlinks)
        if dir_path:
            self.dir_path = dir_path

    def _update_pb_save(self):
        state = self.__save_tiff or self.__save_txt
        self.pb_save.setEnabled(state)

    def cb_save_tiff_state_changed(self, state):
        self.__save_tiff = (state == Qt.Checked)
        self._update_pb_save()
        self._update_saved_file_groups()

    def cb_save_txt_state_changed(self, state):
        self.__save_txt = (state == Qt.Checked)
        self._update_pb_save()
        self._update_saved_file_groups()

    def cb_interpolate_state_changed(self, state):
        self.__interpolate_on = (state == Qt.Checked)

    def cb_quantitative_state_changed(self, state):
        self.__quant_norm_on = (state == Qt.Checked)
        self._update_saved_file_groups()

    def combo_select_dataset_current_index_changed(self, index):
        self.__dset_sel = index + 1
        self._update_saved_file_groups()

    def combo_normalization_current_index_changed(self, index):
        self.__scaler_sel = index
        self._update_saved_file_groups()

    def get_selected_dset_name(self):
        index = self.__dset_sel - 1
        n = len(self.__dset_list)
        if index < 0 or index >= n:
            return None
        else:
            return self.__dset_list[index]

    def get_selected_scaler_name(self):
        index = self.__scaler_sel - 1
        n = len(self.__scaler_list)
        if index < 0 or index >= n:
            return None
        else:
            return self.__scaler_list[index]

    def _fill_dataset_combo(self):
        self.combo_select_dataset.clear()
        self.combo_select_dataset.addItems(self.__dset_list)
        self.combo_select_dataset.setCurrentIndex(self.__dset_sel - 1)

    def _fill_scaler_combo(self):
        scalers = ["Normalize by ..."] + self.__scaler_list
        self.combo_normalization.clear()
        self.combo_normalization.addItems(scalers)
        self.combo_normalization.setCurrentIndex(self.__scaler_sel)

    def _update_saved_file_groups(self):
        dset_name = self.get_selected_dset_name()
        scaler_name = self.get_selected_scaler_name()
        is_fit, is_roi = False, False
        if dset_name is not None:
            is_fit = dset_name.endswith("fit")
            is_roi = dset_name.endswith("roi")

        text_common = ""
        if is_fit:
            text_common += "  - Fitted XRF maps\n"
        elif is_roi:
            text_common += "  - ROI maps\n"
        if (is_fit or is_roi) and (scaler_name is not None):
            text_common += f"  - Normalized maps (scaler '{scaler_name}')\n"
        if is_fit and self.__quant_norm_on:
            text_common += "  - Quantitative maps (if calibration data is loaded)\n"
        text_common += "  - Scalers\n"
        text_common += "  - Positional coordinates\n"

        text = ""
        if self.__save_tiff:
            text += "TIFF FORMAT:\n" + text_common
        if self.__save_txt:
            if text:
                text += "\n"
            text += "TXT_FORMAT:\n" + text_common
        if not text:
            text = "No files will be saved"
        self.te_saved_files.setText(text)
