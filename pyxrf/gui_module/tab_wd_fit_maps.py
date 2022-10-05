import os

from qtpy.QtWidgets import QPushButton, QVBoxLayout, QGroupBox, QLabel, QGridLayout, QMessageBox
from qtpy.QtCore import Slot, Signal, QThreadPool, QRunnable

from .useful_widgets import global_gui_parameters, set_tooltip, LineEditExtended, IntValidatorStrict
from .form_base_widget import FormBaseWidget
from .dlg_export_to_tiff_and_txt import DialogExportToTiffAndTxt
from .dlg_save_calibration import DialogSaveCalibration

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
            "included in the selection.",
        )
        set_tooltip(
            self.le_start_row,
            "Number of the <b>first row</b> of the map to be included in the selection. "
            "The number must be less than the number entered into 'End row' box.",
        )
        set_tooltip(
            self.le_start_col,
            "Number of the <b>first column</b> of the map to be included in the selection. "
            "The number must be less than the number entered into 'End column' box.",
        )
        set_tooltip(
            self.le_end_row,
            "Number of the <b>last row</b> included in the selection. "
            "The number must be greater than the number entered into 'Start row' box.",
        )
        set_tooltip(
            self.le_end_col,
            "Number of the <b>last column</b> included in the selection. "
            "The number must be greater than the number entered into 'Start column' box.",
        )

        set_tooltip(
            self.pb_start_map_fitting,
            "Click to start <b>fitting of the XRF Maps</b>. The generated XRF Maps can be viewed "
            "in <b>'XRF Maps' tab</b>",
        )

        set_tooltip(
            self.pb_compute_roi_maps,
            "Opens the window for setting up <b>spectral ROIs</b> and computating XRF Maps based on the ROIs",
        )

        set_tooltip(self.pb_save_to_db, "Save generated XRF Maps to a <b>database</b> via Databroker")

        set_tooltip(
            self.pb_save_q_calibration,
            "Opens a Dialog Box which allows to preview and save <b>Quantitative Calibration data</b>",
        )
        set_tooltip(
            self.pb_export_to_tiff_and_txt,
            "Open a Dialog box which allows to export XRF Maps as <b>TIFF</b> and <b>TXT</b> files",
        )

        set_tooltip(
            self.pb_load_quant_calib,
            "Open a window with GUI tools for loading and managing previously saved "
            "<b>Quantitative Calibration data</b> used for processing (normalization) "
            "of XRF Maps. The loaded calibration data is applied to XRF Maps if 'Quantitative' "
            "box is checked in 'XRF Maps' tab",
        )

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
        selected_area = {"row_min": 1, "row_max": 1, "col_min": 1, "col_max": 1}
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
            msg += (
                "No quantitative standard is selected. "
                "Use 'Load Quantitative Standard ...' button in 'Model' tab"
            )
        if not self.gpc.is_quant_standard_fitting_available():
            msg = msg + "\n" if msg else msg
            msg += (
                "Select a dataset containing fitted XRF maps for the quantitative standard "
                "in XRF Maps tab (dataset name must end with 'fit')."
            )
        if msg:
            msgbox = QMessageBox(
                QMessageBox.Information, "Additional Steps Needed", msg, QMessageBox.Ok, parent=self
            )
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
            msgbox = QMessageBox(QMessageBox.Critical, "Error", msg, QMessageBox.Ok, parent=self)
            msgbox.exec()

    def pb_export_to_tiff_and_txt_clicked(self):
        # TODO: Propagate full path to the saved file here
        dir_path = self.gpc.get_file_directory()
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
                self.gpc.export_xrf_maps(
                    results_path=result_path,
                    dataset_name=dataset_name,
                    scaler_name=scaler_name,
                    interpolate_on=interpolate_on,
                    quant_norm_on=quant_norm_on,
                    file_formats=file_formats,
                )
                if file_formats:
                    formats_text = " and ".join([_.upper() for _ in file_formats])
                else:
                    formats_text = "No"
                msg = f"{formats_text} files were saved to the directory '{result_path}'"
                logger.info(msg)

                msgbox = QMessageBox(QMessageBox.Information, "Files Saved", msg, QMessageBox.Ok, parent=self)
                msgbox.exec()

            except Exception as ex:
                msg = str(ex)
                msgbox = QMessageBox(QMessageBox.Critical, "Error", msg, QMessageBox.Ok, parent=self)
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
            msgbox = QMessageBox(
                QMessageBox.Critical, "Failed to Fit Individual Pixel Spectra", msg, QMessageBox.Ok, parent=self
            )
            msgbox.exec()

        self.signal_map_fitting_complete.emit()
        self.update_global_state.emit()
        if success:
            self.signal_activate_tab_xrf_maps.emit()

    """
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
    """

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
        self._area_row_min = area["row_min"]
        self._area_row_max = area["row_max"]
        self._area_col_min = area["col_min"]
        self._area_col_max = area["col_max"]
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
        area = {
            "row_min": self._area_row_min,
            "row_max": self._area_row_max,
            "col_min": self._area_col_min,
            "col_max": self._area_col_max,
        }
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
