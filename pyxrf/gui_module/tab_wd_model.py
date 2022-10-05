import os

from qtpy.QtWidgets import (
    QPushButton,
    QHBoxLayout,
    QVBoxLayout,
    QGroupBox,
    QLabel,
    QComboBox,
    QFileDialog,
    QMessageBox,
)
from qtpy.QtCore import Slot, Signal, QThreadPool, QRunnable

from .useful_widgets import LineEditReadOnly, global_gui_parameters, set_tooltip

from .form_base_widget import FormBaseWidget
from .dlg_find_elements import DialogFindElements
from .dlg_select_quant_standard import DialogSelectQuantStandard
from .wnd_detailed_fitting_params import fitting_preset_names

import logging

logger = logging.getLogger(__name__)


class ModelWidget(FormBaseWidget):

    # Signal that is sent (to main window) to update global state of the program
    update_global_state = Signal()
    computations_complete = Signal(object)
    # Signal is emitted when a new model is loaded (or computed).
    # True - model loaded successfully, False - otherwise
    # In particular, the signal may be used to update the widgets that depend on incident energy,
    #   because it may change as the model is loaded.
    signal_model_loaded = Signal(bool)
    # Incident energy or selected range changed (plots need to be redrawn)
    signal_incident_energy_or_range_changed = Signal()
    # Sent after the completion of total spectrum fitting
    signal_total_spectrum_fitting_completed = Signal(bool)

    def __init__(self, *, gpc, gui_vars):
        super().__init__()

        self._fit_available = False
        # Currently selected emission line.
        self._selected_eline = ""

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

        vbox = QVBoxLayout()

        self._setup_model_params_group()
        vbox.addWidget(self.group_model_params)
        vbox.addSpacing(v_spacing)

        self._setup_add_remove_elines_button()
        vbox.addWidget(self.pb_manage_emission_lines)
        vbox.addSpacing(v_spacing)

        self._setup_settings_group()
        vbox.addWidget(self.group_settings)
        vbox.addSpacing(v_spacing)

        self._setup_model_fitting_group()
        vbox.addWidget(self.group_model_fitting)

        self.setLayout(vbox)

        self._set_tooltips()

        # Timer is currently used to simulate processing
        self._timer = None
        self._timer_counter = 0

    def _setup_model_params_group(self):

        self.group_model_params = QGroupBox("Load/Save Model Parameters")

        self.pb_find_elines = QPushButton("Find Automatically ...")
        self.pb_find_elines.clicked.connect(self.pb_find_elines_clicked)

        self.pb_load_elines = QPushButton("Load From File ...")
        self.pb_load_elines.clicked.connect(self.pb_load_elines_clicked)

        self.pb_load_qstandard = QPushButton("Load Quantitative Standard ...")
        self.pb_load_qstandard.clicked.connect(self.pb_load_qstandard_clicked)

        self.pb_save_elines = QPushButton("Save Parameters to File ...")
        self.pb_save_elines.clicked.connect(self.pb_save_elines_clicked)

        # This field will display the name of he last loaded parameter file,
        #   Serial/Name of the quantitative standard, or 'no parameters' message
        self.le_param_fln = LineEditReadOnly("No parameter file is loaded")

        vbox = QVBoxLayout()
        hbox = QHBoxLayout()
        hbox.addWidget(self.pb_find_elines)
        hbox.addWidget(self.pb_load_elines)
        vbox.addLayout(hbox)
        hbox = QHBoxLayout()
        hbox.addWidget(self.pb_save_elines)
        vbox.addLayout(hbox)
        hbox = QHBoxLayout()
        hbox.addWidget(self.pb_load_qstandard)
        vbox.addLayout(hbox)
        vbox.addWidget(self.le_param_fln)

        self.group_model_params.setLayout(vbox)

    def _setup_add_remove_elines_button(self):

        self.pb_manage_emission_lines = QPushButton("Add/Remove Emission Lines ...")
        self.pb_manage_emission_lines.clicked.connect(self.pb_manage_emission_lines_clicked)

    def _setup_settings_group(self):

        self.group_settings = QGroupBox("Settings for Fitting Algorithm")

        self.pb_fit_param_general = QPushButton("General ...")
        self.pb_fit_param_general.clicked.connect(self.pb_fit_param_general_clicked)

        self.pb_fit_param_shared = QPushButton("Shared ...")
        self.pb_fit_param_shared.clicked.connect(self.pb_fit_param_shared_clicked)

        self.pb_fit_param_lines = QPushButton("Lines ...")
        self.pb_fit_param_lines.clicked.connect(self.pb_fit_param_lines_clicked)

        fit_strategy_list = self.gpc.get_fit_strategy_list()
        combo_items = [fitting_preset_names[_] for _ in fit_strategy_list]
        combo_items = ["None"] + combo_items
        self.cb_step1 = QComboBox()
        self.cb_step1.setMinimumWidth(150)
        self.cb_step1.addItems(combo_items)
        self.cb_step1.setCurrentIndex(1)  # Should also be set based on data
        self.cb_step2 = QComboBox()
        self.cb_step2.setMinimumWidth(150)
        self.cb_step2.addItems(combo_items)

        vbox = QVBoxLayout()
        hbox = QHBoxLayout()
        hbox.addWidget(self.pb_fit_param_general)
        hbox.addWidget(self.pb_fit_param_shared)
        hbox.addWidget(self.pb_fit_param_lines)
        vbox.addLayout(hbox)

        hbox = QHBoxLayout()
        hbox.addWidget(QLabel("Fitting step 1:"))
        hbox.addSpacing(20)
        hbox.addWidget(self.cb_step1)
        hbox.addStretch(1)
        vbox.addLayout(hbox)

        hbox = QHBoxLayout()
        hbox.addWidget(QLabel("Fitting step 2:"))
        hbox.addSpacing(20)
        hbox.addWidget(self.cb_step2)
        hbox.addStretch(1)
        vbox.addLayout(hbox)

        self.group_settings.setLayout(vbox)

    def _setup_model_fitting_group(self):

        self.group_model_fitting = QGroupBox("Model Fitting Based on Total Spectrum")

        self.pb_start_fitting = QPushButton("Start Fitting")
        self.pb_start_fitting.clicked.connect(self.pb_start_fitting_clicked)

        self.pb_save_spectrum = QPushButton("Save Spectrum/Fit ...")
        self.pb_save_spectrum.clicked.connect(self.pb_save_spectrum_clicked)

        self.le_fitting_results = LineEditReadOnly()

        vbox = QVBoxLayout()

        hbox = QHBoxLayout()
        hbox.addWidget(self.pb_start_fitting)
        hbox.addWidget(self.pb_save_spectrum)
        vbox.addLayout(hbox)

        vbox.addWidget(self.le_fitting_results)

        self.group_model_fitting.setLayout(vbox)

    def _set_tooltips(self):
        set_tooltip(self.pb_find_elines, "Automatically find emission lines from <b>total spectrum</b>.")
        set_tooltip(
            self.pb_load_elines,
            "Load model parameters, including selected emission lines from <b>JSON</b> file, "
            "which was previously save using <b>Save Parameters to File ...</b>.",
        )
        set_tooltip(
            self.pb_load_qstandard,
            "Load <b>quantitative standard</b>. The model is reset and the emission lines "
            "that fit within the selected range of energies are added to the list "
            "of emission lines.",
        )
        set_tooltip(
            self.pb_save_elines,
            "Save the model parameters including the parameters of the selected emission lines "
            "to <b>JSON</b> file.",
        )
        set_tooltip(
            self.le_param_fln,
            "The name of the recently loaded <b>parameter file</b> or serial number "
            "and name of the loaded <b>quantitative standard</b>",
        )

        set_tooltip(
            self.pb_manage_emission_lines,
            "Open a user friendly interface that allows to <b>add and remove emission lines</b> "
            "to the list or <b>modify parameters</b> of the selected emission lines",
        )

        set_tooltip(self.pb_fit_param_general, "<b>General settings</b> for fitting algorithms.")
        set_tooltip(
            self.pb_fit_param_shared,
            "Access to low-level control of the total spectrum fitting algorithm: parameters shared "
            "by models of all emission lines.",
        )
        set_tooltip(
            self.pb_fit_param_lines,
            "Access to low-level control of the total spectrum fitting algorithm: adjust parameters "
            "for each emission line of the selected elements; modify preset fitting configurations.",
        )
        set_tooltip(
            self.cb_step1,
            "Select preset fitting configuration for <b>Step 1</b>. "
            "Click <b>Elements...</b> and <b>Global Parameters...</b> "
            "buttons to open dialog boxes to configure the presets.",
        )
        set_tooltip(
            self.cb_step2,
            "Select preset fitting configuration for <b>Step 2</b>. "
            "Click <b>Elements...</b> and <b>Global Parameters...</b> "
            "buttons to open dialog boxes to configure the presets.",
        )

        set_tooltip(
            self.pb_start_fitting,
            "Click the button to <b>run fitting of total spectrum</b>. The result of fitting includes "
            "the refined set of emission line parameters. The fitted spectrum is displayed in "
            "<b>'Fitting Model'</b> tab and can be saved by clicking <b>'Save Spectrum/Fit ...'</b> button.",
        )
        set_tooltip(
            self.pb_save_spectrum,
            "Save <b>raw and fitted total spectra</b>. Click <b>'Start Fitting'</b> to perform fitting "
            "before saving the spectrum",
        )
        set_tooltip(self.le_fitting_results, "<b>Output parameters</b> produced by the fitting algorithm")

    def update_widget_state(self, condition=None):
        if condition == "tooltips":
            self._set_tooltips()

        state_file_loaded = self.gui_vars["gui_state"]["state_file_loaded"]
        state_model_exist = self.gui_vars["gui_state"]["state_model_exists"]
        # state_model_fit_exists = self.gui_vars["gui_state"]["state_model_fit_exists"]

        self.group_model_params.setEnabled(state_file_loaded)
        self.pb_save_elines.setEnabled(state_file_loaded & state_model_exist)

        self.pb_manage_emission_lines.setEnabled(state_file_loaded & state_model_exist)

        self.group_settings.setEnabled(state_file_loaded & state_model_exist)

        self.group_settings.setEnabled(state_file_loaded & state_model_exist)

        self.group_model_fitting.setEnabled(state_file_loaded & state_model_exist)
        # self.pb_save_spectrum.setEnabled(state_file_loaded & state_model_exist & state_model_fit_exists)
        self.pb_save_spectrum.setEnabled(state_file_loaded & state_model_exist)

    def pb_find_elines_clicked(self):
        dialog_data = self.gpc.get_autofind_elements_params()
        dlg = DialogFindElements()
        dlg.set_dialog_data(dialog_data)
        ret = dlg.exec()
        if ret:
            dialog_data = dlg.get_dialog_data()
            find_elements_requested = dlg.find_elements_requested
            update_model = self.gui_vars["gui_state"]["state_model_exists"]

            def cb():
                range_changed = self.gpc.set_autofind_elements_params(
                    dialog_data, update_model=update_model, update_fitting_params=not find_elements_requested
                )
                if find_elements_requested:
                    self.gpc.find_elements_automatically()
                return {"range_changed": range_changed, "find_elements_requested": find_elements_requested}

            self._compute_in_background(cb, self.slot_find_elines_clicked)

    @Slot(object)
    def slot_find_elines_clicked(self, result):
        range_changed = result["range_changed"]
        find_elements_requested = result["find_elements_requested"]

        self._set_fit_status(False)
        self._recover_after_compute(self.slot_find_elines_clicked)

        if range_changed:
            self.signal_incident_energy_or_range_changed.emit()
            self.gpc.fitting_parameters_changed()

        if find_elements_requested:
            msg = "Emission lines were detected automatically"
        else:
            msg = "Parameters were upadated"
        self.le_param_fln.setText(msg)

        if find_elements_requested:
            self.gui_vars["gui_state"]["state_model_exists"] = True
            self.gui_vars["gui_state"]["state_model_fit_exists"] = False
            self.signal_model_loaded.emit(True)
            self.update_global_state.emit()
            logger.info("Automated element search is complete")

    @Slot(str)
    def slot_selection_item_changed(self, eline):
        self._selected_eline = eline

    def _get_load_elines_cb(self):
        def cb(file_name, incident_energy_from_param_file=None):
            try:
                completed, question = self.gpc.load_parameters_from_file(
                    file_name, incident_energy_from_param_file
                )
                success = True
                change_state = True
                msg = ""
            except IOError as ex:
                completed, question = True, ""
                success = False
                change_state = False
                msg = str(ex)
            except Exception as ex:
                completed, question = True, ""
                success = False
                change_state = True
                msg = str(ex)

            result_dict = {
                "completed": completed,
                "question": question,
                "success": success,
                "change_state": change_state,
                "msg": msg,
                "file_name": file_name,
            }
            return result_dict

        return cb

    def pb_load_elines_clicked(self):
        current_dir = self.gpc.get_current_working_directory()
        file_name = QFileDialog.getOpenFileName(
            self, "Select File with Model Parameters", current_dir, "JSON (*.json);; All (*)"
        )
        file_name = file_name[0]
        if file_name:
            cb = self._get_load_elines_cb()
            self._compute_in_background(
                cb, self.slot_load_elines_clicked, file_name=file_name, incident_energy_from_param_file=None
            )

    @Slot(object)
    def slot_load_elines_clicked(self, results):
        self._recover_after_compute(self.slot_load_elines_clicked)

        completed = results["completed"]
        file_name = results["file_name"]
        msg = results["msg"]

        if not completed:
            mb = QMessageBox(
                QMessageBox.Question,
                "Question",
                results["question"],
                QMessageBox.Yes | QMessageBox.No,
                parent=self,
            )
            answer = mb.exec() == QMessageBox.Yes
            cb = self._get_load_elines_cb()
            self._compute_in_background(
                cb, self.slot_load_elines_clicked, file_name=file_name, incident_energy_from_param_file=answer
            )
            return

        if results["success"]:
            _, fln = os.path.split(file_name)
            msg = f"File: '{fln}'"
            self.le_param_fln.setText(msg)

            self._set_fit_status(False)

            self.gui_vars["gui_state"]["state_model_exists"] = True
            self.gui_vars["gui_state"]["state_model_fit_exists"] = False
            self.signal_model_loaded.emit(True)
            self.update_global_state.emit()
        else:
            if results["change_state"]:
                logger.error(f"Exception: error occurred while loading parameters: {msg}")
                mb_error = QMessageBox(
                    QMessageBox.Critical,
                    "Error",
                    f"Error occurred while processing loaded parameters: {msg}",
                    QMessageBox.Ok,
                    parent=self,
                )
                mb_error.exec()
                # Here the parameters were loaded and processing was partially performed,
                #   so change the state of the program
                self.gui_vars["gui_state"]["state_model_exists"] = False
                self.gui_vars["gui_state"]["state_model_fit_exists"] = False
                self.signal_model_loaded.emit(False)
                self.update_global_state.emit()
            else:
                # It doesn't seem that the state of the program needs to be changed if
                #   the file was not loaded at all
                logger.error(f"Exception: {msg}")
                mb_error = QMessageBox(QMessageBox.Critical, "Error", f"{msg}", QMessageBox.Ok, parent=self)
                mb_error.exec()

    def pb_load_qstandard_clicked(self):
        qe_param_built_in, qe_param_custom, qe_standard_selected = self.gpc.get_quant_standard_list()
        dlg = DialogSelectQuantStandard()
        dlg.set_standards(qe_param_built_in, qe_param_custom, qe_standard_selected)
        ret = dlg.exec()
        if ret:
            selected_standard = dlg.get_selected_standard()

            def cb(selected_standard):
                try:
                    if selected_standard is None:
                        raise RuntimeError("The selected standard is not found.")
                    self.gpc.set_selected_quant_standard(selected_standard)
                    success, msg = True, ""
                except Exception as ex:
                    success, msg = False, str(ex)
                return {"success": success, "msg": msg, "selected_standard": selected_standard}

            self._compute_in_background(cb, self.slot_load_qstandard_clicked, selected_standard=selected_standard)

    @Slot(object)
    def slot_load_qstandard_clicked(self, result):
        self._recover_after_compute(self.slot_load_qstandard_clicked)

        if result["success"]:
            selected_standard = result["selected_standard"]
            msg = f"QS: '{selected_standard['name']}'"
            if self.gpc.is_quant_standard_custom(selected_standard):
                msg += " (user-defined)"
            self.le_param_fln.setText(msg)

            self.gpc.process_peaks_from_quantitative_sample_data()

            self._set_fit_status(False)

            self.gui_vars["gui_state"]["state_model_exists"] = True
            self.gui_vars["gui_state"]["state_model_fit_exists"] = False
            self.signal_model_loaded.emit(True)
            self.update_global_state.emit()
        else:
            msg = result["msg"]
            msgbox = QMessageBox(
                QMessageBox.Critical, "Failed to Load Quantitative Standard", msg, QMessageBox.Ok, parent=self
            )
            msgbox.exec()

    def pb_save_elines_clicked(self):
        current_dir = self.gpc.get_current_working_directory()
        scan_id = self.gpc.get_metadata_scan_id()
        fln = os.path.join(current_dir, f"pyxrf_model_parameters_{scan_id}.json")
        file_name = QFileDialog.getSaveFileName(
            self, "Select File to Save Model Parameters", fln, "JSON (*.json);; All (*)"
        )
        file_name = file_name[0]
        if file_name:
            try:
                self.gpc.save_param_to_file(file_name)
                logger.debug(f"Model parameters were saved to the file '{file_name}'")
            except Exception as ex:
                msg = str(ex)
                msgbox = QMessageBox(QMessageBox.Critical, "Error", msg, QMessageBox.Ok, parent=self)
                msgbox.exec()
        else:
            logger.debug("Saving model parameters was skipped.")

    def pb_manage_emission_lines_clicked(self):
        # Position the window in relation ot the main window (only when called once)
        pos = self.ref_main_window.pos()
        self.ref_main_window.wnd_manage_emission_lines.position_once(pos.x(), pos.y())

        if not self.ref_main_window.wnd_manage_emission_lines.isVisible():
            self.ref_main_window.wnd_manage_emission_lines.show()
        self.ref_main_window.wnd_manage_emission_lines.activateWindow()

    def pb_fit_param_general_clicked(self):
        # Position the window in relation ot the main window (only when called once)
        pos = self.ref_main_window.pos()
        self.ref_main_window.wnd_general_fitting_settings.position_once(pos.x(), pos.y())

        if not self.ref_main_window.wnd_general_fitting_settings.isVisible():
            self.ref_main_window.wnd_general_fitting_settings.show()
        self.ref_main_window.wnd_general_fitting_settings.activateWindow()

    def pb_fit_param_shared_clicked(self):
        # Position the window in relation ot the main window (only when called once)
        pos = self.ref_main_window.pos()
        self.ref_main_window.wnd_fitting_parameters_shared.position_once(pos.x(), pos.y())

        if not self.ref_main_window.wnd_fitting_parameters_shared.isVisible():
            self.ref_main_window.wnd_fitting_parameters_shared.show()
        self.ref_main_window.wnd_fitting_parameters_shared.activateWindow()

    def pb_fit_param_lines_clicked(self):
        # Position the window in relation ot the main window (only when called once)
        pos = self.ref_main_window.pos()
        self.ref_main_window.wnd_fitting_parameters_lines.position_once(pos.x(), pos.y())

        if not self.ref_main_window.wnd_fitting_parameters_lines.isVisible():
            self.ref_main_window.wnd_fitting_parameters_lines.show()
        self.ref_main_window.wnd_fitting_parameters_lines.activateWindow()

    def pb_save_spectrum_clicked(self):
        current_dir = self.gpc.get_current_working_directory()
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Directory to Save Spectrum/Fit",
            current_dir,
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks,
        )
        if directory:
            try:
                self.gpc.save_spectrum(directory, save_fit=self._fit_available)
                logger.debug(f"Spectrum/Fit is saved to directory {directory}")
            except Exception as ex:
                msg = str(ex)
                msgbox = QMessageBox(QMessageBox.Critical, "Error", msg, QMessageBox.Ok, parent=self)
                msgbox.exec()
        else:
            logger.debug("Spectrum/Fit saving is cancelled")

    def pb_start_fitting_clicked(self):
        def cb():
            try:
                self.gpc.total_spectrum_fitting()
                success, msg = True, ""
            except Exception as ex:
                success, msg = False, str(ex)

            return {"success": success, "msg": msg}

        self._compute_in_background(cb, self.slot_start_fitting_clicked)

    @Slot(object)
    def slot_start_fitting_clicked(self, result):
        self._recover_after_compute(self.slot_start_fitting_clicked)

        success = result["success"]
        if success:
            self._set_fit_status(True)
        else:
            msg = result["msg"]
            msgbox = QMessageBox(
                QMessageBox.Critical, "Failed to Fit Total Spectrum", msg, QMessageBox.Ok, parent=self
            )
            msgbox.exec()

        # Reload the table
        self.signal_total_spectrum_fitting_completed.emit(success)

    def _update_le_fitting_results(self):
        rf = self.gpc.compute_current_rfactor(self._fit_available)
        rf_text = f"{rf:.4f}" if rf is not None else "n/a"
        if self._fit_available:
            _ = self.gpc.get_iter_and_var_number()
            iter = _["iter_number"]
            nvar = _["var_number"]
            self.le_fitting_results.setText(f"Iterations: {iter}  Variables: {nvar}  R-factor: {rf_text}")
        else:
            self.le_fitting_results.setText(f"R-factor: {rf_text}")

    @Slot()
    def update_fit_status(self):
        self._fit_available = self.gui_vars["gui_state"]["state_model_fit_exists"]
        self._update_le_fitting_results()

    @Slot()
    def clear_fit_status(self):
        # Clear fit status (reset it to False - no valid fit is available)
        self.gui_vars["gui_state"]["state_model_fit_exists"] = False
        self.update_fit_status()

    def _set_fit_status(self, status):
        self.gui_vars["gui_state"]["state_model_fit_exists"] = status
        self.update_fit_status()

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
