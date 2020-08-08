import os
import numpy as np
from threading import Thread
import copy

from PyQt5.QtWidgets import (QPushButton, QHBoxLayout, QVBoxLayout, QGroupBox, QLineEdit,
                             QCheckBox, QLabel, QComboBox, QDialog, QDialogButtonBox,
                             QFileDialog, QRadioButton, QButtonGroup, QGridLayout, QTableWidget,
                             QTableWidgetItem, QHeaderView, QMessageBox)
from PyQt5.QtGui import QBrush, QColor, QDoubleValidator, QRegExpValidator
from PyQt5.QtCore import Qt, pyqtSlot, QTimer, pyqtSignal, QRegExp

from .useful_widgets import (LineEditReadOnly, global_gui_parameters, ElementSelection,
                             get_background_css, SecondaryWindow, set_tooltip, LineEditExtended)

from .form_base_widget import FormBaseWidget
from .dlg_find_elements import DialogFindElements
from .dlg_select_quant_standard import DialogSelectQuantStandard

import logging
logger = logging.getLogger(__name__)

_fitting_preset_names = {
    "None": "None",
    "fit_with_tail": "With Tail",
    "free_more": "Free",
    "e_calibration": "E-axis",
    "linear": "Area",
    "adjust_element1": "Custom 1",
    "adjust_element2": "Custom 2",
    "adjust_element3": "Custom 3",
}


class ModelWidget(FormBaseWidget):

    # Signal that is sent (to main window) to update global state of the program
    update_global_state = pyqtSignal()
    computations_complete = pyqtSignal()
    # Signal is emitted when a new model is loaded (or computed).
    # True - model loaded successfully, False - otherwise
    # In particular, the signal may be used to update the widgets that depend on incident energy,
    #   because it may change as the model is loaded.
    signal_model_loaded = pyqtSignal(bool)
    # Incident energy or selected range changed (plots need to be redrawn)
    signal_incident_energy_or_range_changed = pyqtSignal()

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
        # Reference to background thread used to run computations. The reference is
        #   meaningful only when the computations are run.
        self.bckg_thread = None

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
        self.pb_manage_emission_lines.clicked.connect(
            self.pb_manage_emission_lines_clicked)

    def _setup_settings_group(self):

        self.group_settings = QGroupBox("Settings for Fitting Algorithm")

        self.pb_general = QPushButton("General ...")
        self.pb_general.clicked.connect(self.pb_general_clicked)

        self.pb_elements = QPushButton("Elements ...")
        self.pb_elements.clicked.connect(self.pb_elements_clicked)

        self.pb_global_params = QPushButton("Global Parameters ...")
        self.pb_global_params.clicked.connect(self.pb_global_params_clicked)

        combo_items = list(_fitting_preset_names.values())
        self.cb_step1 = QComboBox()
        self.cb_step1.setMinimumWidth(150)
        self.cb_step1.addItems(combo_items)
        self.cb_step1.setCurrentIndex(1)  # Should also be set based on data
        self.cb_step2 = QComboBox()
        self.cb_step2.setMinimumWidth(150)
        self.cb_step2.addItems(combo_items)

        vbox = QVBoxLayout()
        hbox = QHBoxLayout()
        hbox.addWidget(self.pb_general)
        hbox.addWidget(self.pb_elements)
        hbox.addWidget(self.pb_global_params)
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

        self.lb_fitting_results = LineEditReadOnly(
            f"Iterations: {0}  Variables: {0}  R-squared: {0.000}")

        vbox = QVBoxLayout()

        hbox = QHBoxLayout()
        hbox.addWidget(self.pb_start_fitting)
        hbox.addWidget(self.pb_save_spectrum)
        vbox.addLayout(hbox)

        vbox.addWidget(self.lb_fitting_results)

        self.group_model_fitting.setLayout(vbox)

    def _set_tooltips(self):
        set_tooltip(self.pb_find_elines,
                    "Automatically find emission lines from <b>total spectrum</b>.")
        set_tooltip(
            self.pb_load_elines,
            "Load model parameters, including selected emission lines from <b>JSON</b> file, "
            "which was previously save using <b>Save Parameters to File ...</b>.")
        set_tooltip(
            self.pb_load_qstandard,
            "Load <b>quantitative standard</b>. The model is reset and the emission lines "
            "that fit within the selected range of energies are added to the list "
            "of emission lines.")
        set_tooltip(
            self.pb_save_elines,
            "Save the model parameters including the parameters of the selected emission lines "
            "to <b>JSON</b> file.")
        set_tooltip(
            self.le_param_fln,
            "The name of the recently loaded <b>parameter file</b> or serial number "
            "and name of the loaded <b>quantitative standard</b>")

        set_tooltip(
            self.pb_manage_emission_lines,
            "Open a user friendly interface that allows to <b>add and remove emission lines</b> "
            "to the list or <b>modify parameters</b> of the selected emission lines")

        set_tooltip(self.pb_general, "<b>General settings</b> for fitting algorithms.")
        set_tooltip(
            self.pb_elements,
            "Manually adjust fitting parameters for the <b>selected emission lines</b>, "
            "including preset fitting configurations")
        set_tooltip(self.pb_global_params,
                    "Manually adjust <b>global fitting parameters</b>, "
                    "including <b>preset fitting configurations</b>")
        set_tooltip(self.cb_step1, "Select preset fitting configuration for <b>Step 1</b>. "
                                   "Click <b>Elements...</b> and <b>Global Parameters...</b> "
                                   "buttons to open dialog boxes to configure the presets.")
        set_tooltip(self.cb_step2, "Select preset fitting configuration for <b>Step 2</b>. "
                                   "Click <b>Elements...</b> and <b>Global Parameters...</b> "
                                   "buttons to open dialog boxes to configure the presets.")

        set_tooltip(
            self.pb_start_fitting,
            "Click the button to <b>run fitting of total spectrum</b>. The result of fitting includes "
            "the refined set of emission line parameters. The fitted spectrum is displayed in "
            "<b>'Fitting Model'</b> tab and can be saved by clicking <b>'Save Spectrum/Fit ...'</b> button.")
        set_tooltip(
            self.pb_save_spectrum,
            "Save <b>raw and fitted total spectra</b>. Click <b>'Start Fitting'</b> to perform fitting "
            "before saving the spectrum")
        set_tooltip(self.lb_fitting_results,
                    "<b>Output parameters</b> produced by the fitting algorithm")

    def update_widget_state(self, condition=None):
        if condition == "tooltips":
            self._set_tooltips()

        state_file_loaded = self.gui_vars["gui_state"]["state_file_loaded"]
        state_model_exist = self.gui_vars["gui_state"]["state_model_exists"]
        state_model_fit_exists = self.gui_vars["gui_state"]["state_model_fit_exists"]

        self.group_model_params.setEnabled(state_file_loaded)
        self.pb_save_elines.setEnabled(state_file_loaded & state_model_exist)

        self.pb_manage_emission_lines.setEnabled(state_file_loaded & state_model_exist)

        self.group_settings.setEnabled(state_file_loaded & state_model_exist)

        self.group_settings.setEnabled(state_file_loaded & state_model_exist)

        self.group_model_fitting.setEnabled(state_file_loaded & state_model_exist)
        self.pb_save_spectrum.setEnabled(state_file_loaded & state_model_exist & state_model_fit_exists)

    def pb_find_elines_clicked(self):
        dialog_data = self.gpc.get_autofind_elements_params()
        dlg = DialogFindElements()
        dlg.set_dialog_data(dialog_data)
        ret = dlg.exec()
        if ret:
            dialog_data = dlg.get_dialog_data()
            logger.debug("Saving parameters from 'DialogFindElements'")
            if self.gpc.set_autofind_elements_params(dialog_data):
                self.signal_incident_energy_or_range_changed.emit()
            if dlg.find_elements_requested:
                logger.debug("Starting automated element search")

                def cb():
                    self.gpc.find_elements_automatically()
                    self.computations_complete.emit()

                self.computations_complete.connect(self.slot_find_elines_clicked)
                self.gui_vars["gui_state"]["running_computations"] = True
                self.update_global_state.emit()
                self.bckg_thread = Thread(target=cb)
                self.bckg_thread.start()

    def slot_find_elines_clicked(self):
        self.computations_complete.disconnect(self.slot_find_elines_clicked)
        self.gui_vars["gui_state"]["running_computations"] = False
        self.update_global_state.emit()

        self.gui_vars["gui_state"]["state_model_exists"] = True
        self.gui_vars["gui_state"]["state_model_fit_exists"] = False
        self.signal_model_loaded.emit(True)
        self.update_global_state.emit()
        logger.info("Automated element search is complete")

    def pb_load_elines_clicked(self):
        # TODO: Propagate current directory here and use it in the dialog call
        current_dir = self.gpc.get_current_working_directory()
        file_name = QFileDialog.getOpenFileName(self, "Select File with Model Parameters",
                                                current_dir,
                                                "JSON (*.json);; All (*)")
        file_name = file_name[0]
        if file_name:
            # TODO: emit signal ('parameters changed', i.e. incident energy and range changed)
            # TODO: start necessary processing
            try:
                def _ask_question(text):
                    def question():
                        mb = QMessageBox(QMessageBox.Question, "Question",
                                         text, QMessageBox.Yes | QMessageBox.No,
                                         parent=self)
                        if mb.exec() == QMessageBox.Yes:
                            return True
                        else:
                            return False
                    return question

                self.gpc.load_parameters_from_file(file_name, _ask_question)
            except IOError as ex:
                logger.error(f"Exception: {ex}")
                mb_error = QMessageBox(QMessageBox.Critical, "Error",
                                       f"{ex}", QMessageBox.Ok, parent=self)
                mb_error.exec()
                # It doesn't seem that the state of the program needs to be changed if
                #   the file was not loaded at all
            except Exception as ex:
                logger.error(f"Exception: error occurred while loading parameters: {ex}")
                mb_error = QMessageBox(QMessageBox.Critical, "Error",
                                       f"Error occurred while processing loaded parameters: {ex}",
                                       QMessageBox.Ok, parent=self)
                mb_error.exec()
                # Here the parameters were loaded and processing was partially performed,
                #   so change the state of the program
                self.gui_vars["gui_state"]["state_model_exists"] = False
                self.gui_vars["gui_state"]["state_model_fit_exists"] = False
                self.signal_model_loaded.emit(False)
                self.update_global_state.emit()

            else:
                self.gui_vars["gui_state"]["state_model_exists"] = True
                self.gui_vars["gui_state"]["state_model_fit_exists"] = False
                self.signal_model_loaded.emit(True)
                self.update_global_state.emit()

            print(f"Loading model parameters from file: {file_name}")

    def pb_load_qstandard_clicked(self):
        qe_param_built_in, qe_param_custom, qe_standard_selected = self.gpc.get_quant_standard_list()
        dlg = DialogSelectQuantStandard()
        dlg.set_standards(qe_param_built_in, qe_param_custom, qe_standard_selected)
        ret = dlg.exec()
        if ret:
            selected_standard = dlg.get_selected_standard()
            if selected_standard is not None:
                self.gpc.set_selected_quant_standard(selected_standard)

                msg = f"QS: '{selected_standard['name']}'"
                if self.gpc.is_quant_standard_custom(selected_standard):
                    msg += " (user-defined)"
                self.le_param_fln.setText(msg)

                self.gpc.process_peaks_from_quantitative_sample_data()

                self.gui_vars["gui_state"]["state_model_exists"] = True
                self.gui_vars["gui_state"]["state_model_fit_exists"] = False
                self.signal_model_loaded.emit(True)
                self.update_global_state.emit()

                standard_index = dlg.selected_standard_index
                print(f"Loading quantitative standard: {standard_index}")
            else:
                logger.error("No quantitative standard was selected.")
        else:
            print("Cancelled loading quantitative standard")

    def pb_save_elines_clicked(self):
        # TODO: Propagate full path to the saved file here
        fln = os.path.expanduser("~/model_parameters.json")
        file_name = QFileDialog.getSaveFileName(self, "Select File to Save Model Parameters",
                                                fln,
                                                "JSON (*.json);; All (*)")
        file_name = file_name[0]
        if file_name:
            print(f"Saving model parameters to file file: {file_name}")

    def pb_manage_emission_lines_clicked(self):
        # Position the window in relation ot the main window (only when called once)
        pos = self.ref_main_window.pos()
        self.ref_main_window.wnd_manage_emission_lines.position_once(pos.x(), pos.y())

        if not self.ref_main_window.wnd_manage_emission_lines.isVisible():
            self.ref_main_window.wnd_manage_emission_lines.show()
        self.ref_main_window.wnd_manage_emission_lines.activateWindow()

    def pb_general_clicked(self):
        dlg = DialogGeneralFittingSettings()
        ret = dlg.exec()
        if ret:
            print("Dialog closed. Changes accepted.")
        else:
            print("Cancelled.")

    def pb_elements_clicked(self):
        dlg = DialogElementSettings()
        ret = dlg.exec()
        if ret:
            print("'Elements' dialog closed. Changes accepted.")
        else:
            print("Cancelled.")

    def pb_global_params_clicked(self):
        dlg = DialogGlobalParamsSettings()
        ret = dlg.exec()
        if ret:
            print("'Global Parameters' dialog closed. Changes accepted.")
        else:
            print("Cancelled.")

    def pb_save_spectrum_clicked(self):
        # TODO: Propagate current directory here and use it in the dialog call
        current_dir = os.path.expanduser("~")
        dir = QFileDialog.getExistingDirectory(
            self, "Select Directory to Save Spectrum/Fit", current_dir,
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks)
        if dir:
            print(f"Spectrum/Fit is saved to directory {dir}")
        else:
            print("Spectrum/Fit saving is cancelled")

    def pb_start_fitting_clicked(self):

        self.gui_vars["gui_state"]["running_computations"] = True
        self.update_global_state.emit()

        if not self._timer:
            self._timer = QTimer()
        self._timer.timeout.connect(self.timerExpired)
        self._timer.setInterval(40)
        self._timer_counter = 0
        self._timer.start()

    @pyqtSlot()
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
            status_bar.showMessage("Total spectrum fitting is successfully completed. "
                                   "Results are presented in 'Fitting Model' tab.", 5000)
            self.gui_vars["gui_state"]["running_computations"] = False
            self.update_global_state.emit()


class WndManageEmissionLines(SecondaryWindow):

    signal_selected_element_changed = pyqtSignal(str)
    signal_update_element_selection_list = pyqtSignal()
    signal_update_add_remove_btn_state = pyqtSignal(bool, bool)
    signal_marker_state_changed = pyqtSignal(bool)

    def __init__(self,  *, gpc, gui_vars):
        super().__init__()

        # Global processing classes
        self.gpc = gpc
        # Global GUI variables (used for control of GUI state)
        self.gui_vars = gui_vars

        self._enable_events = False

        self._eline_list = []  # List of emission lines (used in the line selection combo)
        self._table_contents = []  # Keep a copy of table contents (list of dict)
        self._selected_eline = ""

        self.initialize()

        self._enable_events = True

        # Marker state is reported by Matplotlib plot in 'line_plot' model
        def cb(marker_state):
            self.signal_marker_state_changed.emit(marker_state)
        self.gpc.set_marker_reporter(cb)
        self.signal_marker_state_changed.connect(self.slot_marker_state_changed)

        # Update button states
        self._update_add_remove_btn_state()
        self._update_add_edit_userpeak_btn_state()
        self._update_add_edit_pileup_peak_btn_state()

    def initialize(self):
        self.setWindowTitle("PyXRF: Add/Remove Emission Lines")
        self.resize(600, 600)

        top_buttons = self._setup_select_elines()
        self._setup_elines_table()
        bottom_buttons = self._setup_action_buttons()

        vbox = QVBoxLayout()

        # Group of buttons above the table
        vbox.addLayout(top_buttons)

        # Tables
        hbox = QHBoxLayout()
        hbox.addWidget(self.tbl_elines)
        vbox.addLayout(hbox)

        vbox.addLayout(bottom_buttons)

        self.setLayout(vbox)

        self._set_tooltips()

    def _setup_select_elines(self):

        self.cb_select_all = QCheckBox("All")
        self.cb_select_all.setChecked(True)
        self.cb_select_all.toggled.connect(self.cb_select_all_toggled)

        self.element_selection = ElementSelection()

        # The following field should switched to 'editable' state from when needed
        self.le_peak_intensity = LineEditReadOnly()

        self.pb_add_eline = QPushButton("Add")
        self.pb_add_eline.clicked.connect(self.pb_add_eline_clicked)

        self.pb_remove_eline = QPushButton("Remove")
        self.pb_remove_eline.clicked.connect(self.pb_remove_eline_clicked)

        self.pb_user_peaks = QPushButton("New User Peak ...")
        self.pb_user_peaks.clicked.connect(self.pb_user_peaks_clicked)
        self.pb_pileup_peaks = QPushButton("New Pileup Peak ...")
        self.pb_pileup_peaks.clicked.connect(self.pb_pileup_peaks_clicked)

        self.element_selection.signal_current_item_changed.connect(
            self.element_selection_item_changed)

        vbox = QVBoxLayout()

        hbox = QHBoxLayout()
        hbox.addWidget(self.element_selection)
        hbox.addWidget(self.le_peak_intensity)
        hbox.addWidget(self.pb_add_eline)
        hbox.addWidget(self.pb_remove_eline)
        vbox.addLayout(hbox)

        hbox = QHBoxLayout()
        hbox.addWidget(self.cb_select_all)
        hbox.addStretch(1)
        hbox.addWidget(self.pb_user_peaks)
        hbox.addWidget(self.pb_pileup_peaks)
        vbox.addLayout(hbox)

        # Wrap vbox into hbox, because it will be inserted into vbox
        hbox = QHBoxLayout()
        hbox.addLayout(vbox)
        hbox.addStretch(1)
        return hbox

    def _setup_elines_table(self):
        """The table has only functionality necessary to demonstrate how it is going
        to look. A lot more code is needed to actually make it run."""

        self.tbl_elines = QTableWidget()

        self.tbl_labels = ["Z", "Line", "E, keV", "Peak Int.", "Rel. Int.(%)", "CS"]
        self.tbl_cols_editable = ["Peak Int."]
        self.tbl_value_min = {"Rel. Int.(%)": 0.1}
        tbl_cols_resize_to_content = ["Z", "Line"]

        self.tbl_elines.setColumnCount(len(self.tbl_labels))
        self.tbl_elines.verticalHeader().hide()
        self.tbl_elines.setHorizontalHeaderLabels(self.tbl_labels)

        self.tbl_elines.setSelectionBehavior(QTableWidget.SelectRows)
        self.tbl_elines.setSelectionMode(QTableWidget.SingleSelection)
        self.tbl_elines.itemSelectionChanged.connect(self.tbl_elines_item_selection_changed)
        self.tbl_elines.itemChanged.connect(self.tbl_elines_item_changed)

        header = self.tbl_elines.horizontalHeader()
        for n, lbl in enumerate(self.tbl_labels):
            # Set stretching for the columns
            if lbl in tbl_cols_resize_to_content:
                header.setSectionResizeMode(n, QHeaderView.ResizeToContents)
            else:
                header.setSectionResizeMode(n, QHeaderView.Stretch)
            # Set alignment for the columns headers (HEADERS only)
            if n == 0:
                header.setDefaultAlignment(Qt.AlignCenter)
            else:
                header.setDefaultAlignment(Qt.AlignRight)

    def _setup_action_buttons(self):

        self.pb_update = QPushButton("Update")
        self.pb_undo = QPushButton("Undo")

        self.pb_remove_rel = QPushButton("Remove Rel.Int.(%) <")
        self.le_remove_rel = QLineEdit("1.0")
        self.pb_remove_unchecked = QPushButton("Remove Unchecked Lines")

        hbox = QHBoxLayout()
        vbox = QVBoxLayout()
        vbox.addWidget(self.pb_undo)
        vbox.addWidget(self.pb_update)
        hbox.addLayout(vbox)
        hbox.addSpacing(20)
        vbox = QVBoxLayout()
        hbox2 = QHBoxLayout()
        hbox2.addWidget(self.pb_remove_rel)
        hbox2.addWidget(self.le_remove_rel)
        vbox.addLayout(hbox2)
        vbox.addWidget(self.pb_remove_unchecked)
        hbox.addLayout(vbox)
        hbox.addStretch(1)
        return hbox

    def _set_tooltips(self):
        set_tooltip(self.cb_select_all,
                    "<b>Select/Deselect All</b> emission lines in the list")
        set_tooltip(self.element_selection,
                    "<b>Set active</b> emission line")
        set_tooltip(self.le_peak_intensity,
                    "Set or modify <b>intensity</b> of the active peak.")
        set_tooltip(self.pb_add_eline,
                    "<b>Add</b> emission line to the list.")
        set_tooltip(self.pb_remove_eline,
                    "<b>Remove</b> emission line from the list.")
        set_tooltip(self.pb_user_peaks,
                    "Open dialog box to add or modify parameters of the <b>user-defined peak</b>")
        set_tooltip(self.pb_pileup_peaks,
                    "Open dialog box to add or modify parameters of the <b>pileup peak</b>")

        set_tooltip(self.tbl_elines,
                    "The list of the selected <b>emission lines</b>")

        set_tooltip(self.pb_update,
                    "Update the internally stored list of selected emission lines "
                    "and their parameters. This button is <b>deprecated</b>, but still may be "
                    "needed in some situations. In future releases it will be <b>removed</b> or replaced "
                    "with 'Accept' button. Substantial changes to the computational code is needed before "
                    "it happens.")
        set_tooltip(self.pb_undo,
                    "<b>Undo</b> changes to the table of selected emission lines. Doesn't always work.")
        set_tooltip(self.pb_remove_rel,
                    "<b>Remove emission lines</b> from the list if their relative intensity is less "
                    "then specified threshold.")
        set_tooltip(self.le_remove_rel,
                    "<b>Threshold</b> that controls which emission lines are removed "
                    "when <b>Remove Rel.Int.(%)</b> button is pressed.")
        set_tooltip(self.pb_remove_unchecked,
                    "Remove <b>unchecked</b> emission lines from the list.")

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

    def fill_eline_table(self, table_contents):
        self._table_contents = copy.deepcopy(table_contents)

        self._enable_events = False

        self.tbl_elines.setRowCount(len(table_contents))
        for nr, row in enumerate(table_contents):
            sel_status = row["sel_status"]
            row_data = [row["z"], row["eline"], row["energy"],
                        row["peak_int"], row["rel_int"], row["cs"]]

            for nc, entry in enumerate(row_data):
                label = self.tbl_labels[nc]

                s = None
                # The case when the value (Rel. Int.) is limited from the bottom
                #   We don't want to print very small numbers here
                if label in self.tbl_value_min:
                    v = self.tbl_value_min[label]
                    if isinstance(entry, (float, np.float64)) and (entry < v):
                        s = f"<{v:.2f}"
                if s is None:
                    if isinstance(entry, (float, np.float64)):
                        s = f"{entry:.2f}" if entry else "-"
                    else:
                        s = f"{entry}" if entry else "-"

                item = QTableWidgetItem(s)

                # Add check box to the first element of each row
                if nc == 0:
                    item.setFlags(item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
                    item.setCheckState(Qt.Checked if sel_status else Qt.Unchecked)
                    item.setTextAlignment(Qt.AlignCenter | Qt.AlignVCenter)
                else:
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)

                # Set all columns not editable (unless needed)
                if label not in self.tbl_cols_editable:
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)

                # Set alternating background colors for the table rows
                #   Make background for editable items a little brighter
                brightness = 240 if label in self.tbl_cols_editable else 220
                if nr % 2:
                    brush = QBrush(QColor(255, brightness, brightness))  # Light-blue
                else:
                    brush = QBrush(QColor(brightness, 255, brightness))  # Light-green
                item.setBackground(brush)

                self.tbl_elines.setItem(nr, nc, item)

        self._enable_events = True
        # Update the rest of the widgets
        self._update_widgets_based_on_table_state()

    @pyqtSlot()
    def update_widget_data(self):
        # This is typically a new set of emission lines. Clear the selection both
        #   in the table and in the element selection tool.
        self.element_selection.set_current_item("")
        self.tbl_elines.clearSelection()
        self._set_selected_eline("")
        # Now update the tables
        self._update_eline_selection_list()
        self._update_eline_table()
        self._update_add_remove_btn_state()

    def pb_pileup_peaks_clicked(self):
        data = {}

        eline = self._selected_eline
        if self.gpc.get_eline_name_category(eline) == "pileup":
            logger.error(f"Attempt to add pileup peak while another pileup peak is selected.")
            return

        energy, marker_visible = self.gpc.get_suggested_manual_peak_energy()
        best_guess = self.gpc.get_guessed_pileup_peak_components(energy=energy,
                                                                 tolerance=0.1)
        if best_guess is not None:
            el1, el2, energy = best_guess
        else:
            # No peaks were found, enter peaks manually
            el1, el2, energy = "", "", 0
        data["element1"] = el1
        data["element2"] = el2
        data["energy"] = energy
        data["range_low"], data["range_high"] = self.gpc.get_selected_energy_range()

        if not marker_visible:
            # We shouldn't end up here, but this will protect from crashing in case
            #   the button was not disabled (a bug).
            msg = "Select location of the new peak center (energy)\n" \
                  "by clicking on the plot in 'Fit Model' tab"
            msgbox = QMessageBox(QMessageBox.Information, "User Input Required",
                                 msg, QMessageBox.Ok, parent=self)
            msgbox.exec()
        else:
            dlg = DialogPileupPeakParameters()

            def func():
                def f(e1, e2):
                    try:
                        name = self.gpc.generate_pileup_peak_name(e1, e2)
                        e = self.gpc.get_pileup_peak_energy(name)
                    except Exception:
                        e = 0
                    return e
                return f

            dlg.set_compute_energy_function(func())
            dlg.set_parameters(data)
            if dlg.exec():
                print("Pileup peak is added")

    def pb_user_peaks_clicked(self):
        eline = self._selected_eline
        # If current peak is user_defined peak
        is_userpeak = self.gpc.get_eline_name_category(eline) == "userpeak"

        if is_userpeak:
            data = {}
            data["enabled"] = True
            data["name"] = eline
            data["maxv"] = self.gpc.get_eline_intensity(eline)
            data["energy"], data["fwhm"] = self.gpc.get_current_userpeak_energy_fwhm()

            dlg = DialogEditUserPeakParameters()
            dlg.set_parameters(data=data)
            if dlg.exec():

                print("Editing of user defined peak is completed")

        else:
            data = {}
            data["name"] = self.gpc.get_unused_userpeak_name()
            data["energy"], marker_visible = self.gpc.get_suggested_manual_peak_energy()
            if marker_visible:
                dlg = DialogNewUserPeak()
                dlg.set_parameters(data=data)
                if dlg.exec():
                    try:
                        eline = data["name"]
                        self.gpc.add_peak_manual(eline)
                        self._update_eline_table()  # Update the table
                        self.tbl_elines_set_selection(eline)  # Select new emission line
                        self._set_selected_eline(eline)
                        logger.info(f"New user defined peak {eline} is added")
                    except RuntimeError as ex:
                        msg = str(ex)
                        msgbox = QMessageBox(QMessageBox.Critical, "Error",
                                             msg, QMessageBox.Ok, parent=self)
                        msgbox.exec()
                        # Reload the table anyway (nothing is going to be selected)
                        self._update_eline_table()
            else:
                msg = "Select location of the new peak center (energy)\n" \
                      "by clicking on the plot in 'Fit Model' tab"
                msgbox = QMessageBox(QMessageBox.Information, "User Input Required",
                                     msg, QMessageBox.Ok, parent=self)
                msgbox.exec()

    @pyqtSlot()
    def pb_add_eline_clicked(self):
        logger.debug("'Add line' clicked")
        # It is assumed that this button is active only if an element is selected from the list
        #   of available emission lines. It can't be used to add user-defined peaks or pileup peaks.
        eline = self._selected_eline
        if eline:
            try:
                self.gpc.add_peak_manual(eline)
                self._update_eline_table()  # Update the table
                self.tbl_elines_set_selection(eline)  # Select new emission line
            except RuntimeError as ex:
                msg = str(ex)
                msgbox = QMessageBox(QMessageBox.Critical, "Error",
                                     msg, QMessageBox.Ok, parent=self)
                msgbox.exec()
                # Reload the table anyway (nothing is going to be selected)
                self._update_eline_table()

    @pyqtSlot()
    def pb_remove_eline_clicked(self):
        logger.debug("'Remove line' clicked")
        eline = self._selected_eline
        if eline:
            # If currently selected line is the emission line (like Ca_K), we want
            # it to remain selected after it is deleted. This means that nothing is selected
            # in the table. For other lines, nothing should remain selected.
            self.tbl_elines.clearSelection()
            self.gpc.remove_peak_manual(eline)
            self._update_eline_table()  # Update the table
            if self.gpc.get_eline_name_category(eline) != "eline":
                eline = ""
            # This will update widgets
            self._set_selected_eline(eline)

    def cb_select_all_toggled(self, state):
        self._enable_events = False

        eline_list, state_list = [], []

        for n_row in range(self.tbl_elines.rowCount()):
            eline = self._table_contents[n_row]["eline"]
            # Do not deselect lines in category 'other'. They probably never to be deleted.
            # They also could be deselected manually.
            if self.gpc.get_eline_name_category(eline) == "other" and not state:
                to_check = True
            else:
                to_check = state
            self.tbl_elines.item(n_row, 0).setCheckState(Qt.Checked if to_check else Qt.Unchecked)
            eline_list.append(eline)
            state_list.append(to_check)

        self.gpc.set_checked_emission_lines(eline_list, state_list)
        self._enable_events = True

    def tbl_elines_item_changed(self, item):
        if self._enable_events:
            n_row, n_col = self.tbl_elines.row(item), self.tbl_elines.column(item)
            # Checkbox was clicked
            if n_col == 0:
                state = bool(item.checkState())
                eline = self._table_contents[n_row]["eline"]
                self.gpc.set_checked_emission_lines([eline], [state])

    def tbl_elines_item_selection_changed(self):
        sel_ranges = self.tbl_elines.selectedRanges()
        # The table is configured to have one or no selected ranges
        # 'Open' button should be enabled only if a range (row) is selected
        if sel_ranges:
            index = sel_ranges[0].topRow()
            eline = self._table_contents[index]["eline"]
            if self._enable_events:
                self._enable_events = False
                self._set_selected_eline(eline)
                self.element_selection.set_current_item(eline)
                self._enable_events = True

    def tbl_elines_set_selection(self, eline):
        """
        Select the row with emission line `eline` in the table. Deselect everything if
        the emission line does not exist.
        """
        index = self._get_eline_index_in_table(eline)
        self.tbl_elines.clearSelection()
        if index >= 0:
            self.tbl_elines.selectRow(index)

    def element_selection_item_changed(self, index, eline):
        self.signal_selected_element_changed.emit(eline)

        if self._enable_events:
            self._enable_events = False
            self._set_selected_eline(eline)
            self.tbl_elines_set_selection(eline)
            self._enable_events = True

    def _display_peak_intensity(self, eline):
        v = self.gpc.get_eline_intensity(eline)
        s = f"{v:.10g}" if v is not None else ""
        self.le_peak_intensity.setText(s)

    @pyqtSlot(str)
    def slot_selection_item_changed(self, eline):
        self.element_selection.set_current_item(eline)

    @pyqtSlot(bool)
    def slot_marker_state_changed(self, state):
        # If userpeak is selected and plot is clicked (marker is set), then user
        #   should be allowed to add userpeak at a new location. So deselect the userpeak
        #   from the table (if it is selected)
        logger.debug(f"Vertical marker on the fit plot changed state to {state}.")
        if state:
            self._deselect_special_peak_in_table()
        # Now update state of all buttons
        self._update_add_remove_btn_state()
        self._update_add_edit_userpeak_btn_state()
        self._update_add_edit_pileup_peak_btn_state()

    def _deselect_special_peak_in_table(self):
        """Deselect userpeak if a userpeak is selected"""
        if self.gpc.get_eline_name_category(self._selected_eline) in ("userpeak", "pileup"):
            # Clear all selections
            self.tbl_elines_set_selection("")
            self._set_selected_eline("")
            # We also want to show marker at the new position
            self.gpc.show_marker_at_current_position()

    def _update_widgets_based_on_table_state(self):
        index, eline = self._get_current_index_in_table()
        if index >= 0:
            # Selection exists. Update the state of element selection widget.
            self.element_selection.set_current_item(eline)
        else:
            # No selection, update the state based on element selection widget.
            index, eline = self._get_current_index_in_table()
            self.tbl_elines_set_selection(eline)
        self._update_add_remove_btn_state(eline)
        self._update_add_edit_userpeak_btn_state()
        self._update_add_edit_pileup_peak_btn_state()

    def _update_eline_selection_list(self):
        self._eline_list = self.gpc.get_full_eline_list()
        self.element_selection.set_item_list(self._eline_list)
        self.signal_update_element_selection_list.emit()

    def _update_eline_table(self):
        eline_table = self.gpc.get_selected_eline_table()
        self.fill_eline_table(eline_table)

    def _get_eline_index_in_table(self, eline):
        try:
            index = [_["eline"] for _ in self._table_contents].index(eline)
        except ValueError:
            index = -1
        return index

    def _get_eline_index_in_list(self, eline):
        try:
            index = self._eline_list.index(eline)
        except ValueError:
            index = -1
        return index

    def _get_current_index_in_table(self):
        sel_ranges = self.tbl_elines.selectedRanges()
        # The table is configured to have one or no selected ranges
        # 'Open' button should be enabled only if a range (row) is selected
        if sel_ranges:
            index = sel_ranges[0].topRow()
            eline = self._table_contents[index]["eline"]
        else:
            index, eline = -1, ""
        return index, eline

    def _get_current_index_in_list(self):
        index, eline = self.element_selection.get_current_item()
        return index, eline

    def _update_add_remove_btn_state(self, eline=None):
        if eline is None:
            index_in_table, eline = self._get_current_index_in_table()
            index_in_list, eline = self._get_current_index_in_list()
        else:
            index_in_table = self._get_eline_index_in_table(eline)
            index_in_list = self._get_eline_index_in_list(eline)
        add_enabled, remove_enabled = True, True
        if index_in_list < 0 and index_in_table < 0:
            add_enabled, remove_enabled = False, False
        else:
            if index_in_table >= 0:
                if self.gpc.get_eline_name_category(eline) != "other":
                    add_enabled = False
                else:
                    add_enabled, remove_enabled = False, False
            else:
                remove_enabled = False
        self.pb_add_eline.setEnabled(add_enabled)
        self.pb_remove_eline.setEnabled(remove_enabled)
        self.signal_update_add_remove_btn_state.emit(add_enabled, remove_enabled)

    def _update_add_edit_userpeak_btn_state(self):

        enabled = True
        if self.gpc.get_eline_name_category(self._selected_eline) == "userpeak":
            enabled = False

        # Finally check if marker is set (you need it for adding peaks)
        _, marker_set = self.gpc.get_suggested_manual_peak_energy()
        if not marker_set:
            enabled = False

        self.pb_user_peaks.setEnabled(enabled)

    def _update_add_edit_pileup_peak_btn_state(self):

        enabled = True
        add_peak = True
        if self.gpc.get_eline_name_category(self._selected_eline) == "pileup":
            add_peak = False

        # Finally check if marker is set (you need it for adding peaks)
        _, marker_set = self.gpc.get_suggested_manual_peak_energy()
        # Ignore set marker for userpeaks (marker is used to display location of userpeaks)
        if self.gpc.get_eline_name_category(self._selected_eline) == "userpeak":
            marker_set = False

        if not marker_set and add_peak:
            enabled = False

        if add_peak:
            btn_text = "New Pileup Peak ..."
        else:
            btn_text = "Edit Pileup Peak ..."
        self.pb_pileup_peaks.setText(btn_text)
        self.pb_pileup_peaks.setEnabled(enabled)

    def _set_selected_eline(self, eline):
        self._update_add_remove_btn_state(eline)
        if self.gpc.get_eline_name_category(eline) == "userpeak":
            self.pb_user_peaks.setText("Edit User Peak ...")
        else:
            self.pb_user_peaks.setText("New User Peak ...")
        if eline != self._selected_eline:
            self._selected_eline = eline
            self.gpc.set_selected_eline(eline)
            self._display_peak_intensity(eline)
        else:
            # Peak intensity may change in some circumstances, so renew the displayed value.
            self._display_peak_intensity(eline)
        # Update button states after 'self._selected_eline' is set
        self._update_add_edit_userpeak_btn_state()
        self._update_add_edit_pileup_peak_btn_state()


class DialogGeneralFittingSettings(QDialog):

    def __init__(self, parent=None):

        super().__init__(parent)

        self.setWindowTitle("General Settings for Fitting Algorithm")

        vbox_left = QVBoxLayout()

        # ===== Top-left section of the dialog box =====
        self.le_max_iterations = QLineEdit()
        set_tooltip(self.le_max_iterations,
                    "<b>Maximum number of iterations</b> used for total spectrum fitting.")
        self.le_tolerance_stopping = QLineEdit()
        set_tooltip(self.le_tolerance_stopping,
                    "<b>Tolerance</b> setting for total spectrum fitting.")
        self.le_escape_ratio = QLineEdit()
        set_tooltip(self.le_escape_ratio,
                    "Parameter for total spectrum fitting: <b>escape ration</b>")
        grid = QGridLayout()
        grid.addWidget(QLabel("Iterations (max):"), 0, 0)
        grid.addWidget(self.le_max_iterations, 0, 1)
        grid.addWidget(QLabel("Tolerance (stopping):"), 1, 0)
        grid.addWidget(self.le_tolerance_stopping, 1, 1)
        grid.addWidget(QLabel("Escape peak ratio:"), 2, 0)
        grid.addWidget(self.le_escape_ratio, 2, 1)
        self.group_total_spectrum_fitting = QGroupBox(
            "Fitting of Total Spectrum (Model)")
        self.group_total_spectrum_fitting.setLayout(grid)
        vbox_left.addWidget(self.group_total_spectrum_fitting)

        # ===== Bottom-left section of the dialog box =====

        # Incident energy and the selected range
        self.le_incident_energy = QLineEdit()
        set_tooltip(self.le_incident_energy,
                    "<b>Incident energy</b> in keV")
        self.le_range_low = QLineEdit()
        set_tooltip(self.le_range_low,
                    "<b>Lower boundary</b> of the selected range in keV.")
        self.le_range_high = QLineEdit()
        set_tooltip(self.le_range_high,
                    "<b>Upper boundary</b> of the selected range in keV.")
        self.group_energy_range = QGroupBox("Incident Energy and Selected Range")
        grid = QGridLayout()
        grid.addWidget(QLabel("Incident energy, keV"), 0, 0)
        grid.addWidget(self.le_incident_energy, 0, 1)
        grid.addWidget(QLabel("Range (low), keV"), 1, 0)
        grid.addWidget(self.le_range_low, 1, 1)
        grid.addWidget(QLabel("Range (high), keV"), 2, 0)
        grid.addWidget(self.le_range_high, 2, 1)
        self.group_energy_range.setLayout(grid)
        vbox_left.addWidget(self.group_energy_range)

        vbox_right = QVBoxLayout()

        # ===== Top-right section of the dialog box =====

        self.cb_linear_baseline = QCheckBox("Subtract linear baseline")
        set_tooltip(self.cb_linear_baseline,
                    "Subtract baseline as represented as a constant. <b>XRF Map generation</b>. "
                    "Baseline subtraction is performed as part of NNLS fitting.")
        self.cb_snip_baseline = QCheckBox("Subtract baseline using SNIP")
        set_tooltip(self.cb_snip_baseline,
                    "Subtract baseline using SNIP method. <b>XRF Map generation</b>. "
                    "This is a separate step of processing and can be used together with "
                    "'linear' baseline subtraction if needed.")

        # This option is not supported. In the future it may be removed
        #   if not needed or implemented.
        self.cb_add_const_to_data = QCheckBox("Add const. bias to data")
        self.cb_add_const_to_data.setEnabled(False)
        self.lb_add_const_to_data = QLabel("Constant bias:")
        self.lb_add_const_to_data.setEnabled(False)
        self.le_add_const_to_data = QLineEdit()
        self.le_add_const_to_data.setEnabled(False)

        vbox = QVBoxLayout()
        vbox.addWidget(self.cb_linear_baseline)
        vbox.addWidget(self.cb_snip_baseline)
        vbox.addWidget(self.cb_add_const_to_data)
        hbox = QHBoxLayout()
        hbox.addWidget(self.lb_add_const_to_data)
        hbox.addWidget(self.le_add_const_to_data)
        vbox.addLayout(hbox)

        self.group_pixel_fitting = QGroupBox("Fitting of Single Spectra (XRF Maps)")
        self.group_pixel_fitting.setLayout(vbox)

        vbox_right.addWidget(self.group_pixel_fitting)

        # ===== Bottom-right section of the dialog box =====

        self.le_snip_window_size = QLineEdit()
        set_tooltip(self.le_snip_window_size,
                    "Window size for <b>SNIP</b> algorithm. Used both for total spectrum fitting "
                    "and XRF Map generation.")

        vbox = QVBoxLayout()
        hbox = QHBoxLayout()
        hbox.addWidget(QLabel("SNIP window size(*):"))
        hbox.addWidget(self.le_snip_window_size)
        vbox.addLayout(hbox)
        vbox.addWidget(QLabel("*Total spectrum fitting always includes \n"
                              "    SNIP baseline subtraction"))

        self.group_all_fitting = QGroupBox("All Fitting")
        self.group_all_fitting.setLayout(vbox)

        vbox_right.addWidget(self.group_all_fitting)
        vbox_right.addStretch(1)

        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.button(QDialogButtonBox.Cancel).setDefault(True)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        hbox = QHBoxLayout()
        hbox.addLayout(vbox_left)
        hbox.addLayout(vbox_right)

        vbox = QVBoxLayout()
        vbox.addLayout(hbox)
        vbox.addWidget(button_box)

        self.setLayout(vbox)


class _FittingSettings():

    def __init__(self, energy_column=True):

        labels_presets = [_ for _ in _fitting_preset_names.values() if _ != "None"]

        # Labels for horizontal header
        self.tbl_labels = ["Name", "E, keV", "Value", "Min", "Max"] + labels_presets

        # Labels for editable columns
        self.tbl_cols_editable = ("Value", "Min", "Max")

        # Labels for the columns that contain combo boxes
        self.tbl_cols_combobox = labels_presets

        # The list of columns with fixed size
        self.tbl_cols_stretch = ("Value", "Min", "Max")

        # Table item representation if different from default
        self.tbl_format = {"E, keV": ".4f", "Value": ".8g", "Min": ".8g", "Max": ".8g"}

        # Checkbox items. All table items that are checkboxes are identical
        self.cbox_settings_items = ("none", "fixed", "lohi", "lo", "hi")

        if not energy_column:
            self.tbl_labels.pop(1)

    def setup_table(self):

        table = QTableWidget()
        table.setColumnCount(len(self.tbl_labels))
        table.verticalHeader().hide()
        table.setHorizontalHeaderLabels(self.tbl_labels)

        header = table.horizontalHeader()
        for n, lbl in enumerate(self.tbl_labels):
            # Set stretching for the columns
            if lbl in self.tbl_cols_stretch:
                header.setSectionResizeMode(n, QHeaderView.Stretch)
            else:
                header.setSectionResizeMode(n, QHeaderView.ResizeToContents)

        return table

    def fill_table(self, table, table_contents):

        table.setRowCount(len(table_contents))
        for nr, row in enumerate(table_contents):
            for nc, entry in enumerate(row):
                label = self.tbl_labels[nc]

                # Set alternating background colors for the table rows
                #   Make background for editable items a little brighter
                brightness = 240 if label in self.tbl_cols_editable else 220
                if nr % 2:
                    rgb_bckg = (255, brightness, brightness)
                else:
                    rgb_bckg = (brightness, 255, brightness)

                if self.tbl_labels[nc] not in self.tbl_cols_combobox:
                    if self.tbl_labels[nc] in self.tbl_format:
                        fmt = self.tbl_format[self.tbl_labels[nc]]
                        s = ("{:" + fmt + "}").format(entry)
                    else:
                        s = f"{entry}"

                    item = QTableWidgetItem(s)
                    if nc > 0:
                        item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)

                    # Set all columns not editable (unless needed)
                    if label not in self.tbl_cols_editable:
                        item.setFlags(item.flags() & ~Qt.ItemIsEditable)

                    # Make all items not selectable (we are not using selections)
                    item.setFlags(item.flags() & ~Qt.ItemIsSelectable)

                    # Note, that there is no way to set style sheet for QTableWidgetItem
                    item.setBackground(QBrush(QColor(*rgb_bckg)))

                    table.setItem(nr, nc, item)

                else:
                    item = QComboBox()

                    css1 = get_background_css(rgb_bckg, widget="QComboBox", editable=False)
                    css2 = get_background_css(rgb_bckg, widget="QWidget", editable=False)
                    item.setStyleSheet(css2 + css1)

                    item.addItems(self.cbox_settings_items)
                    if item.findText(entry) < 0:
                        print(f"Text '{entry}' is not found. The ComboBox is not set properly.")
                    item.setCurrentText(entry)  # Try selecting the item anyway
                    table.setCellWidget(nr, nc, item)


class DialogElementSettings(QDialog):

    def __init__(self, parent=None):

        super().__init__(parent)

        self.setWindowTitle("Fitting Parameters for Individual Emission Lines")
        self.setMinimumWidth(1100)
        self.setMinimumHeight(500)
        self.resize(1100, 500)

        hbox_el_select = self._setup_element_selection()
        self._setup_table()

        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.button(QDialogButtonBox.Cancel).setDefault(True)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        vbox = QVBoxLayout()
        vbox.addLayout(hbox_el_select)
        vbox.addWidget(self.table)
        vbox.addWidget(button_box)

        self.setLayout(vbox)

    def _setup_element_selection(self):

        self.cbox_element_sel = QComboBox()
        set_tooltip(self.cbox_element_sel,
                    "Select K, L or M <b>emission line</b> to edit the optimization parameters "
                    "used for the line during total spectrum fitting.")
        cb_sample_items = ("Ca_K", "Ti_K", "Fe_K")
        self.cbox_element_sel.addItems(cb_sample_items)

        hbox = QHBoxLayout()
        hbox.addWidget(QLabel("Select element:"))
        hbox.addWidget(self.cbox_element_sel)
        hbox.addStretch(1)

        return hbox

    def _setup_table(self):

        self.table_settings = _FittingSettings(energy_column=True)
        self.table = self.table_settings.setup_table()
        set_tooltip(self.table,
                    "Edit optimization parameters for the selected emission line. "
                    "Processing presets may be configured by specifying optimization strategy "
                    "for each parameter may be selected. A preset for each fitting step "
                    "of the total spectrum fitting may be selected in <b>Model</b> tab.")

        sample_contents = [
            ["Ca_ka1_area", 3.6917, 11799.14, 0, 10000000.0,
             "none", "none", "none", "none", "none", "fixed", "fixed"],
            ["Ca_ka1_delta_center", 3.6917, 0.0, -0.005, 0.005,
             "fixed", "fixed", "fixed", "fixed", "fixed", "fixed", "fixed"],
            ["Ca_ka1_delta_sigma", 3.6917, 0.0, -0.02, 0.02,
             "fixed", "fixed", "fixed", "fixed", "fixed", "fixed", "fixed"],
            ["Ca_ka1_ratio_adjust", 3.6917, 0.0, 0.1, 5.0,
             "fixed", "fixed", "fixed", "fixed", "fixed", "fixed", "fixed"],
        ]
        self.fill_table(sample_contents)

    def fill_table(self, table_contents):

        self.table_settings.fill_table(self.table, table_contents)


class DialogGlobalParamsSettings(QDialog):

    def __init__(self, parent=None):

        super().__init__(parent)

        self.setWindowTitle("Global Parameters for Fitting of Emission Lines")
        self.setMinimumWidth(1000)
        self.setMinimumHeight(500)
        self.resize(1000, 500)

        self._setup_table()

        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.button(QDialogButtonBox.Cancel).setDefault(True)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        vbox = QVBoxLayout()
        vbox.addWidget(self.table)
        vbox.addWidget(button_box)

        self.setLayout(vbox)

    def _setup_table(self):

        self.table_settings = _FittingSettings(energy_column=False)
        self.table = self.table_settings.setup_table()
        set_tooltip(self.table,
                    "Edit global optimization parameters. "
                    "Processing presets may be configured by specifying optimization strategy "
                    "for each parameter may be selected. A preset for each fitting step "
                    "of the total spectrum fitting may be selected in <b>Model</b> tab.")

        sample_contents = [
            ["coherent_sct_amplitude", 1866.280064, 1.0, 10000000000.0,
             "none", "none", "none", "none", "fixed", "fixed", "fixed"],
            ["coherent_sct_energy", 12.0, 9.0, 13.0,
             "fixed", "lohi", "fixed", "fixed", "fixed", "fixed", "fixed"],
            ["compton_amplitude", 1105.970249, 0.0, 10000000000.0,
             "none", "none", "none", "none", "fixed", "fixed", "fixed"],
            ["compton_angle", 70.0000000025, 70.0, 105.0,
             "lohi", "lohi", "fixed", "fixed", "fixed", "fixed", "fixed"],
        ]
        self.fill_table(sample_contents)

    def fill_table(self, table_contents):

        self.table_settings.fill_table(self.table, table_contents)


class DialogPileupPeakParameters(QDialog):

    def __init__(self, parent=None):

        super().__init__(parent)

        self._data = {"element1": "", "element2": "", "energy": 0}
        # Reference to function that computes pileup energy based on two emission lines
        self._compute_energy = None

        self.setWindowTitle("Pileup Peak Parameters")

        self.le_element1 = LineEditExtended()
        set_tooltip(self.le_element1,
                    "The <b>name</b> of the emission line #1")
        self.le_element2 = LineEditExtended()
        set_tooltip(self.le_element2,
                    "The <b>name</b> of the emission line #2")
        self.peak_energy = LineEditReadOnly()
        set_tooltip(self.peak_energy,
                    "The <b>energy</b> (location) of the pileup peak center. The energy can not"
                    "be edited: it is set based on the selected emission lines")

        self._validator_eline = QRegExpValidator()
        # TODO: the following regex is too broad: [a-z] should be narrowed down
        self._validator_eline.setRegExp(QRegExp(r"^[A-Z][a-z]?_[KLM][a-z]\d$"))

        instructions = QLabel("Specify two emission lines, e.g. Si_Ka1 and Fe_Ka1")

        grid = QGridLayout()
        grid.addWidget(QLabel("Emission line 1:"), 0, 0)
        grid.addWidget(self.le_element1, 0, 1)
        grid.addWidget(QLabel("Emission line 2:"), 1, 0)
        grid.addWidget(self.le_element2, 1, 1)
        grid.addWidget(QLabel("Peak energy, keV:"), 2, 0)
        grid.addWidget(self.peak_energy, 2, 1)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.button(QDialogButtonBox.Cancel).setDefault(True)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        self.pb_ok = button_box.button(QDialogButtonBox.Ok)

        vbox = QVBoxLayout()
        vbox.addWidget(instructions)
        vbox.addSpacing(10)
        vbox.addLayout(grid)
        vbox.addWidget(button_box)

        self.setLayout(vbox)

        self.le_element1.editingFinished.connect(self.le_element1_editing_finished)
        self.le_element2.editingFinished.connect(self.le_element2_editing_finished)

        self.le_element1.textChanged.connect(self.le_element1_text_changed)
        self.le_element2.textChanged.connect(self.le_element2_text_changed)

    def set_compute_energy_function(self, func):
        self._compute_energy = func

    def set_parameters(self, data):
        self._data = data.copy()
        self._show_data()

    def _format_float(self, v):
        return f"{v:.12g}"

    def _show_data(self):
        self.le_element1.setText(self._data["element1"])
        self.le_element2.setText(self._data["element2"])
        self._show_energy()

    def _show_energy(self, energy=None):
        if energy is None:
            energy = self._data["energy"]
        text = self._format_float(energy)
        self.peak_energy.setText(text)
        self._validate_all()

    def _validate_le_element1(self, text=None):
        return self._validate_eline(self.le_element1, text)

    def _validate_le_element2(self, text=None):
        return self._validate_eline(self.le_element2, text)

    def _validate_peak_energy(self, energy=None):
        # Peak energy is not edited, so it is always has a valid floating point number
        #   The only problem is that it may be out of range.
        if energy is None:
            energy = self._data["energy"]
        valid = self._data["range_low"] < energy < self._data["range_high"]
        self.peak_energy.setValid(valid)
        return valid

    def _validate_all(self):
        valid = (self._validate_le_element1() and
                 self._validate_le_element2())
        # Energy doesn't influence the success of validation, but the Ok button
        #   should be disabled if energy if out of range.
        valid_energy = self._validate_peak_energy()
        self.pb_ok.setEnabled(valid and valid_energy)
        return valid

    def _validate_eline(self, le_widget, text):
        if text is None:
            text = le_widget.text()

        valid = True
        if self._validator_eline.validate(text, 0)[0] != QDoubleValidator.Acceptable:
            valid = False
        else:
            # Try to compute energy for the pileup peak
            if self._compute_energy(text, text) == 0:
                valid = False
        le_widget.setValid(valid)
        return valid

    def _refresh_energy(self):
        eline1 = self._data["element1"]
        eline2 = self._data["element2"]
        try:
            energy = self._compute_energy(eline1, eline2)
            self._data["energy"] = energy
            self._show_energy()
        except Exception:
            self._data["energy"] = 0
            self._show_energy()

    def le_element1_editing_finished(self):
        if self._validate_le_element1():
            self._data["element1"] = self.le_element1.text()
            self._refresh_energy()
        else:
            self.le_element1.setText(self._data["element1"])
        self._validate_all()

    def le_element2_editing_finished(self):
        if self._validate_le_element2():
            self._data["element2"] = self.le_element2.text()
            self._refresh_energy()
        else:
            self.le_element2.setText(self._data["element2"])
        self._validate_all()

    def le_element1_text_changed(self, text):
        if self._validate_all():
            self._data["element1"] = self.le_element1.text()
            print(f"element1 - {self._data['element1']}")
            self._refresh_energy()

    def le_element2_text_changed(self, text):
        if self._validate_all():
            self._data["element2"] = self.le_element2.text()
            print(f"element2 - {self._data['element2']}")
            self._refresh_energy()


class DialogNewUserPeak(QDialog):

    def __init__(self, parent=None):

        super().__init__(parent)

        self._data = {"name": "", "energy": 0}

        self.setWindowTitle("Add New User-Defined Peak")
        self.setMinimumWidth(400)

        self.le_name = LineEditReadOnly()
        set_tooltip(self.le_name, "<b>Name</b> of the user-defined peak.")
        self.le_energy = LineEditReadOnly()
        set_tooltip(self.le_energy,
                    "<b>Energy</b> (keV) of the center of the user-defined peak.")

        vbox = QVBoxLayout()

        grid = QGridLayout()
        grid.addWidget(QLabel("Peak name:"), 0, 0)
        grid.addWidget(self.le_name, 0, 1)
        grid.addWidget(QLabel("Energy, keV"), 1, 0)
        grid.addWidget(self.le_energy, 1, 1)
        vbox.addLayout(grid)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.button(QDialogButtonBox.Cancel).setDefault(True)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        vbox.addWidget(button_box)

        self.setLayout(vbox)

    def set_parameters(self, data):
        self._data = data.copy()
        self._show_data()

    def _format_float(self, v):
        return f"{v:.12g}"

    def _show_data(self, existing=None):
        self.le_name.setText(self._data["name"])
        self.le_energy.setText(self._format_float(self._data["energy"]))


class DialogEditUserPeakParameters(QDialog):

    def __init__(self, parent=None):
        super().__init__(parent)

        self._data = {"name": "", "maxv": 0, "energy": 0, "fwhm": 0}

        self.setWindowTitle("Edit User-Defined Peak Parameters")
        self.setMinimumWidth(400)

        self.le_name = LineEditReadOnly()
        set_tooltip(self.le_name, "<b>Name</b> of the user-defined peak.")
        self.le_intensity = LineEditExtended()
        set_tooltip(self.le_intensity,
                    "Peak <b>intensity</b>. Typically it is not necessary to set the "
                    "intensity precisely, since it is refined during fitting of total "
                    "spectrum.")
        self.le_energy = LineEditExtended()
        set_tooltip(self.le_energy,
                    "<b>Energy</b> (keV) of the center of the user-defined peak.")
        self.le_fwhm = LineEditExtended()
        set_tooltip(self.le_fwhm, "<b>FWHM</b> (in keV) of the user-defined peak.")

        self._validator = QDoubleValidator()

        vbox = QVBoxLayout()

        grid = QGridLayout()
        grid.addWidget(QLabel("Peak name:"), 0, 0)
        grid.addWidget(self.le_name, 0, 1)
        grid.addWidget(QLabel("Energy, keV"), 1, 0)
        grid.addWidget(self.le_energy, 1, 1)
        grid.addWidget(QLabel("Intensity:"), 2, 0)
        grid.addWidget(self.le_intensity, 2, 1)
        grid.addWidget(QLabel("FWHM, keV"), 3, 0)
        grid.addWidget(self.le_fwhm, 3, 1)
        vbox.addLayout(grid)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.button(QDialogButtonBox.Cancel).setDefault(True)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        self.pb_ok = button_box.button(QDialogButtonBox.Ok)

        vbox.addWidget(button_box)

        self.setLayout(vbox)

        self.le_intensity.editingFinished.connect(self.le_intensity_editing_finished)
        self.le_energy.editingFinished.connect(self.le_energy_editing_finished)
        self.le_fwhm.editingFinished.connect(self.le_fwhm_editing_finished)

        self.le_intensity.textChanged.connect(self.le_intensity_text_changed)
        self.le_energy.textChanged.connect(self.le_energy_text_changed)
        self.le_fwhm.textChanged.connect(self.le_fwhm_text_changed)

    def set_parameters(self, data):
        self._data = data.copy()
        self._show_data()

    def get_parameters(self):
        return self._data

    def _format_float(self, v):
        return f"{v:.12g}"

    def _show_data(self, existing=None):
        self.le_name.setText(self._data["name"])
        self.le_energy.setText(self._format_float(self._data["energy"]))
        self.le_intensity.setText(self._format_float(self._data["maxv"]))
        self.le_fwhm.setText(self._format_float(self._data["fwhm"]))

    # The following validation functions are identical, but they will be different,
    #   because the relationships
    def _validate_le_intensity(self, text=None):
        return self._validate_le(self.le_intensity, text, lambda v: v > 0)

    def _validate_le_energy(self, text=None):
        return self._validate_le(self.le_energy, text, lambda v: v > 0)

    def _validate_le_fwhm(self, text=None):
        return self._validate_le(self.le_fwhm, text, lambda v: v > 0)

    def _validate_all(self):
        valid = (self._validate_le_intensity() and
                 self._validate_le_energy() and
                 self._validate_le_fwhm())
        self.pb_ok.setEnabled(valid)

    def _validate_le(self, le_widget, text, condition):
        if text is None:
            text = le_widget.text()
        valid = True
        if self._validator.validate(text, 0)[0] != QDoubleValidator.Acceptable:
            valid = False
        elif not condition(float(text)):
            valid = False
        le_widget.setValid(valid)
        return valid

    def le_intensity_editing_finished(self):
        if self._validate_le_intensity():
            self._data["maxv"] = float(self.le_intensity.text())
        else:
            self.le_intensity.setText(self._format_float(self._data["maxv"]))
        self._validate_all()

    def le_energy_editing_finished(self):
        if self._validate_le_energy():
            self._data["energy"] = float(self.le_energy.text())
        else:
            self.le_energy.setText(self._format_float(self._data["energy"]))
        self._validate_all()

    def le_fwhm_editing_finished(self):
        if self._validate_le_fwhm():
            self._data["fwhm"] = float(self.le_fwhm.text())
        else:
            self.le_fwhm.setText(self._format_float(self._data["fwhm"]))
        self._validate_all()

    def le_intensity_text_changed(self, text):
        self._validate_all()

    def le_energy_text_changed(self, text):
        self._validate_all()

    def le_fwhm_text_changed(self, text):
        self._validate_all()
