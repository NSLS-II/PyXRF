import os
import numpy as np

from PyQt5.QtWidgets import (QPushButton, QHBoxLayout, QVBoxLayout, QGroupBox, QLineEdit,
                             QCheckBox, QLabel, QComboBox, QDialog, QDialogButtonBox,
                             QFileDialog, QRadioButton, QButtonGroup, QGridLayout, QTableWidget,
                             QTableWidgetItem, QHeaderView, QMessageBox)
from PyQt5.QtGui import QBrush, QColor
from PyQt5.QtCore import Qt, pyqtSlot, QTimer, pyqtSignal

from .useful_widgets import (LineEditReadOnly, global_gui_parameters, ElementSelection,
                             get_background_css, SecondaryWindow, set_tooltip)

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
            # TODO: emit signal ('parameters changed', i.e. incident energy and range changed)
            if dlg.find_elements_requested:
                # TODO: start element search
                logger.debug("Starting automated element search")

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

                self.gpc.find_peaks()

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

    def __init__(self,  *, gpc, gui_vars):
        super().__init__()

        # Global processing classes
        self.gpc = gpc
        # Global GUI variables (used for control of GUI state)
        self.gui_vars = gui_vars

        self.initialize()

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

        self.element_selection = ElementSelection()

        # The following field should switched to 'editable' state from when needed
        self.le_peak_intensity = LineEditReadOnly()
        self.pb_add_eline = QPushButton("Add")
        self.pb_remove_eline = QPushButton("Remove")

        self.pb_user_peaks = QPushButton("User Peaks ...")
        self.pb_user_peaks.clicked.connect(self.pb_user_peaks_clicked)
        self.pb_pileup_peaks = QPushButton("Pileup Peaks ...")
        self.pb_pileup_peaks.clicked.connect(self.pb_pileup_peaks_clicked)

        # Some emission lines to populate the combo box
        eline_sample_list = ["Li_K", "B_K", "C_K", "N_K", "Fe_K", "Userpeak1"]
        self.element_selection.addItems(eline_sample_list)

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

        # Fill the table with some sample data
        sample_table = [[18, "Ar_K", 2.9574, 146548.42, 3.7, 268.08],
                        [20, "Ca_K", 3.6917, 119826.75, 3.02, 561.45],
                        [22, "Ti_K", 4.5109, 323794.32, 8.17, 1066.53],
                        [26, "Fe_K", 6.4039, 3964079.85, 100.0, 3025.15],
                        [30, "Zn_K", 8.6389, 41893.18, 1.06, 6706.05],
                        [65, "Tb_L", 6.2728, 11853.24, 0.3, 6148.37],
                        ["", "Userpeak1", "", 28322.97, 0.71, ""],
                        ["", "compton", "", 2342.37, 0.05, ""],
                        ["", "elastic", "", 8825.48, 0.22, ""],
                        ["", "background", "", 10118.05, 0.26, ""]]

        self.fill_eline_table(sample_table)

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

        self.tbl_elines.setRowCount(len(table_contents))
        for nr, row in enumerate(table_contents):
            for nc, entry in enumerate(row):
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
                    item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
                    item.setCheckState(Qt.Checked)  # All items are checked
                    item.setTextAlignment(Qt.AlignCenter | Qt.AlignVCenter)
                else:
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)

                # Set all columns not editable (unless needed)
                if label not in self.tbl_cols_editable:
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)

                # Make all items not selectable (we are not using selections)
                item.setFlags(item.flags() & ~Qt.ItemIsSelectable)

                # Set alternating background colors for the table rows
                #   Make background for editable items a little brighter
                brightness = 240 if label in self.tbl_cols_editable else 220
                if nr % 2:
                    brush = QBrush(QColor(255, brightness, brightness))  # Light-blue
                else:
                    brush = QBrush(QColor(brightness, 255, brightness))  # Light-green
                item.setBackground(brush)

                self.tbl_elines.setItem(nr, nc, item)

    def pb_pileup_peaks_clicked(self):
        dlg = DialogPileupPeakParameters()
        if dlg.exec():
            print("Pileup peak is added")

    def pb_user_peaks_clicked(self):
        dlg = DialogUserPeakParameters()
        if dlg.exec():
            print("User defined peak is added")


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

        self.setWindowTitle("Pileup Peak Parameters")

        self.rb_edit_existing = QRadioButton("Edit Existing")
        set_tooltip(self.rb_edit_existing,
                    "Edit parameters of the <b>existing</b> user-defined peak")
        self.rb_add_new = QRadioButton("Add New")
        set_tooltip(self.rb_add_new, "Add <b>new</b> user-defined peak.")

        self.btn_group = QButtonGroup()
        self.btn_group.addButton(self.rb_edit_existing)
        self.btn_group.addButton(self.rb_add_new)

        self.rb_add_new.setChecked(True)

        self.le_element1 = QLineEdit()
        set_tooltip(self.le_element1,
                    "The <b>name</b> of the emission line #1")
        self.le_element2 = QLineEdit()
        set_tooltip(self.le_element2,
                    "The <b>name</b> of the emission line #2")
        self.peak_intensity = QLineEdit()
        set_tooltip(self.peak_intensity,
                    "The <b>intensity</b> of the pileup peak. Typically it is not necessary "
                    "to set the intensity precisely, since it is refined during fitting of total "
                    "spectrum.")

        instructions = QLabel("Specify two emission lines, e.g. Si_Ka1 and Fe_Ka1")

        grid = QGridLayout()
        grid.addWidget(QLabel("Emission line 1:"), 0, 0)
        grid.addWidget(self.le_element1, 0, 1)
        grid.addWidget(QLabel("Emission line 2:"), 1, 0)
        grid.addWidget(self.le_element2, 1, 1)
        grid.addWidget(QLabel("Peak intensity:"), 2, 0)
        grid.addWidget(self.peak_intensity, 2, 1)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.button(QDialogButtonBox.Cancel).setDefault(True)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        vbox = QVBoxLayout()

        hbox = QHBoxLayout()
        hbox.addStretch(1)
        hbox.addWidget(self.rb_edit_existing)
        hbox.addWidget(self.rb_add_new)
        hbox.addStretch(1)
        vbox.addLayout(hbox)

        vbox.addSpacing(10)

        vbox.addWidget(instructions)
        vbox.addSpacing(10)
        vbox.addLayout(grid)
        vbox.addWidget(button_box)

        self.setLayout(vbox)


class DialogUserPeakParameters(QDialog):

    def __init__(self, parent=None):

        super().__init__(parent)

        self.setWindowTitle("User Defined Peak Parameters")
        self.setMinimumWidth(400)

        self.rb_edit_existing = QRadioButton("Edit Existing")
        set_tooltip(self.rb_edit_existing,
                    "Edit parameters of the <b>existing</b> user-defined peak")
        self.rb_add_new = QRadioButton("Add New")
        set_tooltip(self.rb_add_new, "Add <b>new</b> user-defined peak.")

        self.btn_group = QButtonGroup()
        self.btn_group.addButton(self.rb_edit_existing)
        self.btn_group.addButton(self.rb_add_new)

        self.rb_add_new.setChecked(True)

        self.le_name = LineEditReadOnly()
        set_tooltip(self.le_name, "<b>Name</b> of the user-defined peak.")
        self.le_intensity = QLineEdit()
        set_tooltip(self.le_intensity,
                    "Peak <b>intensity</b>. Typically it is not necessary to set the "
                    "intensity precisely, since it is refined during fitting of total "
                    "spectrum.")
        self.le_energy = QLineEdit()
        set_tooltip(self.le_energy,
                    "<b>Energy</b> (keV) of the center of the user-defined peak.")
        self.le_fwhm = QLineEdit()
        set_tooltip(self.le_fwhm, "<b>FWHM</b> (in keV) of the user-defined peak.")

        vbox = QVBoxLayout()

        hbox = QHBoxLayout()
        hbox.addStretch(1)
        hbox.addWidget(self.rb_edit_existing)
        hbox.addWidget(self.rb_add_new)
        hbox.addStretch(1)
        vbox.addLayout(hbox)

        grid = QGridLayout()
        grid.addWidget(QLabel("Peak name:"), 0, 0)
        grid.addWidget(self.le_name, 0, 1)
        grid.addWidget(QLabel("Intensity:"), 1, 0)
        grid.addWidget(self.le_intensity, 1, 1)
        grid.addWidget(QLabel("Energy, keV"), 2, 0)
        grid.addWidget(self.le_energy, 2, 1)
        grid.addWidget(QLabel("FWHM, keV"), 3, 0)
        grid.addWidget(self.le_fwhm)
        vbox.addLayout(grid)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.button(QDialogButtonBox.Cancel).setDefault(True)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        vbox.addWidget(button_box)

        self.setLayout(vbox)
