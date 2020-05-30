import os
import numpy as np
import textwrap

from PyQt5.QtWidgets import (QPushButton, QHBoxLayout, QVBoxLayout,
                             QGroupBox, QLineEdit, QCheckBox, QLabel,
                             QComboBox, QListWidget, QListWidgetItem,
                             QDialog, QDialogButtonBox, QFileDialog,
                             QRadioButton, QButtonGroup, QGridLayout,
                             QTextEdit, QTableWidget, QTableWidgetItem,
                             QHeaderView, QWidget)
from PyQt5.QtGui import QWindow, QBrush, QColor, QPalette
from PyQt5.QtCore import Qt

from .useful_widgets import (LineEditReadOnly, global_gui_parameters, global_gui_variables,
                             ElementSelection)

from .form_base_widget import FormBaseWidget


class ModelWidget(FormBaseWidget):

    def __init__(self):
        super().__init__()
        self.initialize()

    def initialize(self):

        # Reference to the main window. The main window will hold
        #   references to all non-modal windows that could be opened
        #   from multiple places in the program.
        self.ref_main_window = global_gui_variables["ref_main_window"]

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

    def _setup_model_params_group(self):

        self.group_model_params = QGroupBox("Load/Save Model Parameters")

        self.pb_find_elines = QPushButton("Find Automatically ...")
        self.pb_find_elines.setToolTip(
            "Find emission lines automatically from <b>total spectrum</b>. "
            "Press to open the dialog.")
        self.pb_find_elines.clicked.connect(self.pb_find_elines_clicked)

        self.pb_load_elines = QPushButton("Load From File ...")
        self.pb_load_elines.setToolTip(
            "Load model (emission lines) parameters from <b>JSON</b> file, which was previously "
            "save using <b>Save Parameters to File ...</b>. Press to open the dialog.")
        self.pb_load_elines.clicked.connect(self.pb_load_elines_clicked)

        self.pb_load_qstandard = QPushButton("Load Quantitative Standard ...")
        self.pb_load_qstandard.setToolTip(
            "Load <b>quantitative standard</b>. The model is reset and the emission lines "
            "that fit within the selected range of energy will be added to the list "
            "of emission lines. Press to open the dialog.")
        self.pb_load_qstandard.clicked.connect(self.pb_load_qstandard_clicked)

        self.pb_save_elines = QPushButton("Save Parameters to File ...")
        self.pb_save_elines.setToolTip(
            "Save computed model (emission line) parameters to <b>JSON</b> file. "
            "Press to open the dialog.")
        self.pb_save_elines.clicked.connect(self.pb_save_elines_clicked)

        # This field will display the name of he last loaded parameter file,
        #   Serial/Name of the quantitative standard, or 'no parameters' message
        self.le_param_fln = LineEditReadOnly("No parameter file is loaded")
        self.le_param_fln.setToolTip(
            "The name of the recently loaded <b>parameter file</b> or serial number "
            "and name of the loaded <b>quantitative standard</b>")

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

        # TODO: those items should be loaded from some global source
        combo_items = ("None", "fit_with_tail", "free_more", "e_calibration",
                       "linear", "adjust_element1", "adjust_element2",
                       "adjust_element3")
        self.cb_step1 = QComboBox()
        self.cb_step1.addItems(combo_items)
        self.cb_step1.setCurrentIndex(1)  # Should also be set based on data
        self.cb_step2 = QComboBox()
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
        self.pb_save_spectrum = QPushButton("Save Spectrum/Fit")
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

    def pb_find_elines_clicked(self, event):

        dlg = DialogFindElements(self)
        ret = dlg.exec()
        if ret:
            print(f"Dialog exited with success: ret = {ret}")
            if dlg.find_elements_requested:
                print(f"Run automated element search")
            else:
                print(f"Save parameters without running element search")
        else:
            print(f"Dialog exited with 'Cancel'")

    def pb_load_elines_clicked(self, event):
        # TODO: Propagate current directory here and use it in the dialog call
        current_dir = os.path.expanduser("~")
        file_name = QFileDialog.getOpenFileName(self, "Select File with Model Parameters",
                                                current_dir,
                                                "JSON (*.json);; All (*)")
        file_name = file_name[0]
        if file_name:
            print(f"Loading model parameters from file: {file_name}")

    def pb_load_qstandard_clicked(self, event):
        dlg = DialogSelectQuantStandard()
        ret = dlg.exec()
        if ret:
            standard_index = dlg.selected_standard_index
            print(f"Loading quantitative standard: {standard_index}")
        else:
            print("Cancelled loading quantitative standard")

    def pb_save_elines_clicked(self, event):
        # TODO: Propagate full path to the saved file here
        fln = os.path.expanduser("~/model_parameters.json")
        file_name = QFileDialog.getSaveFileName(self, "Select File to Save Model Parameters",
                                                fln,
                                                "JSON (*.json);; All (*)")
        file_name = file_name[0]
        if file_name:
            print(f"Saving model parameters to file file: {file_name}")

    def pb_manage_emission_lines_clicked(self, event):
        if not self.ref_main_window.wnd_manage_emission_lines.isVisible():
            self.ref_main_window.wnd_manage_emission_lines.show()
        self.ref_main_window.wnd_manage_emission_lines.activateWindow()

    def pb_general_clicked(self, event):
        dlg = DialogGeneralFittingSettings()
        ret = dlg.exec()
        if ret:
            print("Dialog closed. Changes accepted.")
        else:
            print("Cancelled.")

    def pb_elements_clicked(self, event):
        dlg = DialogElementSettings()
        ret = dlg.exec()
        if ret:
            print("'Elements' dialog closed. Changes accepted.")
        else:
            print("Cancelled.")

    def pb_global_params_clicked(self, event):
        dlg = DialogGlobalParamsSettings()
        ret = dlg.exec()
        if ret:
            print("'Global Parameters' dialog closed. Changes accepted.")
        else:
            print("Cancelled.")

    def pb_save_spectrum_clicked(self, event):
        # TODO: Propagate current directory here and use it in the dialog call
        current_dir = os.path.expanduser("~")
        dir = QFileDialog.getExistingDirectory(
            self, "Select Directory to Save Spectrum/Fit", current_dir,
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks)
        if dir:
            print(f"Spectrum/Fit is saved to directory {dir}")
        else:
            print(f"Spectrum/Fit saving is cancelled")


class WndManageEmissionLines(QWidget):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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

    def _setup_select_elines(self):

        self.cb_select_all = QCheckBox("All")
        self.cb_select_all.setChecked(True)

        self.element_selection = ElementSelection()

        # The following field should switched to 'editable' state from when needed
        self.le_peak_intensity = LineEditReadOnly()
        self.pb_add_eline = QPushButton("Add")
        self.pb_remove_eline = QPushButton("Remove")

        self.pb_add_user_peak = QPushButton("Add User Peak ...")
        self.pb_add_pileup_peak = QPushButton("Add Pileup Peak ...")

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
        hbox.addWidget(self.pb_add_user_peak)
        hbox.addWidget(self.pb_add_pileup_peak)
        vbox.addLayout(hbox)

        # Wrap vbox into hbox, because it will be inserted into vbox
        hbox = QHBoxLayout()
        hbox.addLayout(vbox)
        hbox.addStretch(1)
        return hbox

    def _setup_select_deselect_buttons(self):
        self.cb_select_all = QCheckBox("All")
        self.cb_select_all.setChecked(True)
        vbox = QVBoxLayout()
        vbox.addSpacing(70)
        vbox.addWidget(self.cb_select_all)
        return vbox

    def _setup_select_elines_group(self):

        self.group_select_elines = QGroupBox("Add/Remove Emission Lines")

        self.cb_eline_list = QComboBox()
        # The following field should switched to 'editable' state from when needed
        self.le_peak_intensity = LineEditReadOnly()
        self.pb_add_eline = QPushButton("Add")
        self.pb_remove_eline = QPushButton("Remove")

        self.pb_add_user_peak = QPushButton("Add User Peak ...")
        self.pb_add_pileup_peak = QPushButton("Add Pileup Peak ...")

        # Some emission lines to populate the combo box
        eline_sample_list = ["Li_K", "B_K", "C_K", "N_K", "Fe_K"]
        self.cb_eline_list.addItems(eline_sample_list)

        vbox = QVBoxLayout()
        hbox = QHBoxLayout()
        hbox.addWidget(self.cb_eline_list)
        hbox.addWidget(self.le_peak_intensity)
        hbox.addWidget(self.pb_add_eline)
        hbox.addWidget(self.pb_remove_eline)
        vbox.addLayout(hbox)
        hbox = QHBoxLayout()
        hbox.addWidget(self.pb_add_user_peak)
        hbox.addWidget(self.pb_add_pileup_peak)
        vbox.addLayout(hbox)
        self.group_select_elines.setLayout(vbox)

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


class DialogFindElements(QDialog):

    def __init__(self, parent):

        super().__init__(parent)

        self.setWindowTitle("Find Elements in Sample")

        # Check this flag after the dialog is exited with 'True' value
        #   If this flag is True, then run element search, if False,
        #   then simply save the changed parameter values.
        self.find_elements_requested = False

        self.le_e_calib_a0 = QLineEdit()
        self.pb_e_calib_a0_default = QPushButton("Default")
        self.le_e_calib_a1 = QLineEdit()
        self.pb_e_calib_a1_default = QPushButton("Default")
        self.le_e_calib_a2 = QLineEdit()
        self.pb_e_calib_a2_default = QPushButton("Default")
        self.group_energy_axis_calib = QGroupBox("Polynomial Approximation of Energy Axis")
        grid = QGridLayout()
        grid.addWidget(QLabel("Bias (a0):"), 0, 0)
        grid.addWidget(self.le_e_calib_a0, 0, 1)
        grid.addWidget(self.pb_e_calib_a0_default, 0, 2)
        grid.addWidget(QLabel("Linear (a1):"), 1, 0)
        grid.addWidget(self.le_e_calib_a1, 1, 1)
        grid.addWidget(self.pb_e_calib_a1_default, 1, 2)
        grid.addWidget(QLabel("Quadratic (a2):"), 2, 0)
        grid.addWidget(self.le_e_calib_a2, 2, 1)
        grid.addWidget(self.pb_e_calib_a2_default, 2, 2)
        self.group_energy_axis_calib.setLayout(grid)

        self.le_fwhm_b1 = QLineEdit()
        self.pb_fwhm_b1_default = QPushButton("Default")
        self.le_fwhm_b2 = QLineEdit()
        self.pb_fwhm_b2_default = QPushButton("Default")
        self.group_fwhm = QGroupBox("Peak FWHM Settings")
        grid = QGridLayout()
        grid.addWidget(QLabel("Coefficient b1:"), 0, 0)
        grid.addWidget(self.le_fwhm_b1, 0, 1)
        grid.addWidget(self.pb_fwhm_b1_default, 0, 2)
        grid.addWidget(QLabel("Coefficient b2:"), 1, 0)
        grid.addWidget(self.le_fwhm_b2, 1, 1)
        grid.addWidget(self.pb_fwhm_b2_default, 1, 2)
        self.group_fwhm.setLayout(grid)

        self.le_incident_energy = QLineEdit()
        self.pb_incident_energy_default = QPushButton("Default")
        self.le_range_low = QLineEdit()
        self.pb_range_low_default = QPushButton("Default")
        self.le_range_high = QLineEdit()
        self.pb_range_high_default = QPushButton("Default")
        self.group_energy_range = QGroupBox("Incident Energy and Selected Range")
        grid = QGridLayout()
        grid.addWidget(QLabel("Incident energy, keV"), 0, 0)
        grid.addWidget(self.le_incident_energy, 0, 1)
        grid.addWidget(self.pb_incident_energy_default, 0, 2)
        grid.addWidget(QLabel("Range (low), keV"), 1, 0)
        grid.addWidget(self.le_range_low, 1, 1)
        grid.addWidget(self.pb_range_low_default, 1, 2)
        grid.addWidget(QLabel("Range (high), keV"), 2, 0)
        grid.addWidget(self.le_range_high, 2, 1)
        grid.addWidget(self.pb_range_high_default, 2, 2)
        self.group_energy_range.setLayout(grid)

        self.pb_find_elements = QPushButton("Find &Elements")
        self.pb_find_elements.clicked.connect(self.pb_find_elements_clicked)
        self.pb_apply_settings = QPushButton("&Apply Settings")

        # 'Close' button box
        button_box = QDialogButtonBox(
            QDialogButtonBox.Cancel)
        button_box.addButton(self.pb_find_elements, QDialogButtonBox.YesRole)
        button_box.addButton(self.pb_apply_settings, QDialogButtonBox.AcceptRole)
        button_box.button(QDialogButtonBox.Cancel).setDefault(True)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        vbox = QVBoxLayout()
        vbox.addWidget(self.group_energy_axis_calib)
        vbox.addWidget(self.group_fwhm)
        vbox.addWidget(self.group_energy_range)
        vbox.addWidget(button_box)

        self.setLayout(vbox)

    def pb_find_elements_clicked(self, event):
        self.find_elements_requested = True


class DialogSelectQuantStandard(QDialog):

    def __init__(self, parent=None):

        super().__init__(parent)

        self.setWindowTitle("Load Quantitative Standard")

        self.setMinimumHeight(500)
        self.setMinimumWidth(600)
        self.resize(600, 500)

        self.selected_standard_index = -1

        labels = ("Serial #", "Name", "Description")
        col_stretch = (QHeaderView.ResizeToContents,
                       QHeaderView.ResizeToContents,
                       QHeaderView.Stretch)
        sample_table_contents = [
            ["41147", "Micromatter 41147",
             "GaP 21.2 (Ga=15.4, P=5.8) / CaF2 14.6 / V 26.4 / Mn 17.5 / Co 21.7 / Cu 20.3"],
            ["41148", "Micromatter 41148",
             "GaP 21.1 (Ga=15.4, P=5.7) / CaF2 14.6 / V 25.1 / Mn 18.1 / Co 20.6 / Cu 20.5"],
            ["41151", "Micromatter 41151",
             "KCl 21.9 / Ti 21.3 / Fe 24.4 / ZnTe 20.3 / Pb 22.3"],
            ["41164", "Micromatter 41164",
             "CeF3 21.1 / Au 20.6"],
        ]
        self.table = QTableWidget()
        self.table.setMinimumHeight(200)
        self.table.setRowCount(len(sample_table_contents))
        self.table.setColumnCount(len(labels))
        self.table.verticalHeader().hide()
        self.table.setHorizontalHeaderLabels(labels)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.itemSelectionChanged.connect(self.item_selection_changed)
        self.table.itemDoubleClicked.connect(self.item_double_clicked)

        header = self.table.horizontalHeader()
        for n, col_stretch in enumerate(col_stretch):
            # Set stretching for the columns
            header.setSectionResizeMode(n, col_stretch)

        for nr, row in enumerate(sample_table_contents):
            for nc, entry in enumerate(row):
                s = textwrap.fill(entry, width=40)
                item = QTableWidgetItem(s)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.table.setItem(nr, nc, item)
        self.table.resizeRowsToContents()

        brightness = 220
        for nr in range(self.table.rowCount()):
            for nc in range(self.table.columnCount()):
                self.table.item(nr, nc)
                if nr % 2:
                    color = QColor(255, brightness, brightness)
                else:
                    color = QColor(brightness, 255, brightness)
                self.table.item(nr, nc).setBackground(QBrush(color))

        button_box = QDialogButtonBox(
            QDialogButtonBox.Open | QDialogButtonBox.Cancel)
        button_box.button(QDialogButtonBox.Cancel).setDefault(True)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        self.pb_open = button_box.button(QDialogButtonBox.Open)
        self.pb_open.setEnabled(False)

        vbox = QVBoxLayout()
        vbox.addWidget(self.table)
        vbox.addWidget(button_box)
        self.setLayout(vbox)

    def item_selection_changed(self):
        sel_ranges = self.table.selectedRanges()
        # The table is configured to have one or no selected ranges
        # 'Open' button should be enabled only if a range (row) is selected
        if sel_ranges:
            self.selected_standard_index = sel_ranges[0].topRow()
            self.pb_open.setEnabled(True)
            #print(f"Selected row: {sel_ranges[0].topRow()}")
        else:
            self.selected_standard_index = -1
            self.pb_open.setEnabled(False)
            #print("No selection")

    def item_double_clicked(self):
        self.accept()


class DialogGeneralFittingSettings(QDialog):

    def __init__(self, parent=None):

        super().__init__(parent)

        self.setWindowTitle("General Settings for Fitting Algorithm")

        vbox_left = QVBoxLayout()

        # ===== Top-left section of the dialog box =====
        self.le_max_iterations = QLineEdit()
        self.le_tolerance_stopping = QLineEdit()
        self.le_escape_ratio = QLineEdit()
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
        self.le_range_low = QLineEdit()
        self.le_range_high = QLineEdit()
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
        self.cb_snip_baseline = QCheckBox("Subtract baseline using SNIP")

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

        # Labels for horizontal header
        self.tbl_labels = ["Name", "E, keV", "Value", "Min", "Max",
                           "With Tail", "Free", "E axis", "Area",
                           "Custom 1", "Custom 2", "Custom 3"]

        # Labels for editable columns
        self.tbl_cols_editable = ("Value", "Min", "Max")

        # Labels for the columns that contain comboboxes
        self.tbl_cols_combobox = ("With Tail", "Free", "E axis", "Area",
                                  "Custom 1", "Custom 2", "Custom 3")

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
        # table.setAlternatingRowColors(True)

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

                    # It is unclear how background color of QComboBox is set, but
                    #   experimentally it was found, that the color needs to be adjusted
                    #   (made darker) in order to match background colors of the cell.
                    # Also, changing color using palette didn't work (on Linux).
                    rgb_bckg = [_ - 35 if (_ < 255) else _ for _ in rgb_bckg]
                    color_css = f"rgb({rgb_bckg[0]}, {rgb_bckg[1]}, {rgb_bckg[2]})"
                    item.setStyleSheet(f"QComboBox {{ background-color: {color_css}; }}")

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
        cb_sample_items = ("Ca_K", "Ti_K", "Fe_K")
        self.cbox_element_sel.addItems(cb_sample_items)

        hbox = QHBoxLayout()
        hbox.addWidget(QLabel("Selecte element:"))
        hbox.addWidget(self.cbox_element_sel)
        hbox.addStretch(1)

        return hbox


    def _setup_table(self):

        self.table_settings = _FittingSettings(energy_column=True)
        self.table = self.table_settings.setup_table()

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
