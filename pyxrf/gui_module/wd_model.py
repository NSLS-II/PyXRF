import numpy as np
from PyQt5.QtWidgets import (QPushButton, QHBoxLayout, QVBoxLayout,
                             QGroupBox, QLineEdit, QCheckBox, QLabel,
                             QComboBox, QListWidget, QListWidgetItem,
                             QDialog, QDialogButtonBox, QFileDialog,
                             QRadioButton, QButtonGroup, QGridLayout,
                             QTextEdit, QTableWidget, QTableWidgetItem,
                             QHeaderView, QWidget)
from PyQt5.QtGui import QWindow, QBrush, QColor
from PyQt5.QtCore import Qt

from .useful_widgets import LineEditReadOnly, global_gui_parameters, global_gui_variables
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

        self.setLayout(vbox)

    def _setup_model_params_group(self):

        self.group_model_params = QGroupBox("Load/Save Model Parameters")

        self.pb_find_elines = QPushButton("Find Automatically ...")
        self.pb_find_elines.setToolTip(
            "Find emission lines automatically from <b>total spectrum</b>. "
            "Press to open the dialog.")
        self.pb_load_elines = QPushButton("Load From File ...")
        self.pb_load_elines.setToolTip(
            "Load model (emission lines) parameters from <b>JSON</b> file, which was previously "
            "save using <b>Save Parameters to File ...</b>. Press to open the dialog.")
        self.pb_load_qstandard = QPushButton("Load Quantitative Standard ...")
        self.pb_load_qstandard.setToolTip(
            "Load <b>quantitative standard</b>. The model is reset and the emission lines "
            "that fit within the selected range of energy will be added to the list "
            "of emission lines. Press to open the dialog.")
        self.pb_save_elines = QPushButton("Save Parameters to File ...")
        self.pb_save_elines.setToolTip(
            "Save computed model (emission line) parameters to <b>JSON</b> file. "
            "Press to open the dialog.")
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

    def pb_manage_emission_lines_clicked(self, event):
        if not self.ref_main_window.wnd_manage_emission_lines.isVisible():
            self.ref_main_window.wnd_manage_emission_lines.show()
        self.ref_main_window.wnd_manage_emission_lines.activateWindow()


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
                    item.setTextAlignment(Qt.AlignCenter)
                else:
                    item.setTextAlignment(Qt.AlignRight)

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
