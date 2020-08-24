from PyQt5.QtWidgets import (QHBoxLayout, QVBoxLayout, QLabel, QComboBox, QDialog,
                             QDialogButtonBox, QTableWidget, QTableWidgetItem, QHeaderView,)
from PyQt5.QtGui import QBrush, QColor
from PyQt5.QtCore import Qt

from .useful_widgets import (get_background_css, set_tooltip)

import logging
logger = logging.getLogger(__name__)


fitting_preset_names = {
    "None": "None",
    "fit_with_tail": "With Tail",
    "free_more": "Free",
    "e_calibration": "E-axis",
    "linear": "Area",
    "adjust_element1": "Custom 1",
    "adjust_element2": "Custom 2",
    "adjust_element3": "Custom 3",
}


class DialogDetailedFittingParameters(QDialog):

    def __init__(self, parent=None, *, dialog_data):

        super().__init__(parent)

        self._set_dialog_data(dialog_data)
        self._selected_index = -1
        #self._param_dict = dict()
        #self._eline_list = []
        #self._eline_key_dict = dict()
        #self._other_param_list = []
        #self._fit_strategy_list = []
        #self._bound_options = []

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

        self._show_all()

    def _setup_element_selection(self):

        self.combo_element_sel = QComboBox()
        set_tooltip(self.combo_element_sel,
                    "Select K, L or M <b>emission line</b> to edit the optimization parameters "
                    "used for the line during total spectrum fitting.")
        #cb_sample_items = ("Ca_K", "Ti_K", "Fe_K")
        #self.cbox_element_sel.addItems(cb_sample_items)
        self.combo_element_sel.currentIndexChanged.connect(self.combo_element_sel_current_index_changed)

        hbox = QHBoxLayout()
        hbox.addWidget(QLabel("Select element:"))
        hbox.addWidget(self.combo_element_sel)
        hbox.addStretch(1)

        return hbox

    def _setup_table(self):

        self.table_settings = _FittingSettings(fit_strategy_list=self._fit_strategy_list,
                                               bound_options=self._bound_options,
                                               energy_column=True)
        self.table = self.table_settings.setup_table()
        set_tooltip(self.table,
                    "Edit optimization parameters for the selected emission line. "
                    "Processing presets may be configured by specifying optimization strategy "
                    "for each parameter may be selected. A preset for each fitting step "
                    "of the total spectrum fitting may be selected in <b>Model</b> tab.")

        #sample_contents = [
        #    ["Ca_ka1_area", 3.6917, 11799.14, 0, 10000000.0,
        #     "none", "none", "none", "none", "none", "fixed", "fixed"],
        #    ["Ca_ka1_delta_center", 3.6917, 0.0, -0.005, 0.005,
        #     "fixed", "fixed", "fixed", "fixed", "fixed", "fixed", "fixed"],
        #    ["Ca_ka1_delta_sigma", 3.6917, 0.0, -0.02, 0.02,
        #     "fixed", "fixed", "fixed", "fixed", "fixed", "fixed", "fixed"],
        #    ["Ca_ka1_ratio_adjust", 3.6917, 0.0, 0.1, 5.0,
        #     "fixed", "fixed", "fixed", "fixed", "fixed", "fixed", "fixed"],
        #]
        #self._fill_table([])

    def _fill_table(self, table_contents):
        self.table_settings.fill_table(self.table, table_contents)

    def combo_element_sel_current_index_changed(self, index):
        self._selected_index = index
        self._update_table()

    def _set_combo_element_sel_items(self):
        element_list = ["Other Parameters"] +  self._eline_list
        self.combo_element_sel.clear()
        self.combo_element_sel.addItems(element_list)
        # Deselect all (this should clear the table)

        self._selected_index = -1
        self.combo_element_sel.setCurrentIndex(self._selected_index)

    def _set_dialog_data(self, dialog_data):
        self._param_dict = dialog_data["param_dict"]
        self._eline_list = dialog_data["eline_list"]
        self._eline_key_dict = dialog_data["eline_key_dict"]
        self._eline_energy_dict = dialog_data["eline_energy_dict"]
        self._other_param_list = dialog_data["other_param_list"]
        self._fit_strategy_list = dialog_data["fit_strategy_list"]
        self._bound_options = dialog_data["bound_options"]

    def _show_all(self):
        self._set_combo_element_sel_items()
        self._update_table()

    def _update_table(self):
        eline_list = []
        if self._selected_index == 0:
            eline_list = self._other_param_list
            energy_list = [""] * len(eline_list)
        elif self._selected_index > 0:
            eline = self._eline_list[self._selected_index - 1]
            eline_list = self._eline_key_dict[eline]
            energy_list = self._eline_energy_dict[eline]

        table_contents = []
        for n, key in enumerate(eline_list):
            data = [key, energy_list[n],
                    self._param_dict[key]["value"],
                    self._param_dict[key]["min"],
                    self._param_dict[key]["max"]]
            for strategy in self._fit_strategy_list:
                data.append(self._param_dict[key][strategy])
            table_contents.append(data)

        self._fill_table(table_contents)
        #sample_contents = [
        #    ["Ca_ka1_area", 3.6917, 11799.14, 0, 10000000.0,
        #     "none", "none", "none", "none", "none", "fixed", "fixed"],
        #    ["Ca_ka1_delta_center", 3.6917, 0.0, -0.005, 0.005,
        #     "fixed", "fixed", "fixed", "fixed", "fixed", "fixed", "fixed"],
        #    ["Ca_ka1_delta_sigma", 3.6917, 0.0, -0.02, 0.02,
        #     "fixed", "fixed", "fixed", "fixed", "fixed", "fixed", "fixed"],
        #    ["Ca_ka1_ratio_adjust", 3.6917, 0.0, 0.1, 5.0,
        #     "fixed", "fixed", "fixed", "fixed", "fixed", "fixed", "fixed"],
        #]

    def select_eline(self, eline):
        if eline in self._eline_list:
            index = self._eline_list.index(eline) + 1
        elif self._eline_list:
            index = 0
        else:
            index = -1
        self._selected_index = index
        self.combo_element_sel.setCurrentIndex(index)
        self._update_table()

    def get_selected_eline(self):
        if self._selected_index > 0:
            eline = self._eline_list[self._selected_index - 1]
        else:
            eline = ""
        return eline


class _FittingSettings():

    def __init__(self, *, bound_options, fit_strategy_list, energy_column=True, ):

        labels_presets = [fitting_preset_names[_] for _ in fit_strategy_list]

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
        self.cbox_settings_items = bound_options

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

                if label not in self.tbl_cols_combobox:
                    continue
                    if label in self.tbl_format and not isinstance(entry, str):
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
                    #if (nc < 5) or (nc > 11):  ##
                    if nc != 5 or nr > 2: ##
                        print(f"column = {nc}")  ##
                        continue  ##
                    item = QComboBox()

                    css1 = get_background_css(rgb_bckg, widget="QComboBox", editable=False)
                    css2 = get_background_css(rgb_bckg, widget="QWidget", editable=False)
                    item.setStyleSheet(css2 + css1)

                    #item.addItems(self.cbox_settings_items)
                    if nc == 11 and nr == len(table_contents) - 1:  ##
                        entry = "hi"  ##
                    #if item.findText(entry) < 0:
                    #    logger.warning(f"Text '{entry}' is not found. The ComboBox is not set properly.")
                    #item.setCurrentText(entry)  # Try selecting the item anyway
                    table.setCellWidget(nr, nc, item)
