from qtpy.QtWidgets import (QHBoxLayout, QVBoxLayout, QLabel, QComboBox, QDialog,
                            QDialogButtonBox, QTableWidget, QTableWidgetItem, QHeaderView,)
from qtpy.QtGui import QBrush, QColor, QPalette
from qtpy.QtCore import Qt

from .useful_widgets import (get_background_css, set_tooltip, ComboBoxNamedNoWheel)

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

value_names = {
    "value": "Value",
    "min": "Min",
    "max": "Max",
}


class DialogDetailedFittingParameters(QDialog):

    def __init__(self, parent=None, *, dialog_data):

        super().__init__(parent)

        self._enable_events = False

        self._set_dialog_data(dialog_data)
        self._selected_index = -1

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
        self._enable_events = True

    def _setup_element_selection(self):

        self.combo_element_sel = QComboBox()
        set_tooltip(self.combo_element_sel,
                    "Select K, L or M <b>emission line</b> to edit the optimization parameters "
                    "used for the line during total spectrum fitting.")
        self.combo_element_sel.currentIndexChanged.connect(self.combo_element_sel_current_index_changed)

        hbox = QHBoxLayout()
        hbox.addWidget(QLabel("Select element:"))
        hbox.addWidget(self.combo_element_sel)
        hbox.addStretch(1)

        return hbox

    def _setup_table(self):

        self._value_keys = ("value", "min", "max")

        # Labels for horizontal header
        labels_presets = [fitting_preset_names[_] for _ in self._fit_strategy_list]
        labels_values = [value_names[_] for _ in self._value_keys]
        self.tbl_labels = ["Name", "E, keV"] + labels_values + labels_presets
        # Labels for editable columns
        self.tbl_cols_editable = ("Value", "Min", "Max")
        # Labels for the columns that contain combo boxes
        self.tbl_cols_combobox = labels_presets
        # The list of columns with fixed size
        self.tbl_cols_stretch = ("Value", "Min", "Max")
        # Table item representation if different from default
        self.tbl_format = {"E, keV": ".4f", "Value": ".8g", "Min": ".8g", "Max": ".8g"}

        # Combobox items. All comboboxes in the table contain identical list of items.
        self.combo_items = self._bound_options

        self._combo_list = []

        self.table = QTableWidget()
        self.table.setColumnCount(len(self.tbl_labels))
        self.table.verticalHeader().hide()
        self.table.setHorizontalHeaderLabels(self.tbl_labels)

        self.table.setStyleSheet("QTableWidget::item{color: black;}")

        header = self.table.horizontalHeader()
        for n, lbl in enumerate(self.tbl_labels):
            # Set stretching for the columns
            if lbl in self.tbl_cols_stretch:
                header.setSectionResizeMode(n, QHeaderView.Stretch)
            else:
                header.setSectionResizeMode(n, QHeaderView.ResizeToContents)

        self.table.itemChanged.connect(self.tbl_elines_item_changed)
        set_tooltip(self.table,
                    "Edit optimization parameters for the selected emission line. "
                    "Processing presets may be configured by specifying optimization strategy "
                    "for each parameter may be selected. A preset for each fitting step "
                    "of the total spectrum fitting may be selected in <b>Model</b> tab.")

    def _fill_table(self, table_contents):
        self._enable_events = False

        # Clear the list of combo boxes
        for item in self._combo_list:
            item.currentIndexChanged.disconnect(self.combo_strategy_current_index_changed)
        self._combo_list = []

        self.table.clearContents()

        self.table.setRowCount(len(table_contents))
        for nr, row in enumerate(table_contents):
            n_fit_strategy = 0
            row_name = row[0]
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

                    self.table.setItem(nr, nc, item)

                else:
                    if n_fit_strategy < len(self._fit_strategy_list):
                        combo_name = f"{row_name},{self._fit_strategy_list[n_fit_strategy]}"
                    else:
                        combo_name = ""
                    n_fit_strategy += 1

                    item = ComboBoxNamedNoWheel(name=combo_name)

                    # Set text color for QComboBox widget (necessary if the program is used with Dark theme)
                    pal = item.palette()
                    pal.setColor(QPalette.ButtonText, Qt.black)
                    item.setPalette(pal)
                    # Set text color for drop-down view (necessary if the program is used with Dark theme)
                    pal = item.view().palette()
                    pal.setColor(QPalette.Text, Qt.black)
                    item.view().setPalette(pal)

                    css1 = get_background_css(rgb_bckg, widget="QComboBox", editable=False)
                    css2 = get_background_css(rgb_bckg, widget="QWidget", editable=False)
                    item.setStyleSheet(css2 + css1)

                    item.addItems(self.combo_items)
                    if item.findText(entry) < 0:
                        logger.warning(f"Text '{entry}' is not found. The ComboBox is not set properly.")
                    item.setCurrentText(entry)  # Try selecting the item anyway
                    self.table.setCellWidget(nr, nc, item)

                    item.currentIndexChanged.connect(self.combo_strategy_current_index_changed)
                    self._combo_list.append(item)

        self._enable_events = True

    def combo_element_sel_current_index_changed(self, index):
        self._selected_index = index
        self._update_table()

    def combo_strategy_current_index_changed(self, name, index):
        if self._enable_events:
            try:
                name_row, name_strategy = name.split(",")
                option = self._bound_options[index]
                self._param_dict[name_row][name_strategy] = option
            except Exception as ex:
                logger.error(f"Error occurred while changing strategy options: {ex}")

    def tbl_elines_item_changed(self, item):
        if self._enable_events:
            try:
                n_row, n_col = self.table.row(item), self.table.column(item)
                n_key = n_col - 2
                if n_key < 0 or n_key >= len(self._value_keys):
                    raise RuntimeError(f"Incorrect column {n_col}")
                value_key = self._value_keys[n_key]
                eline_key = self._table_contents[n_row][0]
                try:
                    value = float(item.text())
                    self._param_dict[eline_key][value_key] = value
                except Exception:
                    value = self._param_dict[eline_key][value_key]
                    item.setText(f"{value:.8g}")
            except Exception as ex:
                logger.error(f"Error occurred while setting edited value: {ex}")

    def _set_combo_element_sel_items(self):
        element_list = ["Other Parameters"] + self._eline_list
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
        self._enable_events = False

        eline_list = []
        if self._selected_index == 0:
            eline_list = self._other_param_list
            energy_list = [""] * len(eline_list)
        elif self._selected_index > 0:
            eline = self._eline_list[self._selected_index - 1]
            eline_list = self._eline_key_dict[eline]
            energy_list = self._eline_energy_dict[eline]

        self._table_contents = []
        for n, key in enumerate(eline_list):
            data = [key, energy_list[n],
                    self._param_dict[key]["value"],
                    self._param_dict[key]["min"],
                    self._param_dict[key]["max"]]
            for strategy in self._fit_strategy_list:
                data.append(self._param_dict[key][strategy])
            self._table_contents.append(data)

        self._fill_table(self._table_contents)
        self._enable_events = True

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
