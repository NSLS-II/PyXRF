from qtpy.QtWidgets import (
    QHBoxLayout,
    QVBoxLayout,
    QLabel,
    QComboBox,
    QMessageBox,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QPushButton,
    QCheckBox,
)
from qtpy.QtGui import QBrush, QColor, QPalette
from qtpy.QtCore import Qt, Signal, Slot, QThreadPool, QRunnable

from .useful_widgets import get_background_css, set_tooltip, ComboBoxNamedNoWheel, SecondaryWindow

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


class WndDetailedFittingParams(SecondaryWindow):

    # Signal that is sent (to main window) to update global state of the program
    update_global_state = Signal()
    computations_complete = Signal(object)

    def __init__(self, *, window_title, gpc, gui_vars):
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

        self._enable_events = False

        self._dialog_data = {}

        self._load_dialog_data()
        self._selected_index = 0
        self._selected_eline = "-"

        self.setWindowTitle(window_title)
        self.setMinimumWidth(1100)
        self.setMinimumHeight(500)
        self.resize(1100, 500)

        hbox_el_select = self._setup_element_selection()
        self._setup_table()

        vbox = QVBoxLayout()
        vbox.addLayout(hbox_el_select)
        vbox.addWidget(self.table)

        self.setLayout(vbox)

        self.update_form_data()
        self._enable_events = True
        self._data_changed = False

    def _setup_element_selection(self):

        self.combo_element_sel = QComboBox()
        self.combo_element_sel.setMinimumWidth(200)
        self.combo_element_sel.currentIndexChanged.connect(self.combo_element_sel_current_index_changed)

        self._auto_update = False
        self.cb_auto_update = QCheckBox("Auto")
        self.cb_auto_update.setCheckState(Qt.Checked if self._auto_update else Qt.Unchecked)
        self.cb_auto_update.stateChanged.connect(self.cb_auto_update_state_changed)

        self.pb_apply = QPushButton("Apply")
        self.pb_apply.setEnabled(False)
        self.pb_apply.clicked.connect(self.pb_apply_clicked)
        self.pb_cancel = QPushButton("Cancel")
        self.pb_cancel.setEnabled(False)
        self.pb_cancel.clicked.connect(self.pb_cancel_clicked)

        hbox = QHBoxLayout()
        hbox.addWidget(QLabel("Select element:"))
        hbox.addWidget(self.combo_element_sel)
        hbox.addStretch(1)
        hbox.addWidget(self.cb_auto_update)
        hbox.addWidget(self.pb_apply)
        hbox.addWidget(self.pb_cancel)

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
                        item.setTextAlignment(int(Qt.AlignRight | Qt.AlignVCenter))

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

    def _set_tooltips(self):
        set_tooltip(self.pb_apply, "Save changes and <b>update plots</b>.")
        set_tooltip(self.pb_cancel, "<b>Discard</b> all changes.")
        set_tooltip(
            self.cb_auto_update,
            "Automatically <b>update the plots</b> when changes are made. "
            "If unchecked, then button <b>Update Plots</b> must be pressed "
            "to update the plots. Automatic update is often undesirable "
            "when large maps are displayed and multiple changes to parameters "
            "are made.",
        )
        set_tooltip(
            self.combo_element_sel,
            "Select K, L or M <b>emission line</b> to edit the optimization parameters "
            "used for the line during total spectrum fitting.",
        )
        set_tooltip(
            self.table,
            "Edit optimization parameters for the selected emission line. "
            "Processing presets may be configured by specifying optimization strategy "
            "for each parameter may be selected. A preset for each fitting step "
            "of the total spectrum fitting may be selected in <b>Model</b> tab.",
        )

    def combo_element_sel_current_index_changed(self, index):
        self._selected_index = index
        try:
            self._selected_eline = self._eline_list[index]
        except Exception:
            self._selected_eline = "-"
        self._update_table()

    def cb_auto_update_state_changed(self, state):
        self._auto_update = state
        # 'Apply' and 'Cancel' button are always disabled in 'auto' mode (changes are automatically
        #   applied when switching to 'auto' mode), and there should be no pending changes if switching
        #   to 'manual update' mode. So ALWAYS use 'False'.
        self.pb_apply.setEnabled(False)
        self.pb_cancel.setEnabled(False)
        # If changes were made, apply the changes while switching to 'auto' mode
        self.save_form_data()

    def combo_strategy_current_index_changed(self, name, index):
        if self._enable_events:
            try:
                name_row, name_strategy = name.split(",")
                option = self._bound_options[index]
                self._param_dict[name_row][name_strategy] = option
            except Exception as ex:
                logger.error(f"Error occurred while changing strategy options: {ex}")

            self._data_changed = True
            self.auto_save_form_data()
            self._validate_all()

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

                self._data_changed = True
                self.auto_save_form_data()
                self._validate_all()

            except Exception as ex:
                logger.error(f"Error occurred while setting edited value: {ex}")

    def pb_apply_clicked(self):
        """Save dialog data and update plots"""
        self.save_form_data()

    def pb_cancel_clicked(self):
        """Reload data (discard all changes)"""
        self.update_form_data()

    def _set_combo_element_sel_items(self):
        element_list = self._eline_list
        self.combo_element_sel.clear()
        self.combo_element_sel.addItems(element_list)
        # Deselect all (this should clear the table)
        self.select_eline(self._selected_eline)

    def _set_dialog_data(self, dialog_data):
        self._param_dict = dialog_data["param_dict"]
        self._eline_list = dialog_data["eline_list"]
        self._eline_key_dict = dialog_data["eline_key_dict"]
        self._eline_energy_dict = dialog_data["eline_energy_dict"]
        self._other_param_list = dialog_data["other_param_list"]
        self._fit_strategy_list = dialog_data["fit_strategy_list"]
        self._bound_options = dialog_data["bound_options"]

    def _show_all(self):
        selected_eline = self._selected_eline
        self._set_combo_element_sel_items()
        self.select_eline(selected_eline)
        self._update_table()

    def _update_table(self):
        self._enable_events = False

        eline_list = []
        if "shared" in self._selected_eline.lower():  # Shared parameters
            eline_list = self._other_param_list
            energy_list = [""] * len(eline_list)
        elif self._selected_eline == self._eline_list[self._selected_index]:  # Emission lines
            eline = self._selected_eline
            eline_list = self._eline_key_dict[eline]
            energy_list = self._eline_energy_dict[eline]

        self._table_contents = []
        for n, key in enumerate(eline_list):
            data = [
                key,
                energy_list[n],
                self._param_dict[key]["value"],
                self._param_dict[key]["min"],
                self._param_dict[key]["max"],
            ]
            for strategy in self._fit_strategy_list:
                data.append(self._param_dict[key][strategy])
            self._table_contents.append(data)

        self._fill_table(self._table_contents)
        self._enable_events = True

    def select_eline(self, eline):
        if eline in self._eline_list:
            index = self._eline_list.index(eline)
        elif self._eline_list:
            index = 0
            eline = self._eline_list[0]
        else:
            index = -1
            eline = "-"
        self._selected_eline = eline
        self._selected_index = index
        self.combo_element_sel.setCurrentIndex(index)
        self._update_table()

    def update_widget_state(self, condition=None):
        # Update the state of the menu bar
        state = not self.gui_vars["gui_state"]["running_computations"]
        self.setEnabled(state)

        if condition == "tooltips":
            self._set_tooltips()

    def _validate_all(self):
        self.pb_apply.setEnabled(self._data_changed and not self._auto_update)
        self.pb_cancel.setEnabled(self._data_changed and not self._auto_update)
        self.cb_auto_update.setChecked(Qt.Checked if self._auto_update else Qt.Unchecked)

    def _load_dialog_data(self):
        ...

    def _save_dialog_data_function(self):
        raise NotImplementedError()

    def update_form_data(self):
        self._load_dialog_data()
        self._show_all()
        self._data_changed = False
        self._validate_all()

    def auto_save_form_data(self):
        if self._auto_update:
            self.save_form_data()

    def save_form_data(self):
        if self._data_changed:
            f_save_data = self._save_dialog_data_function()

            def cb(dialog_data):
                try:
                    f_save_data(dialog_data)
                    success, msg = True, ""
                except Exception as ex:
                    success, msg = False, str(ex)
                return {"success": success, "msg": msg}

            self._compute_in_background(cb, self.slot_save_form_data, dialog_data=self._dialog_data)

    @Slot(object)
    def slot_save_form_data(self, result):
        self._recover_after_compute(self.slot_save_form_data)

        if not result["success"]:
            msg = result["msg"]
            msgbox = QMessageBox(
                QMessageBox.Critical, "Failed to Apply Fit Parameters", msg, QMessageBox.Ok, parent=self
            )
            msgbox.exec()
        else:
            self._data_changed = False
            self._validate_all()

        self.gui_vars["gui_state"]["state_model_fit_exists"] = False
        self.update_global_state.emit()

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
            class RunTask(QRunnable):
                def run(self):
                    result_dict = func(*args, **kwargs)
                    signal_complete.emit(result_dict)

            return RunTask()

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


class WndDetailedFittingParamsLines(WndDetailedFittingParams):
    def __init__(self, *, gpc, gui_vars):
        window_title = "Fitting Parameters for Individual Emission Lines"
        super().__init__(window_title=window_title, gpc=gpc, gui_vars=gui_vars)

    def _load_dialog_data(self):
        self._dialog_data = self.gpc.get_detailed_fitting_params_lines()
        self._set_dialog_data(self._dialog_data)

    def _save_dialog_data_function(self):
        def f(dialog_data):
            self.gpc.set_detailed_fitting_params(dialog_data)

        return f


class WndDetailedFittingParamsShared(WndDetailedFittingParams):
    def __init__(self, *, gpc, gui_vars):
        window_title = "Shared Detailed Fitting Parameters"
        super().__init__(window_title=window_title, gpc=gpc, gui_vars=gui_vars)

    def _load_dialog_data(self):
        self._dialog_data = self.gpc.get_detailed_fitting_params_shared()
        self._set_dialog_data(self._dialog_data)

    def _save_dialog_data_function(self):
        def f(dialog_data):
            self.gpc.set_detailed_fitting_params(dialog_data)

        return f
