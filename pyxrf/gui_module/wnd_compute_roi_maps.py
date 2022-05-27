from qtpy.QtWidgets import (
    QPushButton,
    QHBoxLayout,
    QVBoxLayout,
    QCheckBox,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QWidget,
    QMessageBox,
)
from qtpy.QtGui import QBrush, QColor
from qtpy.QtCore import Qt, Slot, Signal, QThreadPool, QRunnable

from .useful_widgets import (
    SecondaryWindow,
    set_tooltip,
    LineEditExtended,
    PushButtonNamed,
    CheckBoxNamed,
    RangeManager,
)

import logging

logger = logging.getLogger(__name__)


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

        self.table.setStyleSheet("QTableWidget::item{color: black;}")

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
                        item.setTextAlignment(int(Qt.AlignCenter))
                    else:
                        item.setTextAlignment(int(Qt.AlignRight | Qt.AlignVCenter))

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
            item.setStyleSheet(
                f"QWidget {{ background-color: {color_css}; }} "
                f"QCheckBox {{ color: black; background-color: white }}"
            )
            self.table.setCellWidget(nr, nc + 1, item)

            item = PushButtonNamed("Reset", name=f"{nr}")
            item.clicked.connect(self.pb_default_clicked)
            self.pb_default_list.append(item)
            rgb_bckg = [_ - 35 if (_ < 255) else _ for _ in rgb_bckg]
            color_css = f"rgb({rgb_bckg[0]}, {rgb_bckg[1]}, {rgb_bckg[2]})"
            item.setStyleSheet(f"QPushButton {{ color: black; background-color: {color_css}; }}")
            self.table.setCellWidget(nr, nc + 2, item)

    def _setup_footer(self):

        self.cb_subtract_baseline = QCheckBox("Subtract baseline")
        self.cb_subtract_baseline.setChecked(
            Qt.Checked if self.gpc.get_roi_subtract_background() else Qt.Unchecked
        )
        self.cb_subtract_baseline.toggled.connect(self.cb_subtract_baseline_toggled)

        self.pb_compute_roi = QPushButton("Compute ROIs")
        self.pb_compute_roi.clicked.connect(self.pb_compute_roi_clicked)

        hbox = QHBoxLayout()
        hbox.addWidget(self.cb_subtract_baseline)
        hbox.addStretch(1)
        hbox.addWidget(self.pb_compute_roi)

        return hbox

    def _set_tooltips(self):
        set_tooltip(self.pb_clear, "<b>Clear</b> the list")
        set_tooltip(
            self.pb_use_lines_for_fitting,
            "Copy the contents of <b>the list of emission lines selected for fitting</b> to the list of ROIs",
        )
        set_tooltip(self.le_sel_emission_lines, "The list of <b>emission lines</b> selected for ROI computation.")

        set_tooltip(self.table, "The list of ROIs")

        set_tooltip(
            self.cb_subtract_baseline,
            "<b>Subtract baseline</b> from the pixel spectra before computing ROIs. "
            "Subtracting baseline slows down computations and usually have no benefit. "
            "In most cases it should remain <b>unchecked</b>.",
        )
        set_tooltip(
            self.pb_compute_roi,
            "<b>Run</b> computations of the ROIs. The resulting <b>ROI</b> dataset "
            "may be viewed in <b>XRF Maps</b> tab.",
        )

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
            msgbox = QMessageBox(QMessageBox.Critical, "Failed to Compute ROIs", msg, QMessageBox.Ok, parent=self)
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
