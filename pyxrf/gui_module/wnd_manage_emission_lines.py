import numpy as np
import copy

from qtpy.QtWidgets import (
    QPushButton,
    QHBoxLayout,
    QVBoxLayout,
    QCheckBox,
    QTableWidget,
    QWidget,
    QTableWidgetItem,
    QHeaderView,
    QMessageBox,
)
from qtpy.QtGui import QBrush, QColor, QDoubleValidator
from qtpy.QtCore import Qt, Slot, Signal

from .useful_widgets import (
    LineEditReadOnly,
    ElementSelection,
    SecondaryWindow,
    set_tooltip,
    LineEditExtended,
    CheckBoxNamed,
    get_background_css,
)

from .dlg_edit_user_peak_parameters import DialogEditUserPeakParameters
from .dlg_new_user_peak import DialogNewUserPeak
from .dlg_pileup_peak_parameters import DialogPileupPeakParameters

import logging

logger = logging.getLogger(__name__)


class WndManageEmissionLines(SecondaryWindow):

    signal_selected_element_changed = Signal(str)
    signal_update_element_selection_list = Signal()
    signal_update_add_remove_btn_state = Signal(bool, bool)
    signal_marker_state_changed = Signal(bool)

    signal_parameters_changed = Signal()

    def __init__(self, *, gpc, gui_vars):
        super().__init__()

        # Global processing classes
        self.gpc = gpc
        # Global GUI variables (used for control of GUI state)
        self.gui_vars = gui_vars

        # Threshold used for peak removal (displayed in lineedit)
        self._remove_peak_threshold = self.gpc.get_peak_threshold()

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

        self.element_selection.signal_current_item_changed.connect(self.element_selection_item_changed)

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

        self._validator_peak_height = QDoubleValidator()
        self._validator_peak_height.setBottom(0.01)

        self.tbl_elines = QTableWidget()
        self.tbl_elines.setStyleSheet(
            "QTableWidget::item{color: black;}"
            "QTableWidget::item:selected{background-color: red;}"
            "QTableWidget::item:selected{color: white;}"
        )

        self.tbl_labels = ["", "Z", "Line", "E, keV", "Peak Int.", "Rel. Int.(%)", "CS"]
        self.tbl_cols_editable = ["Peak Int."]
        self.tbl_value_min = {"Rel. Int.(%)": 0.1}
        tbl_cols_resize_to_content = ["", "Z", "Line"]

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

        self.cb_sel_list = []  # List of checkboxes

    def _setup_action_buttons(self):
        self.pb_remove_rel = QPushButton("Remove Rel.Int.(%) <")
        self.pb_remove_rel.clicked.connect(self.pb_remove_rel_clicked)

        self.le_remove_rel = LineEditExtended("")
        self._validator_le_remove_rel = QDoubleValidator()
        self._validator_le_remove_rel.setBottom(0.01)  # Some small number
        self._validator_le_remove_rel.setTop(100.0)
        self.le_remove_rel.setText(self._format_threshold(self._remove_peak_threshold))
        self._update_le_remove_rel_state()
        self.le_remove_rel.textChanged.connect(self.le_remove_rel_text_changed)
        self.le_remove_rel.editingFinished.connect(self.le_remove_rel_editing_finished)

        self.pb_remove_unchecked = QPushButton("Remove Unchecked Lines")
        self.pb_remove_unchecked.clicked.connect(self.pb_remove_unchecked_clicked)

        hbox = QHBoxLayout()
        hbox.addWidget(self.pb_remove_rel)
        hbox.addWidget(self.le_remove_rel)
        hbox.addStretch(1)
        hbox.addWidget(self.pb_remove_unchecked)

        return hbox

    def _set_tooltips(self):
        set_tooltip(self.cb_select_all, "<b>Select/Deselect All</b> emission lines in the list")
        set_tooltip(self.element_selection, "<b>Set active</b> emission line")
        set_tooltip(self.le_peak_intensity, "Set or modify <b>intensity</b> of the active peak.")
        set_tooltip(self.pb_add_eline, "<b>Add</b> emission line to the list.")
        set_tooltip(self.pb_remove_eline, "<b>Remove</b> emission line from the list.")
        set_tooltip(
            self.pb_user_peaks, "Open dialog box to add or modify parameters of the <b>user-defined peak</b>"
        )
        set_tooltip(self.pb_pileup_peaks, "Open dialog box to add or modify parameters of the <b>pileup peak</b>")

        set_tooltip(self.tbl_elines, "The list of the selected <b>emission lines</b>")

        # set_tooltip(self.pb_update,
        #             "Update the internally stored list of selected emission lines "
        #             "and their parameters. This button is <b>deprecated</b>, but still may be "
        #             "needed in some situations. In future releases it will be <b>removed</b> or replaced "
        #             "with 'Accept' button. Substantial changes to the computational code is needed before "
        #             "it happens.")
        # set_tooltip(self.pb_undo,
        #             "<b>Undo</b> changes to the table of selected emission lines. Doesn't always work.")
        set_tooltip(
            self.pb_remove_rel,
            "<b>Remove emission lines</b> from the list if their relative intensity is less "
            "then specified threshold.",
        )
        set_tooltip(
            self.le_remove_rel,
            "<b>Threshold</b> that controls which emission lines are removed "
            "when <b>Remove Rel.Int.(%)</b> button is pressed.",
        )
        set_tooltip(self.pb_remove_unchecked, "Remove <b>unchecked</b> emission lines from the list.")

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

        self.tbl_elines.clearContents()

        # Clear the list of checkboxes
        for cb in self.cb_sel_list:
            cb.stateChanged.connect(self.cb_eline_state_changed)
        self.cb_sel_list = []

        self.tbl_elines.setRowCount(len(table_contents))
        for nr, row in enumerate(table_contents):
            sel_status = row["sel_status"]
            row_data = [None, row["z"], row["eline"], row["energy"], row["peak_int"], row["rel_int"], row["cs"]]

            for nc, entry in enumerate(row_data):

                label = self.tbl_labels[nc]

                # Set alternating background colors for the table rows
                #   Make background for editable items a little brighter
                brightness = 240 if label in self.tbl_cols_editable else 220
                if nr % 2:
                    rgb_bckg = (255, brightness, brightness)  # Light-red
                else:
                    rgb_bckg = (brightness, 255, brightness)  # Light-green

                if nc == 0:
                    item = QWidget()
                    cb = CheckBoxNamed(name=f"{nr}")
                    item_hbox = QHBoxLayout(item)
                    item_hbox.addWidget(cb)
                    item_hbox.setAlignment(Qt.AlignCenter)
                    item_hbox.setContentsMargins(0, 0, 0, 0)

                    css1 = get_background_css(rgb_bckg, widget="QCheckbox", editable=False)
                    css2 = get_background_css(rgb_bckg, widget="QWidget", editable=False)
                    item.setStyleSheet(css2 + css1)

                    cb.setChecked(Qt.Checked if sel_status else Qt.Unchecked)
                    cb.stateChanged.connect(self.cb_eline_state_changed)
                    cb.setStyleSheet("QCheckBox {color: black;}")
                    self.cb_sel_list.append(cb)

                    self.tbl_elines.setCellWidget(nr, nc, item)
                else:

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

                    item.setTextAlignment(int(Qt.AlignRight | Qt.AlignVCenter))

                    # Set all columns not editable (unless needed)
                    if label not in self.tbl_cols_editable:
                        item.setFlags(item.flags() & ~Qt.ItemIsEditable)

                    brush = QBrush(QColor(*rgb_bckg))
                    item.setBackground(brush)

                    self.tbl_elines.setItem(nr, nc, item)

        self._enable_events = True
        # Update the rest of the widgets
        self._update_widgets_based_on_table_state()

    @Slot()
    def update_widget_data(self):
        # This is typically a new set of emission lines. Clear the selection both
        #   in the table and in the element selection tool.
        self.element_selection.set_current_item("")
        self.tbl_elines.clearSelection()
        self._set_selected_eline("")
        # Now update the tables
        self._update_eline_selection_list()
        self.update_eline_table()
        self._update_add_remove_btn_state()

    def pb_pileup_peaks_clicked(self):
        data = {}

        eline = self._selected_eline
        if self.gpc.get_eline_name_category(eline) == "pileup":
            logger.error(f"Attempt to add pileup peak '{eline}' while another pileup peak is selected.")
            return

        energy, marker_visible = self.gpc.get_suggested_manual_peak_energy()
        best_guess = self.gpc.get_guessed_pileup_peak_components(energy=energy, tolerance=0.1)
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
            msg = "Select location of the new peak center (energy)\nby clicking on the plot in 'Fit Model' tab"
            msgbox = QMessageBox(QMessageBox.Information, "User Input Required", msg, QMessageBox.Ok, parent=self)
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
                try:
                    data = dlg.get_parameters()
                    eline1, eline2 = data["element1"], data["element2"]
                    eline = self.gpc.generate_pileup_peak_name(eline1, eline2)
                    self.gpc.add_peak_manual(eline)
                    self.update_eline_table()  # Update the table
                    self.tbl_elines_set_selection(eline)  # Select new emission line
                    self._set_selected_eline(eline)
                    self._set_fit_status(False)
                    logger.info(f"New pileup peak {eline} was added")
                except RuntimeError as ex:
                    msg = str(ex)
                    msgbox = QMessageBox(QMessageBox.Critical, "Error", msg, QMessageBox.Ok, parent=self)
                    msgbox.exec()
                    # Reload the table anyway (nothing is going to be selected)
                    self.update_eline_table()

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
                try:
                    eline = data["name"]
                    data = dlg.get_parameters()
                    self.gpc.update_userpeak(data["name"], data["energy"], data["maxv"], data["fwhm"])
                    self._set_fit_status(False)
                    logger.info(f"User defined peak {eline} was updated.")
                except Exception as ex:
                    msg = str(ex)
                    msgbox = QMessageBox(QMessageBox.Critical, "Error", msg, QMessageBox.Ok, parent=self)
                    msgbox.exec()
                # Reload the table anyway (nothing is going to be selected)
                self.update_eline_table()

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
                        self.update_eline_table()  # Update the table
                        self.tbl_elines_set_selection(eline)  # Select new emission line
                        self._set_selected_eline(eline)
                        self._set_fit_status(False)
                        logger.info(f"New user defined peak {eline} is added")
                    except RuntimeError as ex:
                        msg = str(ex)
                        msgbox = QMessageBox(QMessageBox.Critical, "Error", msg, QMessageBox.Ok, parent=self)
                        msgbox.exec()
                        # Reload the table anyway (nothing is going to be selected)
                        self.update_eline_table()
            else:
                msg = (
                    "Select location of the new peak center (energy)\n"
                    "by clicking on the plot in 'Fit Model' tab"
                )
                msgbox = QMessageBox(
                    QMessageBox.Information, "User Input Required", msg, QMessageBox.Ok, parent=self
                )
                msgbox.exec()

    @Slot()
    def pb_add_eline_clicked(self):
        logger.debug("'Add line' clicked")
        # It is assumed that this button is active only if an element is selected from the list
        #   of available emission lines. It can't be used to add user-defined peaks or pileup peaks.
        eline = self._selected_eline
        if eline:
            try:
                self.gpc.add_peak_manual(eline)
                self.update_eline_table()  # Update the table
                self.tbl_elines_set_selection(eline)  # Select new emission line
                self._set_fit_status(False)
            except RuntimeError as ex:
                msg = str(ex)
                msgbox = QMessageBox(QMessageBox.Critical, "Error", msg, QMessageBox.Ok, parent=self)
                msgbox.exec()
                # Reload the table anyway (nothing is going to be selected)
                self.update_eline_table()

    @Slot()
    def pb_remove_eline_clicked(self):
        logger.debug("'Remove line' clicked")
        eline = self._selected_eline
        if eline:
            # If currently selected line is the emission line (like Ca_K), we want
            # it to remain selected after it is deleted. This means that nothing is selected
            # in the table. For other lines, nothing should remain selected.
            self.tbl_elines.clearSelection()
            self.gpc.remove_peak_manual(eline)
            self.update_eline_table()  # Update the table
            if self.gpc.get_eline_name_category(eline) != "eline":
                eline = ""
            # This will update widgets
            self._set_selected_eline(eline)
            self._set_fit_status(False)

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
            self.cb_sel_list[n_row].setChecked(Qt.Checked if to_check else Qt.Unchecked)
            eline_list.append(eline)
            state_list.append(to_check)

        self.gpc.set_checked_emission_lines(eline_list, state_list)
        self._set_fit_status(False)
        self._enable_events = True

    def cb_eline_state_changed(self, name, state):
        if self._enable_events:
            n_row = int(name)
            state = state == Qt.Checked
            eline = self._table_contents[n_row]["eline"]
            self.gpc.set_checked_emission_lines([eline], [state])
            self._set_fit_status(False)

    def tbl_elines_item_changed(self, item):
        if self._enable_events:
            n_row, n_col = self.tbl_elines.row(item), self.tbl_elines.column(item)
            # Value was changed
            if n_col == 4:
                text = item.text()
                eline = self._table_contents[n_row]["eline"]
                if self._validator_peak_height.validate(text, 0)[0] != QDoubleValidator.Acceptable:
                    val = self._table_contents[n_row]["peak_int"]
                    self._enable_events = False
                    item.setText(f"{val:.2f}")
                    self._enable_events = True
                    self._set_fit_status(False)
                else:
                    self.gpc.update_eline_peak_height(eline, float(text))
                    self.update_eline_table()

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

    def pb_remove_rel_clicked(self):
        try:
            self.gpc.remove_peaks_below_threshold(self._remove_peak_threshold)
        except Exception as ex:
            msg = str(ex)
            msgbox = QMessageBox(QMessageBox.Critical, "Error", msg, QMessageBox.Ok, parent=self)
            msgbox.exec()
        self.update_eline_table()
        # Update the displayed estimated peak amplitude value 'le_peak_intensity'
        self._set_selected_eline(self._selected_eline)
        self._set_fit_status(False)

    def le_remove_rel_text_changed(self, text):
        self._update_le_remove_rel_state(text)

    def le_remove_rel_editing_finished(self):
        text = self.le_remove_rel.text()
        if self._validator_le_remove_rel.validate(text, 0)[0] == QDoubleValidator.Acceptable:
            self._remove_peak_threshold = float(text)
        else:
            self.le_remove_rel.setText(self._format_threshold(self._remove_peak_threshold))

    def pb_remove_unchecked_clicked(self):
        try:
            self.gpc.remove_unchecked_peaks()
        except Exception as ex:
            msg = str(ex)
            msgbox = QMessageBox(QMessageBox.Critical, "Error", msg, QMessageBox.Ok, parent=self)
            msgbox.exec()
        # Reload the table
        self.update_eline_table()
        # Update the displayed estimated peak amplitude value 'le_peak_intensity'
        self._set_selected_eline(self._selected_eline)
        self._set_fit_status(False)

    def _display_peak_intensity(self, eline):
        v = self.gpc.get_eline_intensity(eline)
        s = f"{v:.10g}" if v is not None else ""
        self.le_peak_intensity.setText(s)

    def _update_le_remove_rel_state(self, text=None):
        if text is None:
            text = self.le_remove_rel.text()
        state = self._validator_le_remove_rel.validate(text, 0)[0] == QDoubleValidator.Acceptable
        self.le_remove_rel.setValid(state)
        self.pb_remove_rel.setEnabled(state)

    @Slot(str)
    def slot_selection_item_changed(self, eline):
        self.element_selection.set_current_item(eline)

    @Slot(bool)
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

    def _format_threshold(self, value):
        return f"{value:.2f}"

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
            eline = self._selected_eline
            self.tbl_elines_set_selection(eline)
        self._update_add_remove_btn_state(eline)
        self._update_add_edit_userpeak_btn_state()
        self._update_add_edit_pileup_peak_btn_state()

    def _update_eline_selection_list(self):
        self._eline_list = self.gpc.get_full_eline_list()
        self.element_selection.set_item_list(self._eline_list)
        self.signal_update_element_selection_list.emit()

    @Slot()
    def update_eline_table(self):
        """Update table of emission lines without changing anything else"""
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
        add_peak = True
        if self.gpc.get_eline_name_category(self._selected_eline) == "userpeak":
            add_peak = False

        # Finally check if marker is set (you need it for adding peaks)
        _, marker_set = self.gpc.get_suggested_manual_peak_energy()

        if not marker_set and add_peak:
            enabled = False

        if add_peak:
            btn_text = "New User Peak ..."
        else:
            btn_text = "Edit User Peak ..."
        self.pb_user_peaks.setText(btn_text)
        self.pb_user_peaks.setEnabled(enabled)

    def _update_add_edit_pileup_peak_btn_state(self):

        enabled = True
        if self.gpc.get_eline_name_category(self._selected_eline) == "pileup":
            enabled = False

        # Finally check if marker is set (you need it for adding peaks)
        _, marker_set = self.gpc.get_suggested_manual_peak_energy()
        # Ignore set marker for userpeaks (marker is used to display location of userpeaks)
        if self.gpc.get_eline_name_category(self._selected_eline) == "userpeak":
            marker_set = False

        if not marker_set:
            enabled = False

        self.pb_pileup_peaks.setEnabled(enabled)

    def _set_selected_eline(self, eline):
        self._update_add_remove_btn_state(eline)
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

    def _set_fit_status(self, status):
        self.gui_vars["gui_state"]["state_model_fit_exists"] = status
        self.signal_parameters_changed.emit()
