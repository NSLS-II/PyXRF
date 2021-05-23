from copy import deepcopy
from qtpy.QtWidgets import (
    QVBoxLayout,
    QHBoxLayout,
    QCheckBox,
    QPushButton,
    QHeaderView,
    QTableWidget,
    QTableWidgetItem,
    QSizePolicy,
)
from qtpy.QtGui import QBrush, QColor, QPalette
from qtpy.QtCore import Qt, Signal, Slot

from .useful_widgets import RangeManager, SecondaryWindow, set_tooltip

import logging

logger = logging.getLogger(__name__)


class WndImageWizard(SecondaryWindow):

    signal_redraw_maps = Signal()

    def __init__(self, *, gpc, gui_vars):
        super().__init__()

        # The variable enables/disables plot updates. Setting it False prevents
        #   plot updates while updating the table or Select/Deselect All operation
        self._enable_plot_updates = False
        self._changes_exist = False

        # Global processing classes
        self.gpc = gpc
        # Global GUI variables (used for control of GUI state)
        self.gui_vars = gui_vars

        self.initialize()

    def initialize(self):
        self.setWindowTitle("PyXRF: Image Wizard")

        self.setMinimumHeight(400)
        self.setMinimumWidth(500)
        self.resize(600, 600)

        self.cb_select_all = QCheckBox("All")
        self.cb_select_all.stateChanged.connect(self.cb_select_all_state_changed)

        self._auto_update = True
        self.cb_auto_update = QCheckBox("Auto")
        self.cb_auto_update.setCheckState(Qt.Checked if self._auto_update else Qt.Unchecked)
        self.cb_auto_update.stateChanged.connect(self.cb_auto_update_state_changed)

        self.pb_update_plots = QPushButton("Update Plots")
        self.pb_update_plots.setEnabled(not self._auto_update)
        self.pb_update_plots.clicked.connect(self.pb_update_plots_clicked)

        self._setup_table()

        vbox = QVBoxLayout()

        hbox = QHBoxLayout()
        hbox.addWidget(self.cb_select_all)
        hbox.addStretch(1)
        hbox.addWidget(self.cb_auto_update)
        hbox.addWidget(self.pb_update_plots)
        vbox.addLayout(hbox)

        hbox = QHBoxLayout()
        hbox.addWidget(self.table)
        vbox.addLayout(hbox)

        self.setLayout(vbox)

        self._set_tooltips()

    def _setup_table(self):

        self.table = QTableWidget()
        self._range_data = []  # The variable keeps copy of the table data
        self._limit_data = []  # Copy of the table that holds selection limits
        self._show_data = []  # Second column - bool values that indicate if the map is shown

        self.tbl_labels = ["Element", "Plotted Range", "Minimum", "Maximum"]
        self.tbl_h_alignment = [Qt.AlignCenter, Qt.AlignCenter, Qt.AlignCenter, Qt.AlignCenter]
        self.tbl_section_resize_mode = [
            QHeaderView.Stretch,
            QHeaderView.ResizeToContents,
            QHeaderView.Stretch,
            QHeaderView.Stretch,
        ]
        self.table.setColumnCount(len(self.tbl_labels))
        self.table.verticalHeader().hide()
        self.table.setHorizontalHeaderLabels(self.tbl_labels)

        self.table.setSelectionMode(QTableWidget.NoSelection)
        self.table.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Minimum)

        header = self.table.horizontalHeader()
        for n in range(len(self.tbl_labels)):
            header.setSectionResizeMode(n, self.tbl_section_resize_mode[n])
            header.setDefaultAlignment(self.tbl_h_alignment[n])

        self._checkable_items = []  # The list of items in the 1st column
        self._range_items = []  # The list of RangeManager items

        self.fill_table(self._range_data, self._limit_data, self._show_data)
        self.table.itemChanged.connect(self.table_item_changed)

    def fill_table(self, range_table, limit_table, show_table):

        self._enable_plot_updates = False

        self._clear_table()
        # Copy the table (we want to keep a copy, it's small)
        self._range_data = deepcopy(range_table)
        self._limit_data = deepcopy(limit_table)
        self._show_data = deepcopy(show_table)

        self.table.setRowCount(len(range_table))
        # Color is set for operation with Dark theme
        pal = self.table.palette()
        pal.setColor(QPalette.Text, Qt.black)
        pal.setColor(QPalette.Base, Qt.white)
        self.table.setPalette(pal)

        brightness = 200
        table_colors = [(255, brightness, brightness), (brightness, 255, brightness)]

        for nr, row in enumerate(range_table):
            element, v_min, v_max = row[0], row[1], row[2]
            sel_min, sel_max = self._limit_data[nr][1], self._limit_data[nr][2]
            sel_show = self._show_data[nr][1]
            rgb = table_colors[nr % 2]

            for nc in range(self.table.columnCount()):
                if nc in (0, 2, 3):

                    if nc == 0:
                        item = QTableWidgetItem(element)
                        item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
                        item.setCheckState(Qt.Checked if sel_show else Qt.Unchecked)
                        self._checkable_items.append(item)
                    elif nc == 2:
                        item = QTableWidgetItem(f"{self._format_table_range_value(v_min)}")
                        item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                    elif nc == 3:
                        item = QTableWidgetItem(f"{self._format_table_range_value(v_max)}")
                        item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)

                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)

                    item.setBackground(QBrush(QColor(*rgb)))
                    self.table.setItem(nr, nc, item)

                elif nc == 1:

                    item = RangeManager(name=f"{nr}", add_sliders=True)
                    item.set_range(v_min, v_max)
                    item.set_selection(value_low=sel_min, value_high=sel_max)
                    item.setAlignment(Qt.AlignCenter)
                    item.setFixedWidth(400)
                    item.setBackground(rgb)
                    item.setTextColor([0, 0, 0])  # Color is set for operation with Dark theme
                    item.selection_changed.connect(self.table_range_item_changed)
                    self._range_items.append(item)
                    self.table.setCellWidget(nr, nc, item)

        self.table.resizeColumnsToContents()
        self.table.resizeRowsToContents()

        table_width = 0
        for n_col in range(self.table.columnCount()):
            table_width += self.table.columnWidth(n_col)
        self.table.setFixedWidth(table_width + 150)
        self.setFixedWidth(table_width + 170)

        self._enable_plot_updates = True

    def update_table_ranges(self, range_table, limit_table):
        """Update ranges and selections only. Don't update 'show' status."""
        self._enable_plot_updates = False

        self._range_data = deepcopy(range_table)
        self._limit_data = deepcopy(limit_table)

        for n in range(len(self._range_items)):
            v_min, v_max = self._range_data[n][1], self._range_data[n][2]
            sel_min, sel_max = self._limit_data[n][1], self._limit_data[n][2]
            rng = self._range_items[n]
            rng.set_range(v_min, v_max)
            rng.set_selection(value_low=sel_min, value_high=sel_max)

            self.table.item(n, 2).setText(f"{self._format_table_range_value(v_min)}")
            self.table.item(n, 3).setText(f"{self._format_table_range_value(v_max)}")

        self._enable_plot_updates = True

    def _set_tooltips(self):
        set_tooltip(self.cb_select_all, "<b>Select/Deselect All</b> emission lines in the list")
        set_tooltip(
            self.cb_auto_update,
            "Automatically <b>update the plots</b> when changes are made. "
            "If unchecked, then button <b>Update Plots</b> must be pressed "
            "to update the plots. Automatic update is often undesirable "
            "when large maps are displayed and multiple changes to parameters "
            "are made.",
        )
        set_tooltip(self.pb_update_plots, "<b>Update plots</b> based on currently selected parameters.")
        set_tooltip(
            self.table,
            "Choose <b>emission lines</b> from the currently selected dataset and "
            "select the <b>range of intensity</b> for each emission line",
        )

    def update_widget_state(self, condition=None):
        # Update the state of the menu bar
        state = not self.gui_vars["gui_state"]["running_computations"]
        self.setEnabled(state)

        # Hide the window if required by the program state
        state_xrf_map_exists = self.gui_vars["gui_state"]["state_xrf_map_exists"]
        if not state_xrf_map_exists:
            self.hide()

        if condition == "tooltips":
            self._set_tooltips()

    def cb_select_all_state_changed(self, state):
        self._select_all_items(state)

    def cb_auto_update_state_changed(self, state):
        self._auto_update = state
        self.pb_update_plots.setEnabled(not state)
        # If changes were made, apply the changes while switching to 'auto' mode
        if state and self._changes_exist:
            self._update_map_selections_auto()

    def pb_update_plots_clicked(self):
        """Upload the selections (limit table) and update plot"""
        self._update_map_selections()

    def table_item_changed(self, item):
        try:
            n_row = self._checkable_items.index(item)
            state = item.checkState() == Qt.Checked
            if state != self._show_data[n_row][1]:
                self._show_data[n_row][1] = state
                logger.debug(
                    f"Image wizard: map {self._show_data[n_row][0]} was {'checked' if state else 'unchecked'}"
                )
                self._update_map_selections_auto()
        except ValueError:
            pass

    def table_range_item_changed(self, low, high, name):
        n_row = int(name)
        self._limit_data[n_row][1] = low
        self._limit_data[n_row][2] = high
        logger.debug(
            f"Image Wizard: range changed for the map '{self._limit_data[n_row][0]}'. "
            f"New range: ({low}, {high})"
        )
        self._update_map_selections_auto()

    @Slot()
    def slot_update_table(self):
        """Reload table including ranges and selections for the emission lines"""
        range_table, limit_table, show_table = self.gpc.get_maps_info_table()
        self.fill_table(range_table, limit_table, show_table)

    @Slot()
    def slot_update_ranges(self):
        """Update only ranges and selections for the emission lines"""
        range_table, limit_table, _ = self.gpc.get_maps_info_table()
        self.update_table_ranges(range_table, limit_table)

    def _format_table_range_value(self, value):
        return f"{value:.12g}"

    def _clear_table(self):
        # Disconnect all signals
        for range_item in self._range_items:
            range_item.selection_changed.disconnect(self.table_range_item_changed)
        # Delete all rows
        self.table.clearContents()
        self._checkable_items = []  # The list of items in the 1st column
        self._range_items = []

    def _select_all_items(self, check_state):
        """Select/deselect all items in the table"""
        self._enable_plot_updates = False
        for n_row in range(self.table.rowCount()):
            item = self.table.item(n_row, 0)
            item.setCheckState(check_state)
        self._enable_plot_updates = True
        self._update_map_selections_auto()

    def _update_map_selections_auto(self):
        """Update maps only if 'auto' update is ON. Used as a 'filter'
        to prevent extra plot updates."""
        self._changes_exist = True
        if self._auto_update:
            self._update_map_selections()

    def _update_map_selections(self):
        """Upload the selections (limit table) and update plot"""
        if self._enable_plot_updates:
            self._changes_exist = False
            self.gpc.set_maps_limit_table(self._limit_data, self._show_data)
            self._redraw_maps()

    def _redraw_maps(self):
        logger.debug("Redrawing XRF Maps")
        self.gpc.redraw_maps()
        self.signal_redraw_maps.emit()
