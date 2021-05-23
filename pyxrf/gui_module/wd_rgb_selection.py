from qtpy.QtWidgets import QWidget, QHBoxLayout, QRadioButton, QButtonGroup, QTableWidget, QHeaderView, QSizePolicy
from qtpy.QtGui import QPalette
from qtpy.QtCore import Qt, Signal

import copy

from .useful_widgets import RangeManager, get_background_css, ComboBoxNamed

import logging

logger = logging.getLogger(__name__)


class RgbSelectionWidget(QWidget):

    signal_update_map_selections = Signal()

    def __init__(self):
        super().__init__()

        self._range_table = []
        self._limit_table = []
        self._rgb_keys = ["red", "green", "blue"]
        self._rgb_dict = {_: None for _ in self._rgb_keys}

        widget_layout = self._setup_rgb_widget()
        self.setLayout(widget_layout)

        sp = QSizePolicy()
        sp.setControlType(QSizePolicy.PushButton)
        sp.setHorizontalPolicy(QSizePolicy.Expanding)
        sp.setVerticalPolicy(QSizePolicy.Fixed)
        self.setSizePolicy(sp)

    def _setup_rgb_element(self, n_row, *, rb_check=0):
        """
        Parameters
        ----------
        rb_check: int
            The number of QRadioButton to check. Typically this would be the row number.
        """
        combo_elements = ComboBoxNamed(name=f"{n_row}")
        # combo_elements.setSizeAdjustPolicy(QComboBox.AdjustToContents)

        # Set text color for QComboBox widget (necessary if the program is used with Dark theme)
        pal = combo_elements.palette()
        pal.setColor(QPalette.ButtonText, Qt.black)
        combo_elements.setPalette(pal)
        # Set text color for drop-down view (necessary if the program is used with Dark theme)
        pal = combo_elements.view().palette()
        pal.setColor(QPalette.Text, Qt.black)
        combo_elements.view().setPalette(pal)

        btns = [QRadioButton(), QRadioButton(), QRadioButton()]
        if 0 <= rb_check < len(btns):
            btns[rb_check].setChecked(True)

        # Color is set for operation with Dark theme
        for btn in btns:
            pal = btn.palette()
            pal.setColor(QPalette.Text, Qt.black)
            btn.setPalette(pal)

        btn_group = QButtonGroup()
        for btn in btns:
            btn_group.addButton(btn)

        rng = RangeManager(name=f"{n_row}", add_sliders=True)
        rng.setTextColor([0, 0, 0])  # Set text color to 'black'
        # Set some text in edit boxes (just to demonstrate how the controls will look like)
        rng.le_min_value.setText("0.0")
        rng.le_max_value.setText("1.0")

        rng.setAlignment(Qt.AlignCenter)
        return combo_elements, btns, rng, btn_group

    def _enable_selection_events(self, enable):
        if enable:
            if not self.elements_btn_groups_events_enabled:
                for btn_group in self.elements_btn_groups:
                    btn_group.buttonToggled.connect(self.rb_toggled)
                for el_combo in self.elements_combo:
                    el_combo.currentIndexChanged.connect(self.combo_element_current_index_changed)
                for el_range in self.elements_range:
                    el_range.selection_changed.connect(self.range_selection_changed)
                self.elements_btn_groups_events_enabled = True
        else:
            if self.elements_btn_groups_events_enabled:
                for btn_group in self.elements_btn_groups:
                    btn_group.buttonToggled.disconnect(self.rb_toggled)
                for el_combo in self.elements_combo:
                    el_combo.currentIndexChanged.disconnect(self.combo_element_current_index_changed)
                # Disconnecting the Range Manager signals is not necessary, but let's do it for consistency
                for el_range in self.elements_range:
                    el_range.selection_changed.disconnect(self.range_selection_changed)
                self.elements_btn_groups_events_enabled = False

    def _setup_rgb_widget(self):

        self.elements_combo = []
        self.elements_rb_color = []
        self.elements_range = []
        self.elements_btn_groups = []
        self.elements_btn_groups_events_enabled = False
        self.row_colors = []

        self.table = QTableWidget()
        # Horizontal header entries
        tbl_labels = ["Element", "Red", "Green", "Blue", "Range"]
        # The list of columns that stretch with the table
        self.tbl_cols_stretch = ("Range",)

        self.table.setColumnCount(len(tbl_labels))
        self.table.setRowCount(3)
        self.table.setHorizontalHeaderLabels(tbl_labels)
        self.table.verticalHeader().hide()
        self.table.setSelectionMode(QTableWidget.NoSelection)

        header = self.table.horizontalHeader()
        for n, lbl in enumerate(tbl_labels):
            # Set stretching for the columns
            if lbl in self.tbl_cols_stretch:
                header.setSectionResizeMode(n, QHeaderView.Stretch)
            else:
                header.setSectionResizeMode(n, QHeaderView.ResizeToContents)

        vheader = self.table.verticalHeader()
        vheader.setSectionResizeMode(QHeaderView.Stretch)  # ResizeToContents)

        for n_row in range(3):
            combo_elements, btns, rng, btn_group = self._setup_rgb_element(n_row, rb_check=n_row)

            combo_elements.setMinimumWidth(180)
            self.table.setCellWidget(n_row, 0, combo_elements)
            for i, btn in enumerate(btns):
                item = QWidget()
                item_hbox = QHBoxLayout(item)
                item_hbox.addWidget(btn)
                item_hbox.setAlignment(Qt.AlignCenter)
                item_hbox.setContentsMargins(0, 0, 0, 0)
                item.setMinimumWidth(70)

                self.table.setCellWidget(n_row, i + 1, item)

            rng.setMinimumWidth(200)
            rng.setMaximumWidth(400)
            self.table.setCellWidget(n_row, 4, rng)

            self.elements_combo.append(combo_elements)
            self.elements_rb_color.append(btns)
            self.elements_range.append(rng)
            self.elements_btn_groups.append(btn_group)
            self.row_colors.append(self._rgb_keys[n_row])

        # Colors that are used to paint rows of the table in RGB colors
        br = 150
        self._rgb_row_colors = {"red": (255, br, br), "green": (br, 255, br), "blue": (br, br, 255)}
        self._rgb_color_keys = ["red", "green", "blue"]

        # Set initial colors
        for n_row in range(self.table.rowCount()):
            self._set_row_color(n_row)

        self._enable_selection_events(True)

        self.table.resizeRowsToContents()

        # Table height is computed based on content. It doesn't seem
        #   to account for the height of custom widgets, but the table
        #   looks good enough
        table_height = 0
        for n_row in range(self.table.rowCount()):
            table_height += self.table.rowHeight(n_row)
        self.table.setMaximumHeight(table_height)

        table_width = 650
        self.table.setMinimumWidth(table_width)
        self.table.setMaximumWidth(800)

        hbox = QHBoxLayout()
        hbox.addWidget(self.table)

        return hbox

    def combo_element_current_index_changed(self, name, index):
        if index < 0 or index >= len(self._range_table):
            return
        n_row = int(name)
        sel_eline = self._range_table[index][0]
        row_color = self.row_colors[n_row]
        self._rgb_dict[row_color] = sel_eline

        self.elements_range[n_row].set_range(self._range_table[index][1], self._range_table[index][2])
        self.elements_range[n_row].set_selection(
            value_low=self._limit_table[index][1], value_high=self._limit_table[index][2]
        )
        self._update_map_selections()

    def range_selection_changed(self, v_low, v_high, name):
        n_row = int(name)
        row_color = self.row_colors[n_row]
        sel_eline = self._rgb_dict[row_color]
        ind = None
        try:
            ind = [_[0] for _ in self._limit_table].index(sel_eline)
        except ValueError:
            pass
        if ind is not None:
            self._limit_table[ind][1] = v_low
            self._limit_table[ind][2] = v_high
            self._update_map_selections()
        # We are not preventing users to select the same emission line in to rows.
        #   Update the selected limits in other rows where the same element is selected.
        for nr, el_range in enumerate(self.elements_range):
            if (nr != n_row) and (self._rgb_dict[self.row_colors[nr]] == sel_eline):
                el_range.set_selection(value_low=v_low, value_high=v_high)

    def _get_selected_row_color(self, n_row):

        color_key = None

        btns = self.elements_rb_color[n_row]

        for n, btn in enumerate(btns):
            if btn.isChecked():
                color_key = self._rgb_color_keys[n]
                break

        return color_key

    def _set_row_color(self, n_row, *, color_key=None):
        """
        Parameters
        ----------
        n_row: int
            The row number that needs background color change (0..2 if table has 3 rows)
        color_key: int
            Color key: "red", "green" or "blue"
        """

        if color_key is None:
            color_key = self._get_selected_row_color(n_row)
        if color_key is None:
            return

        self.row_colors[n_row] = color_key
        rgb = self._rgb_row_colors[color_key]

        # The following code is based on the arrangement of the widgets in the table
        #   Modify the code if widgets are arranged differently or the table structure
        #   is changed
        for n_col in range(self.table.columnCount()):
            wd = self.table.cellWidget(n_row, n_col)
            if n_col == 0:
                # Combo box: update both QComboBox and QWidget backgrounds
                #   QWidget - background of the drop-down selection list
                css1 = get_background_css(rgb, widget="QComboBox", editable=False)
                css2 = get_background_css(rgb, widget="QWidget", editable=True)
                wd.setStyleSheet(css2 + css1)
            elif n_col <= 3:
                # 3 QRadioButton's. The buttons are inserted into QWidget objects,
                #   and we need to change backgrounds of QWidgets, not only buttons.
                wd.setStyleSheet(get_background_css(rgb, widget="QWidget", editable=False))
            elif n_col == 4:
                # Custom RangeManager widget, color is updated using custom method
                wd.setBackground(rgb)

        n_col = self._rgb_color_keys.index(color_key)
        for n, n_btn in enumerate(self.elements_rb_color[n_row]):
            check_status = True if n == n_col else False
            n_btn.setChecked(check_status)

    def _fill_table(self):
        self._enable_selection_events(False)

        eline_list = [_[0] for _ in self._range_table]
        for n_row in range(self.table.rowCount()):
            self.elements_combo[n_row].clear()
            self.elements_combo[n_row].addItems(eline_list)

        for n_row, color in enumerate(self._rgb_color_keys):
            # Initially set colors in order
            self._set_row_color(n_row, color_key=color)
            eline_key = self._rgb_dict[color]
            if eline_key is not None:
                try:
                    ind = eline_list.index(eline_key)
                    self.elements_combo[n_row].setCurrentIndex(ind)
                    range_low, range_high = self._range_table[ind][1:]
                    self.elements_range[n_row].set_range(range_low, range_high)
                    sel_low, sel_high = self._limit_table[ind][1:]
                    self.elements_range[n_row].set_selection(value_low=sel_low, value_high=sel_high)
                except ValueError:
                    pass
            else:
                self.elements_combo[n_row].setCurrentIndex(-1)  # Deselect all
                self.elements_range[n_row].set_range(0, 1)
                self.elements_range[n_row].set_selection(value_low=0, value_high=1)

        self._enable_selection_events(True)

    def _find_rbutton(self, button):
        for nr, btns in enumerate(self.elements_rb_color):
            for nc, btn in enumerate(btns):
                if btn == button:
                    # Return tuple (nr, nc)
                    return nr, nc
        # Return None if the button is not found (this shouldn't happen)
        return None

    def rb_toggled(self, button, state):
        if state:  # Ignore signals from unchecked buttons
            nr, nc = self._find_rbutton(button)

            color_current = self.row_colors[nr]
            color_to_set = self._rgb_color_keys[nc]
            nr_switch = self.row_colors.index(color_to_set)

            self._enable_selection_events(False)

            self._set_row_color(nr, color_key=color_to_set)
            self._set_row_color(nr_switch, color_key=color_current)
            # Swap selected maps
            tmp = self._rgb_dict[color_to_set]
            self._rgb_dict[color_to_set] = self._rgb_dict[color_current]
            self._rgb_dict[color_current] = tmp

            self._enable_selection_events(True)

            self._update_map_selections()

    def set_ranges_and_limits(self, *, range_table=None, limit_table=None, rgb_dict=None):
        if range_table is not None:
            self._range_table = copy.deepcopy(range_table)
        if limit_table is not None:
            self._limit_table = copy.deepcopy(limit_table)
        if rgb_dict is not None:
            self._rgb_dict = rgb_dict.copy()
        self._fill_table()

    def _update_map_selections(self):
        """Upload the selections (limit table) and update plot"""
        self.signal_update_map_selections.emit()
