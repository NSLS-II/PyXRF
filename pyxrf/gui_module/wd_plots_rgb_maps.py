from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QRadioButton, QButtonGroup,
                             QComboBox, QCheckBox, QTableWidget, QHeaderView, QSizePolicy, QSpacerItem)
from PyQt5.QtGui import QPalette
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot

import copy

from .useful_widgets import RangeManager, get_background_css, set_tooltip

from matplotlib.backends.backend_qt5agg import \
    FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar


class PlotRgbMaps(QWidget):

    signal_rgb_maps_dataset_selection_changed = pyqtSignal()
    signal_rgb_maps_norm_changed = pyqtSignal()

    def __init__(self, *, gpc, gui_vars):
        super().__init__()

        # Global processing classes
        self.gpc = gpc
        # Global GUI variables (used for control of GUI state)
        self.gui_vars = gui_vars

        self.combo_select_dataset = QComboBox()
        self.combo_select_dataset.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.combo_select_dataset.currentIndexChanged.connect(
            self.combo_select_dataset_current_index_changed)

        self.combo_normalization = QComboBox()
        self.combo_normalization.currentIndexChanged.connect(
            self.combo_normalization_current_index_changed)

        self.cb_interpolate = QCheckBox("Interpolate")
        self.cb_interpolate.setChecked(self.gpc.get_rgb_maps_grid_interpolate())
        self.cb_interpolate.toggled.connect(self.cb_interpolate_toggled)

        self.combo_pixels_positions = QComboBox()
        self._pix_pos_values = ["Pixels", "Positions"]
        self.combo_pixels_positions.addItems(self._pix_pos_values)
        self.combo_pixels_positions.setCurrentIndex(
            self._pix_pos_values.index(self.gpc.get_maps_pixel_or_pos()))
        self.combo_pixels_positions.currentIndexChanged.connect(
            self.combo_pixels_positions_current_index_changed)

        self.mpl_canvas = FigureCanvas(self.gpc.img_model_rgb.fig)
        self.mpl_toolbar = NavigationToolbar(self.mpl_canvas, self)

        self.rgb_selection = RgbSelectionWidget()
        self.slot_update_dataset_info()

        vbox = QVBoxLayout()
        hbox = QHBoxLayout()
        hbox.addWidget(self.combo_select_dataset)
        hbox.addWidget(self.combo_normalization)
        hbox.addStretch(1)
        hbox.addWidget(self.cb_interpolate)
        hbox.addWidget(self.combo_pixels_positions)
        vbox.addLayout(hbox)
        vbox.addWidget(self.mpl_toolbar)
        vbox.addWidget(self.mpl_canvas)
        hbox = QHBoxLayout()
        hbox.addSpacerItem(QSpacerItem(0, 0))
        hbox.addWidget(self.rgb_selection)
        hbox.addSpacerItem(QSpacerItem(0, 0))
        vbox.addLayout(hbox)
        self.setLayout(vbox)

        self._set_tooltips()

    def _set_tooltips(self):
        set_tooltip(self.combo_select_dataset,
                    "Select <b>dataset</b>.")
        set_tooltip(self.combo_normalization,
                    "Select <b>scaler</b> for normalization of displayed XRF maps.")
        set_tooltip(self.cb_interpolate,
                    "Interpolate coordinates to <b>uniform grid</b>.")
        set_tooltip(self.combo_pixels_positions,
                    "Switch axes units between <b>pixels</b> and <b>positional units</b>.")
        set_tooltip(self.rgb_selection,
                    "Select XRF Maps displayed in <b>Red</b>, <b>Green</b> and "
                    "<b>Blue</b> colors and adjust the range of <b>intensity</b> for each "
                    "displayed map.")

    def update_widget_state(self, condition=None):
        if condition == "tooltips":
            self._set_tooltips()

    def combo_select_dataset_current_index_changed(self, index):
        self.gpc.set_rgb_maps_selected_dataset(index + 1)
        self.signal_rgb_maps_dataset_selection_changed.emit()

    def combo_normalization_current_index_changed(self, index):
        self.gpc.set_rgb_maps_scaler_index(index)
        self.signal_rgb_maps_norm_changed.emit()

    def combo_pixels_positions_current_index_changed(self, index):
        self.gpc.set_rgb_maps_pixel_or_pos(self._pix_pos_values[index])

    def cb_interpolate_toggled(self, state):
        self.gpc.set_rgb_maps_grid_interpolate(state)

    @pyqtSlot()
    def slot_update_dataset_info(self):
        self._update_datasets()
        self._update_scalers()

        # Update ranges in the RGB selection widget
        range_table, limit_table, rgb_dict = self.gpc.get_rgb_maps_info_table()
        self.rgb_selection.set_ranges_and_limits(range_table=range_table,
                                                 limit_table=limit_table,
                                                 rgb_dict=rgb_dict)

    @pyqtSlot()
    def slot_update_ranges(self):
        """Update only ranges and selections for the emission lines"""
        range_table, limit_table, _ = self.gpc.get_rgb_maps_info_table()
        self.rgb_selection.set_ranges_and_limits(range_table=range_table, limit_table=limit_table)

    def _update_datasets(self):
        dataset_list, dset_sel = self.gpc.get_rgb_maps_dataset_list()
        self._dataset_list = dataset_list.copy()
        self.combo_select_dataset.clear()
        self.combo_select_dataset.addItems(self._dataset_list)
        # No item should be selected if 'dset_sel' is 0
        self.combo_select_dataset.setCurrentIndex(dset_sel - 1)

    def _update_scalers(self):
        scalers, scaler_sel = self.gpc.get_rgb_maps_scaler_list()
        self._scaler_list = ["Normalize by ..."] + scalers
        self.combo_normalization.clear()
        self.combo_normalization.addItems(self._scaler_list)
        self.combo_normalization.setCurrentIndex(scaler_sel)


class RgbSelectionWidget(QWidget):

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

    def _setup_rgb_element(self, *, rb_check=0):
        """
        Parameters
        ----------
        rb_check: int
            The number of QRadioButton to check. Typically this would be the row number.
        """
        combo_elements = QComboBox()
        combo_elements.setSizeAdjustPolicy(QComboBox.AdjustToContents)

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

        btn_group.buttonToggled.connect(self.rb_toggled)

        rng = RangeManager(add_sliders=True)
        rng.setTextColor([0, 0, 0])  # Set text color to 'black'
        # Set some text in edit boxes (just to demonstrate how the controls will look like)
        rng.le_min_value.setText("0.0")
        rng.le_max_value.setText("1.0")

        rng.setAlignment(Qt.AlignCenter)
        return combo_elements, btns, rng, btn_group

    def _setup_rgb_widget(self):

        self.elements_combo = []
        self.elements_rb_color = []
        self.elements_range = []
        self.elements_btn_groups = []
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
            combo_elements, btns, rng, btn_group = self._setup_rgb_element(rb_check=n_row)

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
        self._rgb_row_colors = {"red": (255, br, br),
                                "green": (br, 255, br),
                                "blue": (br, br, 255)}
        self._rgb_color_keys = ["red", "green", "blue"]

        # Set initial colors
        for n_row in range(self.table.rowCount()):
            self._set_row_color(n_row)

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
                wd.setStyleSheet(
                    get_background_css(rgb, widget="QWidget", editable=False))
            elif n_col == 4:
                # Custom RangeManager widget, color is updated using custom method
                wd.setBackground(rgb)

        n_col = self._rgb_color_keys.index(color_key)
        for n, n_btn in enumerate(self.elements_rb_color[n_row]):
            check_status = True if n == n_col else False
            n_btn.setChecked(check_status)

    def _fill_table(self):

        eline_list = [_[0] for _ in self._range_table]
        for n_row in range(self.table.rowCount()):
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

    def _update_ranges(self):
        pass

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

            self._set_row_color(nr, color_key=color_to_set)
            self._set_row_color(nr_switch, color_key=color_current)

    def set_ranges_and_limits(self, *, range_table=None, limit_table=None, rgb_dict=None):
        if range_table is not None:
            self._range_table = copy.deepcopy(range_table)
        if limit_table is not None:
            self._limit_table = copy.deepcopy(limit_table)
        if rgb_dict is not None:
            self._rgb_dict = rgb_dict.copy()
        self._fill_table()
