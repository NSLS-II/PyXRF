from PyQt5.QtWidgets import (QWidget, QLabel, QVBoxLayout, QHBoxLayout, QRadioButton, QButtonGroup,
                             QComboBox, QCheckBox, QTableWidget, QHeaderView, QSizePolicy, QSpacerItem)
from PyQt5.QtCore import Qt

from .useful_widgets import RangeManager, get_background_css


class PlotRgbMaps(QWidget):

    def __init__(self):
        super().__init__()

        self.combo_select_dataset = QComboBox()
        sample_datasets = ["scan2D_28844_amk_fit", "scan2D_28844_amk_roi",
                           "scan2D_28844_amk_scaler", "positions"]
        # datasets = ["Select Dataset ..."] + sample_datasets
        datasets = sample_datasets
        self.combo_select_dataset.addItems(datasets)

        self.combo_normalization = QComboBox()
        sample_scalers = ["i0", "i0_time", "time", "time_diff"]
        scalers = ["Normalize by ..."] + sample_scalers
        self.combo_normalization.addItems(scalers)

        self.cb_interpolate = QCheckBox("Interpolate")

        self.combo_pixels_positions = QComboBox()
        self.combo_pixels_positions.addItems(["Pixels", "Positions"])

        # The label will be replaced with the widget that will actually plot the data
        label = QLabel()
        comment = \
            "The widget will plot up to 3 XRF maps in RGB representation.\n"\
            "The layout was similar to 'Element Map' tab of the original PyXRF"
        label.setText(comment)
        label.setStyleSheet("QLabel { background-color : white; color : blue; }")
        label.setAlignment(Qt.AlignCenter)

        rgb_selection = RgbSelectionWidget()

        vbox = QVBoxLayout()
        hbox = QHBoxLayout()
        hbox.addWidget(self.combo_select_dataset)
        hbox.addWidget(self.combo_normalization)
        hbox.addStretch(1)
        hbox.addWidget(self.cb_interpolate)
        hbox.addWidget(self.combo_pixels_positions)
        vbox.addLayout(hbox)
        vbox.addWidget(label)
        hbox = QHBoxLayout()
        hbox.addSpacerItem(QSpacerItem(0, 0))
        hbox.addWidget(rgb_selection)
        hbox.addSpacerItem(QSpacerItem(0, 0))
        vbox.addLayout(hbox)
        self.setLayout(vbox)


class RgbSelectionWidget(QWidget):

    def __init__(self):
        super().__init__()

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
        sample_elements = ["Ar_K", "Ca_K", "Ti_K", "Fe_K", "Userpeak1",
                           "i0", "i0_time", "time", "time_diff"]
        elements = [""] + sample_elements

        combo_elements = QComboBox()
        combo_elements.addItems(elements)

        btns = [QRadioButton(), QRadioButton(), QRadioButton()]
        if 0 <= rb_check < len(btns):
            btns[rb_check].setChecked(True)

        btn_group = QButtonGroup()
        for btn in btns:
            btn_group.addButton(btn)

        btn_group.buttonToggled.connect(self.rb_toggled)

        rng = RangeManager(add_sliders=True)
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

        # Colors that are used to paint rows of the table in RGB colors
        br = 150
        self.rgb_row_colors = ((255, br, br),
                               (br, 255, br),
                               (br, br, 255))

        # Set initial colors
        for n_row in range(self.table.rowCount()):
            self.adjust_row_color(n_row)

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

        n_rgb_color = None

        btns = self.elements_rb_color[n_row]

        for n, btn in enumerate(btns):
            if btn.isChecked():
                n_rgb_color = n
                break

        return n_rgb_color

    def adjust_row_color(self, n_row, *, n_rgb_color=None):
        """
        Parameters
        ----------
        n_row: int
            The row number that needs background color change (0..2 if table has 3 rows)
        n_rgb_color: int
            The number color in the RGB table (must have value 0..2)
        """

        if n_rgb_color is None:
            n_rgb_color = self._get_selected_row_color(n_row)
        if n_rgb_color is None:
            return

        rgb = self.rgb_row_colors[n_rgb_color]

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
            self.adjust_row_color(nr, n_rgb_color=nc)
