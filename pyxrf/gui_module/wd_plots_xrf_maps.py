from PyQt5.QtWidgets import (QWidget, QLabel, QVBoxLayout, QHBoxLayout, QComboBox, QCheckBox,
                             QPushButton, QHeaderView, QTableWidget, QTableWidgetItem,
                             QSizePolicy)
from PyQt5.QtGui import QBrush, QColor
from PyQt5.QtCore import Qt

from .useful_widgets import global_gui_variables, RangeManager, SecondaryWindow, set_tooltip


class PlotXrfMaps(QWidget):

    def __init__(self):
        super().__init__()

        # Reference to the main window. The main window will hold
        #   references to all non-modal windows that could be opened
        #   from multiple places in the program.
        self.ref_main_window = global_gui_variables["ref_main_window"]

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

        self.pb_image_wizard = QPushButton("Image Wizard ...")
        self.pb_image_wizard.clicked.connect(self.pb_image_wizard_clicked)

        # self.pb_quant_settings = QPushButton("Quantitative ...")

        self.cb_interpolate = QCheckBox("Interpolate")

        self.cb_scatter_plot = QCheckBox("Scatter plot")
        self.cb_scatter_plot.toggled.connect(self.cb_scatter_plot_toggled)

        self.cb_quantitative = QCheckBox("Quantitative")

        self.combo_color_scheme = QComboBox()
        # TODO: make color schemes global
        color_schemes = ("viridis", "jet", "bone", "gray", "oranges", "hot")
        self.combo_color_scheme.addItems(color_schemes)

        self.combo_linear_log = QComboBox()
        self.combo_linear_log.addItems(["Linear", "Log"])

        self.combo_pixels_positions = QComboBox()
        self.combo_pixels_positions.addItems(["Pixels", "Positions"])

        # The label will be replaced with the widget that will actually plot the data
        label = QLabel()
        comment = \
            "The widget will plot XRF maps with layout similar to 'Element Map' tab\n"\
            "of the original PyXRF"
        label.setText(comment)
        label.setStyleSheet("QLabel { background-color : white; color : blue; }")
        label.setAlignment(Qt.AlignCenter)

        vbox = QVBoxLayout()
        hbox = QHBoxLayout()
        hbox.addWidget(self.combo_select_dataset)
        hbox.addStretch(4)
        hbox.addWidget(self.pb_image_wizard)
        hbox.addStretch(1)
        hbox.addWidget(self.cb_interpolate)
        hbox.addWidget(self.cb_scatter_plot)
        vbox.addLayout(hbox)
        hbox = QHBoxLayout()
        hbox.addWidget(self.combo_normalization)
        hbox.addStretch(3)
        hbox.addWidget(self.cb_quantitative)
        hbox.addStretch(1)
        hbox.addWidget(self.combo_linear_log)
        hbox.addWidget(self.combo_pixels_positions)
        hbox.addWidget(self.combo_color_scheme)
        vbox.addLayout(hbox)
        vbox.addWidget(label)
        self.setLayout(vbox)

        self._set_tooltips()

    def _set_tooltips(self):
        set_tooltip(self.combo_select_dataset,
                    "Select <b>dataset</b>.")
        set_tooltip(self.combo_normalization,
                    "Select <b>scaler</b> for normalization of displayed XRF maps.")
        set_tooltip(self.pb_image_wizard,
                    "Open the window with tools for <b>selection and configuration</b> "
                    "the displayed XRF maps.")
        set_tooltip(self.cb_interpolate,
                    "Interpolate coordinates to <b>uniform grid</b>.")
        set_tooltip(self.cb_scatter_plot,
                    "Display <b>scatter plot</b>.")
        set_tooltip(self.cb_quantitative,
                    "Normalize the displayed XRF maps using loaded "
                    "<b>Quantitative Calibration</b> data.")
        set_tooltip(self.combo_color_scheme,
                    "Select <b>color scheme</b>")
        set_tooltip(self.combo_linear_log,
                    "Switch between <b>linear</b> and <b>logarithmic</b> scale "
                    "for plotting of XRF Map pixel <b>intensity</b>.")
        set_tooltip(self.combo_pixels_positions,
                    "Switch axes units between <b>pixels</b> and <b>positional units</b>.")

    def update_widget_state(self, condition=None):
        if condition == "tooltips":
            self._set_tooltips()

    def pb_image_wizard_clicked(self):
        # Position the window in relation ot the main window (only when called once)
        pos = self.ref_main_window.pos()
        self.ref_main_window.wnd_image_wizard.position_once(pos.x(), pos.y())

        if not self.ref_main_window.wnd_image_wizard.isVisible():
            self.ref_main_window.wnd_image_wizard.show()
        self.ref_main_window.wnd_image_wizard.activateWindow()

    def cb_scatter_plot_toggled(self, state):
        self.cb_interpolate.setVisible(not state)
        self.combo_pixels_positions.setVisible(not state)


class WndImageWizard(SecondaryWindow):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initialize()

    def initialize(self):
        self.setWindowTitle("PyXRF: Image Wizard")

        self.setMinimumHeight(400)
        self.setMinimumWidth(500)
        self.resize(600, 600)

        self.cb_all = QCheckBox("All")
        self.cb_auto_update = QCheckBox("Auto")
        self.pb_update_plot = QPushButton("Update Plots")

        self._setup_table()

        vbox = QVBoxLayout()

        hbox = QHBoxLayout()
        hbox.addWidget(self.cb_all)
        hbox.addStretch(1)
        hbox.addWidget(self.cb_auto_update)
        hbox.addWidget(self.pb_update_plot)
        vbox.addLayout(hbox)

        hbox = QHBoxLayout()
        hbox.addWidget(self.table)
        vbox.addLayout(hbox)

        self.setLayout(vbox)

    def _setup_table(self):

        self.table = QTableWidget()

        sample_content = [
            ["Ar_K", 4.277408, 307.452117],
            ["Ca_K", 0.000000, 1.750295e+03],
            ["Fe_K", 17.902211, 4.576803e+04],
            ["Tb_L", 0.000000, 2.785734e+03],
            ["Ti_K", 0.055362, 1.227637e+04],
            ["Zn_K", 0.000000, 892.008670],
            ["compton", 0.000000, 249.055352],
            ["elastic", 0.000000, 163.153881],
            ["i0", 1.715700e+04, 1.187463e+06],
            ["i0_time", 3.066255e+06, 1.727313e+08],
        ]

        self.tbl_labels = ["Element", "Plotted Range", "Minimum", "Maximum"]
        self.tbl_h_alignment = [Qt.AlignCenter, Qt.AlignCenter, Qt.AlignCenter, Qt.AlignCenter]
        self.tbl_section_resize_mode = [QHeaderView.Stretch, QHeaderView.ResizeToContents,
                                        QHeaderView.Stretch, QHeaderView.Stretch]
        self.table.setColumnCount(len(self.tbl_labels))
        self.table.verticalHeader().hide()
        self.table.setHorizontalHeaderLabels(self.tbl_labels)

        self.table.setSelectionMode(QTableWidget.NoSelection)
        self.table.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Minimum)

        header = self.table.horizontalHeader()
        for n in range(len(self.tbl_labels)):
            header.setSectionResizeMode(n, self.tbl_section_resize_mode[n])
            header.setDefaultAlignment(self.tbl_h_alignment[n])

        self.fill_table(sample_content)

    def _set_tooltips(self):
        pass

    def update_widget_state(self, condition=None):
        if condition == "tooltips":
            self._set_tooltips()

    def fill_table(self, table_contents):

        self.table.setRowCount(len(table_contents))

        brightness = 200
        table_colors = [(255, brightness, brightness), (brightness, 255, brightness)]

        for nr, row in enumerate(table_contents):
            element, v_min, v_max = row[0], row[1], row[2]
            rgb = table_colors[nr % 2]

            for nc in range(self.table.columnCount()):
                if nc in (0, 2, 3):

                    if nc == 0:
                        item = QTableWidgetItem(element)
                        item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
                        item.setCheckState(Qt.Unchecked)
                    elif nc == 2:
                        item = QTableWidgetItem(f"{v_min:.12g}")
                        item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                    elif nc == 3:
                        item = QTableWidgetItem(f"{v_max:.12g}")
                        item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)

                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)

                    item.setBackground(QBrush(QColor(*rgb)))
                    self.table.setItem(nr, nc, item)

                elif nc == 1:

                    item = RangeManager(add_sliders=True)
                    item.le_min_value.setText(f"{v_min:.12g}")
                    item.le_max_value.setText(f"{v_max:.12g}")
                    item.setAlignment(Qt.AlignCenter)
                    item.setFixedWidth(400)
                    item.setBackground(rgb)
                    self.table.setCellWidget(nr, nc, item)

        self.table.resizeColumnsToContents()
        self.table.resizeRowsToContents()

        table_width = 0
        for n_col in range(self.table.columnCount()):
            table_width += self.table.columnWidth(n_col)
        self.table.setFixedWidth(table_width + 150)
        self.setFixedWidth(table_width + 170)
