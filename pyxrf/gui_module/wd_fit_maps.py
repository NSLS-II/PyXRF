import os

from PyQt5.QtWidgets import (QPushButton, QHBoxLayout, QVBoxLayout,
                             QGroupBox, QLineEdit, QCheckBox, QLabel,
                             QComboBox, QListWidget, QListWidgetItem,
                             QDialog, QDialogButtonBox, QFileDialog,
                             QRadioButton, QButtonGroup, QGridLayout,
                             QTextEdit, QTableWidget, QTableWidgetItem,
                             QHeaderView, QWidget, QSpinBox)
from PyQt5.QtGui import QWindow, QBrush, QColor
from PyQt5.QtCore import Qt

from .useful_widgets import LineEditReadOnly, global_gui_parameters, global_gui_variables
from .form_base_widget import FormBaseWidget


class FitMapsWidget(FormBaseWidget):

    def __init__(self):
        super().__init__()
        self.initialize()

    def initialize(self):

        v_spacing = global_gui_parameters["vertical_spacing_in_tabs"]

        # Reference to the main window. The main window will hold
        #   references to all non-modal windows that could be opened
        #   from multiple places in the program.
        self.ref_main_window = global_gui_variables["ref_main_window"]

        self._setup_settings()
        self._setup_start_fitting()
        self._setup_compute_roi_maps()
        self._setup_save_results()

        vbox = QVBoxLayout()
        vbox.addWidget(self.group_settings)
        vbox.addSpacing(v_spacing)

        vbox.addWidget(self.pb_start_map_fitting)
        vbox.addWidget(self.pb_compute_roi_maps)
        vbox.addSpacing(v_spacing)

        vbox.addWidget(self.group_save_results)
        self.setLayout(vbox)

    def _setup_settings(self):
        self.group_settings = QGroupBox("Options")

        self.cb_save_plots = QCheckBox("Save spectra for pixels in the selected region")
        self.le_start_row = QLineEdit("0")
        self.le_start_col = QLineEdit("0")
        self.le_end_row = QLineEdit("0")
        self.le_end_col = QLineEdit("0")
        self.cb_interpolate_with_x_y = QCheckBox("Interpolate with (x,y) coordinates")

        self.group_save_plots = QGroupBox("Save spectra for pixels in the selected region")
        self.group_save_plots.setCheckable(True)
        self.group_save_plots.setChecked(False)

        vbox = QVBoxLayout()
        grid = QGridLayout()
        grid.addWidget(QLabel("Start row:"), 0, 0)
        grid.addWidget(self.le_start_row, 0, 1)
        grid.addWidget(QLabel("column:"), 0, 2)
        grid.addWidget(self.le_start_col, 0, 3)
        grid.addWidget(QLabel("End row(*):"), 1, 0)
        grid.addWidget(self.le_end_row, 1, 1)
        grid.addWidget(QLabel("column:"), 1, 2)
        grid.addWidget(self.le_end_col, 1, 3)
        vbox.addLayout(grid)

        hbox = QHBoxLayout()
        hbox.addStretch(1)
        hbox.addWidget(QLabel("* Point is not included in the selection"))
        hbox.addStretch(1)
        vbox.addLayout(hbox)
        self.group_save_plots.setLayout(vbox)

        vbox = QVBoxLayout()
        vbox.addWidget(self.group_save_plots)
        vbox.addSpacing(5)
        vbox.addWidget(self.cb_interpolate_with_x_y)

        self.group_settings.setLayout(vbox)

    def _setup_start_fitting(self):
        self.pb_start_map_fitting = QPushButton("Start XRF Map Fitting")

    def _setup_compute_roi_maps(self):
        self.pb_compute_roi_maps = QPushButton("Compute XRF Maps Based on ROI ...")
        self.pb_compute_roi_maps.clicked.connect(self.pb_compute_roi_maps_clicked)

    def _setup_save_results(self):
        self.group_save_results = QGroupBox("Save Results")

        self.pb_save_to_db = QPushButton("Save to Database (Databroker)")
        self.pb_save_to_db.setEnabled(False)

        self.pb_save_q_calibration = QPushButton("Save Quantitative Calibration")
        self.pb_save_q_calibration.clicked.connect(self.pb_save_q_calibration_clicked)

        self.pb_save_to_tiff = QPushButton("Save to TIFF")
        self.pb_save_to_tiff.clicked.connect(self.pb_save_to_tiff_clicked)

        self.pb_save_to_txt = QPushButton("Save to TXT")
        self.pb_save_to_txt.clicked.connect(self.pb_save_to_txt_clicked)

        grid = QGridLayout()
        grid.addWidget(self.pb_save_to_db, 0, 0, 1, 2)
        grid.addWidget(self.pb_save_q_calibration, 1, 0, 1, 2)
        grid.addWidget(self.pb_save_to_tiff, 2, 0)
        grid.addWidget(self.pb_save_to_txt, 2, 1)

        self.group_save_results.setLayout(grid)

    def pb_compute_roi_maps_clicked(self, event):
        if not self.ref_main_window.wnd_compute_roi_maps.isVisible():
            self.ref_main_window.wnd_compute_roi_maps.show()
        self.ref_main_window.wnd_compute_roi_maps.activateWindow()

    def pb_save_q_calibration_clicked(self, event):
        # TODO: Propagate full path to the saved file here
        fln = "~/quant_calibration.json"
        fln = os.path.expanduser(fln)
        file_name = QFileDialog.getSaveFileName(self, "Select File to Save Quantitative Standard Calibration",
                                                fln,
                                                "JSON (*.json);; All (*)")
        file_name = file_name[0]
        if file_name:
            print(f"Saving quantitative standard calibration to file file: {file_name}")

    def pb_save_to_tiff_clicked(self, event):
        # TODO: Propagate current directory here and use it in the dialog call
        current_dir = os.path.expanduser("~")
        dir = QFileDialog.getExistingDirectory(
            self, "Select Directory to Save XRF Maps to TIFF Files", current_dir,
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks)
        if dir:
            print(f"XRF Maps are saved as TIFF files to directory {dir}")
        else:
            print(f"Saving of XRF Maps is cancelled")

    def pb_save_to_txt_clicked(self, event):
        # TODO: Propagate current directory here and use it in the dialog call
        current_dir = os.path.expanduser("~")
        dir = QFileDialog.getExistingDirectory(
            self, "Select Directory to Save XRF Maps to TXT Files", current_dir,
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks)
        if dir:
            print(f"XRF Maps are saved as TXT files to directory {dir}")
        else:
            print(f"Saving of XRF Maps is cancelled")


class WndComputeRoiMaps(QWidget):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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

    def _setup_header(self):
        self.pb_clear = QPushButton("Clear")
        self.pb_use_lines_for_fitting = QPushButton("Use Lines Selected For Fitting")
        self.le_sel_emission_lines = QLineEdit()

        sample_elements = "Ar_K, Ca_K, Ti_K, Fe_K"
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
        self.tbl_labels = ["Line", "E, eV", "Min", "Max", "Show", "Reset"]

        # The list of columns with fixed size
        self.tbl_cols_stretch = ("E, eV", "Min", "Max")

        # Table item representation if different from default
        self.tbl_format = {"E, eV": ".0f"}

        # Editable items (highlighted with lighter background)
        self.tbl_cols_editable = {"Min", "Max"}

        # Columns that contain spinbox
        self.tbl_cols_spinbox = ("Min", "Max")

        self.table = QTableWidget()
        self.table.setColumnCount(len(self.tbl_labels))
        self.table.setHorizontalHeaderLabels(self.tbl_labels)
        self.table.verticalHeader().hide()
        self.table.setSelectionMode(QTableWidget.NoSelection)

        header = self.table.horizontalHeader()
        for n, lbl in enumerate(self.tbl_labels):
            # Set stretching for the columns
            if lbl in self.tbl_cols_stretch:
                header.setSectionResizeMode(n, QHeaderView.Stretch)
            else:
                header.setSectionResizeMode(n, QHeaderView.ResizeToContents)

        sample_content = [
            ["Ar_Ka1", 2957, 2745, 3169],
            ["Ca_Ka1", 3691, 3471, 3911],
            ["Ti_Ka1", 4510, 4278, 4742],
            ["Fe_Ka1", 6403, 6151, 6655],
        ]

        self.fill_table(sample_content)

    def fill_table(self, table_contents):

        self.table.setRowCount(len(table_contents))
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

                if self.tbl_labels[nc] not in self.tbl_cols_spinbox:
                    if self.tbl_labels[nc] in self.tbl_format:
                        fmt = self.tbl_format[self.tbl_labels[nc]]
                        s = ("{:" + fmt + "}").format(entry)
                    else:
                        s = f"{entry}"

                    item = QTableWidgetItem(s)
                    if nc > 0:
                        item.setTextAlignment(Qt.AlignCenter)
                    else:
                        item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)

                    # Set all columns not editable (unless needed)
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)

                    # Make all items not selectable (we are not using selections)
                    #item.setFlags(item.flags() & ~Qt.ItemIsSelectable)

                    # Note, that there is no way to set style sheet for QTableWidgetItem
                    item.setBackground(QBrush(QColor(*rgb_bckg)))

                    self.table.setItem(nr, nc, item)
                else:
                    item = QSpinBox()
                    # Set the range (in eV) large enough (there are total of 4096 10eV bins)
                    item.setRange(1, 40950)
                    item.setValue(entry)
                    item.setAlignment(Qt.AlignCenter)

                    color_css = f"rgb({rgb_bckg[0]}, {rgb_bckg[1]}, {rgb_bckg[2]})"
                    item.setStyleSheet(f"QSpinBox {{ background-color: {color_css}; }}")

                    self.table.setCellWidget(nr, nc, item)

            brightness = 220
            if nr % 2:
                rgb_bckg = (255, brightness, brightness)
            else:
                rgb_bckg = (brightness, 255, brightness)


            item = QWidget()
            cb = QCheckBox()
            item_hbox = QHBoxLayout(item)
            item_hbox.addWidget(cb)
            item_hbox.setAlignment(Qt.AlignCenter)
            item_hbox.setContentsMargins(0, 0, 0, 0)
            color_css = f"rgb({rgb_bckg[0]}, {rgb_bckg[1]}, {rgb_bckg[2]})"
            item.setStyleSheet(f"QWidget {{ background-color: {color_css}; }};; "
                               f"QCheckBox {{ background-color: white }}")
            self.table.setCellWidget(nr, nc + 1, item)

            item = QPushButton("Reset")
            rgb_bckg = [_ - 35 if (_ < 255) else _ for _ in rgb_bckg]
            color_css = f"rgb({rgb_bckg[0]}, {rgb_bckg[1]}, {rgb_bckg[2]})"
            item.setStyleSheet(f"QPushButton {{ background-color: {color_css}; }}")
            self.table.setCellWidget(nr, nc + 2, item)

    def _setup_footer(self):

        self.cb_subtract_background = QCheckBox("Subtract background")
        self.pb_compute_roi = QPushButton("Compute ROIs")

        hbox = QHBoxLayout()
        hbox.addWidget(self.cb_subtract_background)
        hbox.addStretch(1)
        hbox.addWidget(self.pb_compute_roi)

        return hbox
