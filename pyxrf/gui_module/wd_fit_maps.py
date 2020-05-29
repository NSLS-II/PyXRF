import os
import json
import yaml
import textwrap

from PyQt5.QtWidgets import (QPushButton, QHBoxLayout, QVBoxLayout,
                             QGroupBox, QLineEdit, QCheckBox, QLabel,
                             QComboBox, QListWidget, QListWidgetItem,
                             QDialog, QDialogButtonBox, QFileDialog,
                             QRadioButton, QButtonGroup, QGridLayout,
                             QTextEdit, QTableWidget, QTableWidgetItem,
                             QHeaderView, QWidget, QSpinBox, QScrollArea,
                             QTabWidget, QSizePolicy, QFrame)
from PyQt5.QtGui import QWindow, QBrush, QColor
from PyQt5.QtCore import Qt

from .useful_widgets import (LineEditReadOnly, global_gui_parameters, global_gui_variables,
                             get_background_css, TextEditReadOnly)
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
        self._setup_quantitative_analysis()

        vbox = QVBoxLayout()
        vbox.addWidget(self.group_settings)
        vbox.addSpacing(v_spacing)

        vbox.addWidget(self.pb_start_map_fitting)
        vbox.addWidget(self.pb_compute_roi_maps)
        vbox.addSpacing(v_spacing)

        vbox.addWidget(self.group_save_results)

        vbox.addWidget(self.group_quant_analysis)

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

    def _setup_quantitative_analysis(self):
        self.group_quant_analysis = QGroupBox("Quantitative Analysis")

        self.pb_load_quant_calib = QPushButton("Load Quantitative Calibration")
        self.pb_load_quant_calib.clicked.connect(self.pb_load_quant_calib_clicked)

        vbox = QVBoxLayout()
        vbox.addWidget(self.pb_load_quant_calib)
        self.group_quant_analysis.setLayout(vbox)

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

    def pb_load_quant_calib_clicked(self, event):
        if not self.ref_main_window.wnd_load_quantitative_calibration.isVisible():
            self.ref_main_window.wnd_load_quantitative_calibration.show()
        self.ref_main_window.wnd_load_quantitative_calibration.activateWindow()


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

        # The list of columns that stretch with the table
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


# Two sets of quantitative calibration data for demonstration of GUI layout
#   Real data will be used when GUI is integrated with the program.
#   The data are representing contents of JSON files, so they should be loaded
#   using 'json' module.
json_quant_calib_1 = \
    """
    {
        "name": "Micromatter 41147",
        "serial": "v41147",
        "description": "GaP 21.2 (Ga=15.4, P=5.8) / CaF2 14.6 / V 26.4 / Mn 17.5 / Co 21.7 / Cu 20.3",
        "element_lines": {
            "Ga_K": {
                "density": 15.4,
                "fluorescence": 6.12047035678267e-05
            },
            "Ga_L": {
                "density": 15.4,
                "fluorescence": 1.1429814846741588e-05
            },
            "P_K": {
                "density": 5.8,
                "fluorescence": 3.177988019213722e-05
            },
            "F_K": {
                "density": 7.105532786885246,
                "fluorescence": 1.8688801284649113e-07
            },
            "Ca_K": {
                "density": 7.494467213114753,
                "fluorescence": 0.0005815345261894806
            },
            "V_K": {
                "density": 26.4,
                "fluorescence": 0.00030309931019669974
            },
            "Mn_K": {
                "density": 17.5,
                "fluorescence": 0.0018328847495676865
            },
            "Co_K": {
                "density": 21.7,
                "fluorescence": 0.0014660067400157218
            },
            "Cu_K": {
                "density": 20.3,
                "fluorescence": 6.435121428993609e-05
            }
        },    
        "incident_energy": 12.0,
        "detector_channel": "sum",
        "scaler_name": "i0",
        "distance_to_sample": 1.0,
        "creation_time_local": "2020-05-27T18:49:14+00:00",
        "source_scan_id": null,
        "source_scan_uid": null
    }
    """

json_quant_calib_2 = \
    """
    {
        "name": "Micromatter 41164 Name Is Long So It Has To Be Printed On Multiple Lines (Some More Words To Make The Name Longer)",
        "serial": "41164",
        "description": "CeF3 21.1 / Au 20.6",
        "element_lines": {
            "F_K": {
                "density": 6.101050068482728,
                "fluorescence": 2.1573457185882552e-07
            },
            "Ce_L": {
                "density": 14.998949931517274,
                "fluorescence": 0.0014368335445700924
            },
            "Au_L": {
                "density": 20.6,
                "fluorescence": 4.4655757003090785e-05
            },
            "Au_M": {
                "density": 20.6,
                "fluorescence": 3.611978659032483e-05
            }
        },
        "incident_energy": 12.0,
        "detector_channel": "sum",
        "scaler_name": "i0",
        "distance_to_sample": 2.0,
        "creation_time_local": "2020-05-27T18:49:53+00:00",
        "source_scan_id": null,
        "source_scan_uid": null
    }
    """

# The data is structured the same way as in the actual program code, so transitioning
#   to real data will be simple
quant_calib = [
    [
        json.loads(json_quant_calib_1),
        {
            "file_path": "/path/to/quantitative/calibration/file/standard_41147.json"
        }
    ],
    [
        json.loads(json_quant_calib_2),
        {
            "file_path": "/extremely/long/path/to"
                         "/quantitative/calibration/file/so/it/had/to/be/"
                         "printed/on/multiple/lines/standard_41164.json"
        }
    ]
]
# The following list is to demonstrate how 'View' button works. Data is treated
#   differently in the actual code, but the resulting format will be similar.
quant_calib_json = [
    yaml.dump(quant_calib[0][0], default_flow_style=False, sort_keys=False, indent=4),
    yaml.dump(quant_calib[1][0], default_flow_style=False, sort_keys=False, indent=4),
]

class WndLoadQuantitativeCalibration(QWidget):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initialize()

    def initialize(self):
        self.setWindowTitle("PyXRF: Load Quantitative Calibration")
        self.setMinimumWidth(750)
        self.setMinimumHeight(400)
        self.resize(750, 600)

        self.pb_load_calib = QPushButton("Load Calibration ...")
        self.pb_load_calib.clicked.connect(self.pb_load_calib_clicked)

        self.cb_auto_update = QCheckBox("Auto")
        self.pb_update_plots = QPushButton("Update Plots")

        self.grp_current_scan = QGroupBox("Parameters of Currently Processed Scan")
        self.le_distance_to_sample = QLineEdit()
        hbox = QHBoxLayout()
        hbox.addWidget(QLabel("Distance-to-sample:"))
        hbox.addWidget(self.le_distance_to_sample)
        hbox.addStretch(1)
        self.grp_current_scan.setLayout(hbox)

        self._setup_tab_widget()
        #self.scroll_area = QScrollArea()

        vbox = QVBoxLayout()

        hbox = QHBoxLayout()
        hbox.addWidget(self.pb_load_calib)
        hbox.addStretch(1)
        hbox.addWidget(self.cb_auto_update)
        hbox.addWidget(self.pb_update_plots)
        vbox.addLayout(hbox)

        vbox.addWidget(self.tab_widget)

        vbox.addWidget(self.grp_current_scan)

        self.setLayout(vbox)

        # Display data
        self.display_loaded_standards()

    def _setup_tab_widget(self):

        self.tab_widget = QTabWidget()
        self.loaded_standards = QWidget()
        # self.display_loaded_standards()
        self.scroll = QScrollArea()
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setWidget(self.loaded_standards)
        self.tab_widget.addTab(self.scroll, "Loaded Standards")
        self.table = QTableWidget()
        self.tab_widget.addTab(self.table, "Selected Emission Lines")

    def display_loaded_standards(self):
        calib_data = quant_calib

        # Create the new widget (this deletes the old widget)
        self.loaded_standards = QWidget()
        self.loaded_standards.setMinimumWidth(700)

        # Also delete references to all components
        self.frames_calib_data = []
        self.pbs_view = []
        self.pbs_remove = []

        # All 'View' buttons are added to the group in order to be connected to the same slot
        self.group_view = QButtonGroup()
        self.group_view.setExclusive(False)
        self.group_view.buttonClicked.connect(self.pb_view_clicked)
        # The same for the 'Remove' buttons
        self.group_remove = QButtonGroup()
        self.group_remove.setExclusive(False)
        self.group_remove.buttonClicked.connect(self.pb_remove_clicked)

        vbox = QVBoxLayout()

        for cdata in calib_data:
            frame = QFrame()
            frame.setFrameStyle(QFrame.StyledPanel)
            frame.setStyleSheet(get_background_css((200, 255, 200), widget="QFrame"))

            _vbox = QVBoxLayout()

            name = cdata[0]['name']  # Standard name (can be arbitrary string
            # If name is long, then print it in a separate line
            _name_is_long = len(name) > 30

            pb_view = QPushButton("View")
            self.group_view.addButton(pb_view)
            pb_remove = QPushButton("Remove")
            self.group_remove.addButton(pb_remove)

            # Row 1: serial, name
            serial = cdata[0]['serial']
            _hbox = QHBoxLayout()
            _hbox.addWidget(QLabel(f"<b>Standard</b> #{serial}"))
            if not _name_is_long:
                _hbox.addWidget(QLabel(f"'{name}'"))
            _hbox.addStretch(1)
            _hbox.addWidget(pb_view)
            _hbox.addWidget(pb_remove)
            _vbox.addLayout(_hbox)

            # Optional row
            if _name_is_long:
                # Wrap name if it is extemely long
                name = textwrap.fill(name, width=80)
                _hbox = QHBoxLayout()
                _hbox.addWidget(QLabel("<b>Name:</b> "), 0, Qt.AlignTop)
                _hbox.addWidget(QLabel(name), 0, Qt.AlignTop)
                _hbox.addStretch(1)
                _vbox.addLayout(_hbox)

            # Row 2: description
            description = textwrap.fill(cdata[0]['description'], width=80)
            _hbox = QHBoxLayout()
            _hbox.addWidget(QLabel("<b>Description:</b>"), 0, Qt.AlignTop)
            _hbox.addWidget(QLabel(f"{description}"), 0, Qt.AlignTop)
            _hbox.addStretch(1)
            _vbox.addLayout(_hbox)

            # Row 3:
            incident_energy = cdata[0]['incident_energy']
            scaler = cdata[0]['scaler_name']
            detector_channel = cdata[0]['detector_channel']
            distance_to_sample = cdata[0]['distance_to_sample']
            _hbox = QHBoxLayout()
            _hbox.addWidget(QLabel(f"<b>Incident energy, keV:</b> {incident_energy}"))
            _hbox.addWidget(QLabel(f"  <b>Scaler:</b> {scaler}"))
            _hbox.addWidget(QLabel(f"  <b>Detector channel:</b> {detector_channel}"))
            _hbox.addWidget(QLabel(f"  <b>Distance-to-sample:</b> {distance_to_sample}"))
            _hbox.addStretch(1)
            _vbox.addLayout(_hbox)

            # Row 4: file name
            fln = textwrap.fill(cdata[1]['file_path'], width=80)
            _hbox = QHBoxLayout()
            _hbox.addWidget(QLabel("<b>Source file:</b>"), 0, Qt.AlignTop)
            _hbox.addWidget(QLabel(fln), 0, Qt.AlignTop)
            _hbox.addStretch(1)
            _vbox.addLayout(_hbox)

            frame.setLayout(_vbox)

            # Now the group box is added to the upper level layout
            vbox.addWidget(frame)
            vbox.addSpacing(5)
            self.frames_calib_data.append(frame)
            self.pbs_view.append(pb_view)
            self.pbs_remove.append(pb_remove)

        # Add the layout to the widget
        self.loaded_standards.setLayout(vbox)
        # ... and put the widget inside the scroll area. This will update the
        # contents of the scroll area.
        self.scroll.setWidget(self.loaded_standards)

    def pb_load_calib_clicked(self):
        # TODO: Propagate current directory here and use it in the dialog call
        current_dir = os.path.expanduser("~")
        file_name = QFileDialog.getOpenFileName(self, "Select File with Quantitative Calibration Data",
                                                current_dir,
                                                "JSON (*.json);; All (*)")
        file_name = file_name[0]
        if file_name:
            print(f"Loading quantitative calibration from file: {file_name}")

    def pb_view_clicked(self, button):
        try:
            n_standard = self.pbs_view.index(button)
            dlg = DialogFindElements(None,
                                     file_path=quant_calib[n_standard][1]['file_path'],
                                     calibration_data=quant_calib_json[n_standard])
            dlg.exec()
        except ValueError:
            print("'View' button was pressed, but not found in the list")

    def pb_remove_clicked(self, button):
        try:
            n_standard = self.pbs_remove.index(button)
            print(f'Removing standard #{n_standard}. The feature is not implemented yet ...')
        except ValueError:
            print("'Remove' button was pressed, but not found in the list")


class DialogFindElements(QDialog):

    def __init__(self, parent=None, *, file_path="", calibration_data=""):

        super().__init__(parent)

        self.setWindowTitle("View Calibration Standard")

        self.setMinimumSize(300, 400)
        self.resize(700, 700)

        # Displayed data (must be set before the dialog is shown
        self.file_path = file_path
        self.calibration_data = calibration_data

        vbox = QVBoxLayout()
        hbox = QHBoxLayout()
        hbox.addWidget(QLabel("<b>Source file:</b> "), 0, Qt.AlignTop)
        file_path = textwrap.fill(file_path, width=80)
        hbox.addWidget(QLabel(file_path), 0, Qt.AlignTop)
        hbox.addStretch(1)
        vbox.addLayout(hbox)

        te = QTextEdit()
        te.setReadOnly(True)
        te.setText(calibration_data)
        vbox.addWidget(te)

        button_box = QDialogButtonBox(QDialogButtonBox.Close)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        vbox.addWidget(button_box)

        self.setLayout(vbox)

