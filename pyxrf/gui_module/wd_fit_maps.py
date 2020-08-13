import os
import textwrap

from PyQt5.QtWidgets import (QPushButton, QHBoxLayout, QVBoxLayout, QGroupBox, QLineEdit,
                             QCheckBox, QLabel, QComboBox, QDialog, QDialogButtonBox, QFileDialog,
                             QRadioButton, QButtonGroup, QGridLayout, QTextEdit, QTableWidget,
                             QTableWidgetItem, QHeaderView, QWidget, QSpinBox, QScrollArea,
                             QTabWidget, QFrame, QMessageBox)
from PyQt5.QtGui import QBrush, QColor, QDoubleValidator
from PyQt5.QtCore import Qt, pyqtSlot, pyqtSignal

from .useful_widgets import (LineEditReadOnly, global_gui_parameters, get_background_css,
                             PushButtonMinimumWidth, SecondaryWindow, set_tooltip, LineEditExtended)
from .form_base_widget import FormBaseWidget

import logging
logger = logging.getLogger(__name__)


class FitMapsWidget(FormBaseWidget):

    # Signal that is sent (to main window) to update global state of the program
    update_global_state = pyqtSignal()

    signal_map_fitting_complete = pyqtSignal()
    signal_activate_tab_xrf_maps = pyqtSignal()

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

        v_spacing = global_gui_parameters["vertical_spacing_in_tabs"]

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

        self._set_tooltips()

        # Timer is currently used to simulate processing
        self._timer = None
        self._timer_counter = 0

    def _setup_settings(self):
        self.group_settings = QGroupBox("Options")

        self.cb_save_plots = QCheckBox("Save spectra for pixels in the selected region")
        self.le_start_row = QLineEdit("0")
        self.le_start_col = QLineEdit("0")

        self.le_end_row = QLineEdit("0")

        self.le_end_col = QLineEdit("0")

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

        self.group_settings.setLayout(vbox)

    def _setup_start_fitting(self):
        self.pb_start_map_fitting = QPushButton("Start XRF Map Fitting")
        self.pb_start_map_fitting.clicked.connect(self.pb_start_map_fitting_clicked)

    def _setup_compute_roi_maps(self):
        self.pb_compute_roi_maps = QPushButton("Compute XRF Maps Based on ROI ...")
        self.pb_compute_roi_maps.clicked.connect(self.pb_compute_roi_maps_clicked)

    def _setup_save_results(self):
        self.group_save_results = QGroupBox("Save Results")

        self.pb_save_to_db = QPushButton("Save to Database (Databroker) ...")
        self.pb_save_to_db.setEnabled(False)

        self.pb_save_q_calibration = QPushButton("Save Quantitative Calibration ...")
        self.pb_save_q_calibration.clicked.connect(self.pb_save_q_calibration_clicked)

        self.pb_export_to_tiff_and_txt = QPushButton("Export to TIFF and TXT ...")
        self.pb_export_to_tiff_and_txt.clicked.connect(self.pb_export_to_tiff_and_txt_clicked)

        grid = QGridLayout()
        grid.addWidget(self.pb_save_to_db, 0, 0, 1, 2)
        grid.addWidget(self.pb_save_q_calibration, 1, 0, 1, 2)
        grid.addWidget(self.pb_export_to_tiff_and_txt, 2, 0, 1, 2)

        self.group_save_results.setLayout(grid)

    def _setup_quantitative_analysis(self):
        self.group_quant_analysis = QGroupBox("Quantitative Analysis")

        self.pb_load_quant_calib = QPushButton("Load Quantitative Calibration ...")
        self.pb_load_quant_calib.clicked.connect(self.pb_load_quant_calib_clicked)

        vbox = QVBoxLayout()
        vbox.addWidget(self.pb_load_quant_calib)
        self.group_quant_analysis.setLayout(vbox)

    def _set_tooltips(self):
        set_tooltip(
            self.group_settings,
            "Raw spectra of individual pixels are saved for the selected region of the map."
            "The region is selected by specifying the <b>Start</b> and <b>End</b> coordinates "
            "in pixels. The coordinates are defined by numbers of row and column. 'End row' and "
            "'End column' are not included in the selection. The end row and column are "
            "<b>not included</b> in the selection. If only 'Start' coordinates are specified, "
            "then one spectrum for the pixel defined by 'Start row' and 'Start column' coordinates "
            "is saved")
        set_tooltip(
            self.le_start_row,
            "Number of the <b>first row</b> of the map to be included in the selection. "
            "The number must be less than the number entered into 'End row' box.")
        set_tooltip(
            self.le_start_col,
            "Number of the <b>first column</b> of the map to be included in the selection. "
            "The number must be less than the number entered into 'End column' box.")
        set_tooltip(
            self.le_end_row,
            "Number of the <b>row following the last row</b> included in the selection. "
            "The number must be greater than the number entered into 'Start row' box. "
            "The field may be left empty. If 'End row' and 'End column' are empty, then "
            "only one spectrum for the pixel with coordinates 'Start row' and 'Start column' "
            "is saved")
        set_tooltip(
            self.le_end_col,
            "Number of the <b>column following the last column</b> included in the selection. "
            "The number must be greater than the number entered into 'Start column' box."
            "The field may be left empty. If 'End row' and 'End column' are empty, then "
            "only one spectrum for the pixel with coordinates 'Start row' and 'Start column' "
            "is saved")

        set_tooltip(
            self.pb_start_map_fitting,
            "Click to start <b>fitting of the XRF Maps</b>. The generated XRF Maps can be viewed "
            "in <b>'XRF Maps' tab</b>")

        set_tooltip(
            self.pb_compute_roi_maps,
            "Opens the window for setting up <b>spectral ROIs</b> and computating XRF Maps "
            "based on the ROIs")

        set_tooltip(self.pb_save_to_db,
                    "Save generated XRF Maps to a <b>database</b> via Databroker")

        set_tooltip(
            self.pb_save_q_calibration,
            "Opens a Dialog Box which allows to preview and save <b>Quantitative Calibration data</b>")
        set_tooltip(
            self.pb_export_to_tiff_and_txt,
            "Open a Dialog box which allows to export XRF Maps as <b>TIFF</b> and <b>TXT</b> files")

        set_tooltip(
            self.pb_load_quant_calib,
            "Open a window with GUI tools for loading and managing previously saved "
            "<b>Quantitative Calibration data</b> used for processing (normalization) "
            "of XRF Maps. The loaded calibration data is applied to XRF Maps if 'Quantitative' "
            "box is checked in 'XRF Maps' tab")

    def update_widget_state(self, condition=None):
        if condition == "tooltips":
            self._set_tooltips()

        state_file_loaded = self.gui_vars["gui_state"]["state_file_loaded"]
        state_model_exist = self.gui_vars["gui_state"]["state_model_exists"]
        state_xrf_map_exists = self.gui_vars["gui_state"]["state_xrf_map_exists"]

        self.group_settings.setEnabled(state_file_loaded & state_model_exist)
        self.pb_start_map_fitting.setEnabled(state_file_loaded & state_model_exist)
        self.pb_compute_roi_maps.setEnabled(state_file_loaded & state_model_exist)
        self.group_save_results.setEnabled(state_xrf_map_exists)
        self.group_quant_analysis.setEnabled(state_xrf_map_exists)

    def pb_compute_roi_maps_clicked(self):
        # Position the window in relation ot the main window (only when called once)
        pos = self.ref_main_window.pos()
        self.ref_main_window.wnd_compute_roi_maps.position_once(pos.x(), pos.y())

        if not self.ref_main_window.wnd_compute_roi_maps.isVisible():
            self.ref_main_window.wnd_compute_roi_maps.show()
        self.ref_main_window.wnd_compute_roi_maps.activateWindow()

    def pb_save_q_calibration_clicked(self):
        # TODO: Propagate full path to the saved file here
        file_path = "~/quant_calibration.json"
        file_path = os.path.expanduser(file_path)

        dlg = DialogSaveCalibration(file_path=file_path)
        res = dlg.exec()
        if res:
            print(f"Saving quantitative calibration to the file '{dlg.file_path}'")

    def pb_export_to_tiff_and_txt_clicked(self):
        # TODO: Propagate full path to the saved file here
        dir_path = os.path.expanduser("~")

        dlg = DialogExportToTiffAndTxt(dir_path=dir_path)
        res = dlg.exec()
        if res:
            print(f"Saving TIFF and TXT files to the directory '{dlg.dir_path}'")

    def pb_load_quant_calib_clicked(self):
        # Position the window in relation ot the main window (only when called once)
        pos = self.ref_main_window.pos()
        self.ref_main_window.wnd_load_quantitative_calibration.position_once(pos.x(), pos.y())

        if not self.ref_main_window.wnd_load_quantitative_calibration.isVisible():
            self.ref_main_window.wnd_load_quantitative_calibration.show()
        self.ref_main_window.wnd_load_quantitative_calibration.activateWindow()

    def pb_start_map_fitting_clicked(self):

        # self.gui_vars["gui_state"]["running_computations"] = True
        # self.update_global_state.emit()

        success = False
        try:
            self.gpc.fit_individual_pixels()
            self.gui_vars["gui_state"]["state_xrf_map_exists"] = True
            success = True
        except Exception as ex:
            msg = str(ex)
            msgbox = QMessageBox(QMessageBox.Critical, "Error",
                                 msg, QMessageBox.Ok, parent=self)
            msgbox.exec()
        self.signal_map_fitting_complete.emit()
        self.update_global_state.emit()
        if success:
            self.signal_activate_tab_xrf_maps.emit()

        # if not self._timer:
        #    self._timer = QTimer()
        # self._timer.timeout.connect(self.timerExpired)
        # self._timer.setInterval(80)
        # self._timer_counter = 0
        # self._timer.start()

    @pyqtSlot()
    def timerExpired(self):
        self._timer_counter += 1
        progress_bar = self.ref_main_window.statusProgressBar
        progress_bar.setValue(self._timer_counter)
        if self._timer_counter >= 100:
            self._timer.stop()
            self._timer.timeout.disconnect(self.timerExpired)
            self._timer = None
            progress_bar.setValue(0)
            status_bar = self.ref_main_window.statusBar()
            status_bar.showMessage("XRF Maps are generated. "
                                   "Results are presented in 'XRF Maps' tab.", 5000)
            self.gui_vars["gui_state"]["running_computations"] = False
            self.update_global_state.emit()


class WndComputeRoiMaps(SecondaryWindow):

    def __init__(self, *, gpc, gui_vars):
        super().__init__()

        # Global processing classes
        self.gpc = gpc
        # Global GUI variables (used for control of GUI state)
        self.gui_vars = gui_vars

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

        self.cb_subtract_baseline = QCheckBox("Subtract baseline")
        self.pb_compute_roi = QPushButton("Compute ROIs")

        hbox = QHBoxLayout()
        hbox.addWidget(self.cb_subtract_baseline)
        hbox.addStretch(1)
        hbox.addWidget(self.pb_compute_roi)

        return hbox

    def _set_tooltips(self):
        set_tooltip(self.pb_clear,
                    "<b>Clear</b> the list")
        set_tooltip(self.pb_use_lines_for_fitting,
                    "Copy the contents of <b>the list of emission lines selected for "
                    "fitting</b> to the list of ROIs")
        set_tooltip(self.le_sel_emission_lines,
                    "The list of <b>emission lines</b> selected for ROI computation.")

        set_tooltip(self.table, "The list of ROIs")

        set_tooltip(self.cb_subtract_baseline,
                    "<b>Subtract baseline</b> from the pixel spectra before computing ROIs. "
                    "Subtracting baseline slows down computations and usually have no benefit. "
                    "In most cases it should remain <b>unchecked</b>.")
        set_tooltip(self.pb_compute_roi,
                    "<b>Run</b> computations of the ROIs. The resulting <b>ROI</b> dataset "
                    "may be viewed in <b>XRF Maps</b> tab.")

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


class WndLoadQuantitativeCalibration(SecondaryWindow):

    signal_quantitative_calibration_changed = pyqtSignal()

    def __init__(self, *, gpc, gui_vars):
        super().__init__()

        # Global processing classes
        self.gpc = gpc
        # Global GUI variables (used for control of GUI state)
        self.gui_vars = gui_vars

        self.initialize()

    def initialize(self):
        self.table_header_display_names = False

        self.setWindowTitle("PyXRF: Load Quantitative Calibration")
        self.setMinimumWidth(750)
        self.setMinimumHeight(400)
        self.resize(750, 600)

        self.pb_load_calib = QPushButton("Load Calibration ...")
        self.pb_load_calib.clicked.connect(self.pb_load_calib_clicked)

        self._changes_exist = False
        self._auto_update = True
        self.cb_auto_update = QCheckBox("Auto")
        self.cb_auto_update.setCheckState(Qt.Checked if self._auto_update else Qt.Unchecked)
        self.cb_auto_update.stateChanged.connect(self.cb_auto_update_state_changed)

        self.pb_update_plots = QPushButton("Update Plots")
        self.pb_update_plots.clicked.connect(self.pb_update_plots_clicked)

        self.grp_current_scan = QGroupBox("Parameters of Currently Processed Scan")

        self._distance_to_sample = 0.0
        self.le_distance_to_sample = LineEditExtended()
        le_dist_validator = QDoubleValidator()
        le_dist_validator.setBottom(0)
        self.le_distance_to_sample.setValidator(le_dist_validator)
        self._set_distance_to_sample()
        self.le_distance_to_sample.editingFinished.connect(self.le_distance_to_sample_editing_finished)
        self.le_distance_to_sample.focusOut.connect(self.le_distance_to_sample_focus_out)

        hbox = QHBoxLayout()
        hbox.addWidget(QLabel("Distance-to-sample:"))
        hbox.addWidget(self.le_distance_to_sample)
        hbox.addStretch(1)
        self.grp_current_scan.setLayout(hbox)

        self.eline_rb_exclusive = []  # Holds the list of groups of exclusive radio buttons
        self._setup_tab_widget()

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
        self.update_all_data()

        self._set_tooltips()

    def _setup_tab_widget(self):

        self.tab_widget = QTabWidget()
        self.loaded_standards = QWidget()
        # self.display_loaded_standards()
        self.scroll = QScrollArea()
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setWidget(self.loaded_standards)
        self.tab_widget.addTab(self.scroll, "Loaded Standards")

        self.combo_set_table_header = QComboBox()
        self.combo_set_table_header.addItems(["Standard Serial #", "Standard Name"])
        self.combo_set_table_header.currentIndexChanged.connect(
            self.combo_set_table_header_index_changed)

        vbox = QVBoxLayout()
        vbox.addSpacing(5)
        hbox = QHBoxLayout()
        hbox.addWidget(QLabel("Display in table header:"))
        hbox.addWidget(self.combo_set_table_header)
        hbox.addStretch(1)
        vbox.addLayout(hbox)
        self.table = QTableWidget()
        self.table.verticalHeader().hide()
        self.table.setSelectionMode(QTableWidget.NoSelection)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setMinimumSectionSize(150)
        vbox.addWidget(self.table)

        frame = QFrame()
        vbox.setContentsMargins(0, 0, 0, 0)
        frame.setLayout(vbox)

        self.tab_widget.addTab(frame, "Selected Emission Lines")

    def display_loaded_standards(self):
        calib_data = self.gpc.get_quant_calibration_data()
        calib_settings = self.gpc.get_quant_calibration_settings()

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

        for cdata, csettings in zip(calib_data, calib_settings):
            frame = QFrame()
            frame.setFrameStyle(QFrame.StyledPanel)
            frame.setStyleSheet(get_background_css((200, 255, 200), widget="QFrame"))

            _vbox = QVBoxLayout()

            name = cdata['name']  # Standard name (can be arbitrary string
            # If name is long, then print it in a separate line
            _name_is_long = len(name) > 30

            pb_view = QPushButton("View ...")
            self.group_view.addButton(pb_view)
            pb_remove = QPushButton("Remove")
            self.group_remove.addButton(pb_remove)

            # Row 1: serial, name
            serial = cdata['serial']
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
            description = textwrap.fill(cdata['description'], width=80)
            _hbox = QHBoxLayout()
            _hbox.addWidget(QLabel("<b>Description:</b>"), 0, Qt.AlignTop)
            _hbox.addWidget(QLabel(f"{description}"), 0, Qt.AlignTop)
            _hbox.addStretch(1)
            _vbox.addLayout(_hbox)

            # Row 3:
            incident_energy = cdata['incident_energy']
            scaler = cdata['scaler_name']
            detector_channel = cdata['detector_channel']
            distance_to_sample = cdata['distance_to_sample']
            _hbox = QHBoxLayout()
            _hbox.addWidget(QLabel(f"<b>Incident energy, keV:</b> {incident_energy}"))
            _hbox.addWidget(QLabel(f"  <b>Scaler:</b> {scaler}"))
            _hbox.addWidget(QLabel(f"  <b>Detector channel:</b> {detector_channel}"))
            _hbox.addWidget(QLabel(f"  <b>Distance-to-sample:</b> {distance_to_sample}"))
            _hbox.addStretch(1)
            _vbox.addLayout(_hbox)

            # Row 4: file name
            fln = textwrap.fill(csettings['file_path'], width=80)
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

    def display_table_header(self):
        calib_data = self.gpc.get_quant_calibration_data()
        header_by_name = self.table_header_display_names

        tbl_labels = ["Lines"]
        for n, cdata in enumerate(calib_data):
            if header_by_name:
                txt = cdata['name']
            else:
                txt = cdata['serial']
            txt = textwrap.fill(txt, width=20)
            tbl_labels.append(txt)

        self.table.setHorizontalHeaderLabels(tbl_labels)

    def display_standard_selection_table(self):
        calib_data = self.gpc.get_quant_calibration_data()
        self._quant_file_paths = self.gpc.get_quant_calibration_file_path_list()

        brightness = 220
        table_colors = [(255, brightness, brightness), (brightness, 255, brightness)]

        # Disconnect all radio button signals before clearing the table
        for bgroup in self.eline_rb_exclusive:
            bgroup.buttonToggled.disconnect(self.rb_selection_toggled)

        # This list will hold radio button groups for horizontal rows
        #   Those are exclusive groups. They are not going to be
        #   used directly, but they must be kept alive in order
        #   for the radiobuttons to work properly. Most of the groups
        #   will contain only 1 radiobutton, which will always remain checked.
        self.eline_rb_exclusive = []
        # The following list will contain the list of radio buttons for each
        #   row. If there is no radiobutton in a position, then the element is
        #   set to None.
        # N rows: the number of emission lines, N cols: the number of standards
        self.eline_rb_lists = []

        self.table.clear()

        if not calib_data:
            self.table.setRowCount(0)
            self.table.setColumnCount(0)
        else:
            # Create the sorted list of available element lines
            line_set = set()
            for cdata in calib_data:
                ks = list(cdata['element_lines'].keys())
                line_set.update(list(ks))
            self.eline_list = list(line_set)
            self.eline_list.sort()

            for n in range(len(self.eline_list)):
                self.eline_rb_exclusive.append(QButtonGroup())
                self.eline_rb_lists.append([None] * len(calib_data))

            self.table.setColumnCount(len(calib_data) + 1)
            self.table.setRowCount(len(self.eline_list))
            self.display_table_header()

            for n, eline in enumerate(self.eline_list):

                rgb = table_colors[n % 2]

                item = QTableWidgetItem(eline)
                item.setTextAlignment(Qt.AlignCenter)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                item.setBackground(QBrush(QColor(*rgb)))
                self.table.setItem(n, 0, item)

                for ns, cdata in enumerate(calib_data):
                    q_file_path = self._quant_file_paths[ns]  # Used to identify standard
                    if eline in cdata['element_lines']:
                        rb = QRadioButton()
                        if self.gpc.get_quant_calibration_is_eline_selected(eline, q_file_path):
                            rb.setChecked(True)

                        self.eline_rb_lists[n][ns] = rb
                        # self.eline_rb_by_standard[ns].addButton(rb)
                        self.eline_rb_exclusive[n].addButton(rb)

                        item = QWidget()
                        item_hbox = QHBoxLayout(item)
                        item_hbox.addWidget(rb)
                        item_hbox.setAlignment(Qt.AlignCenter)
                        item_hbox.setContentsMargins(0, 0, 0, 0)

                        item.setStyleSheet(get_background_css(rgb))

                        # Generate tooltip
                        density = cdata['element_lines'][eline]['density']
                        fluorescence = cdata['element_lines'][eline]['fluorescence']
                        ttip = (f"Fluorescence (F): {fluorescence:12g}\n"
                                f"Density (D): {density:12g}\n")
                        # Avoid very small values of density (probably zero)
                        if abs(density) > 1e-30:
                            ttip += f"F/D: {fluorescence/density:12g}"

                        item.setToolTip(ttip)

                        self.table.setCellWidget(n, ns + 1, item)
                    else:
                        # There is no radio button, but we still need to fill the cell
                        item = QTableWidgetItem("")
                        item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                        item.setBackground(QBrush(QColor(*rgb)))
                        self.table.setItem(n, ns + 1, item)

            # Now the table is set (specifically radio buttons).
            # So we can connect the button groups with the event processing function
            for bgroup in self.eline_rb_exclusive:
                bgroup.buttonToggled.connect(self.rb_selection_toggled)

    @pyqtSlot()
    def update_all_data(self):
        self.display_loaded_standards()
        self.display_standard_selection_table()
        self._set_distance_to_sample()

    def _set_distance_to_sample(self):
        """Set 'le_distance_to_sample` without updating maps"""
        distance_to_sample = self.gpc.get_quant_calibration_distance_to_sample()
        if distance_to_sample is None:
            distance_to_sample = 0.0
        self._distance_to_sample = distance_to_sample
        self._set_le_distance_to_sample(distance_to_sample)

    def _set_tooltips(self):
        set_tooltip(self.pb_load_calib, "Load <b>calibration data</b> from JSON file.")
        set_tooltip(self.cb_auto_update, "Automatically <b>update the plots</b> when changes are made. "
                                         "If unchecked, then button <b>Update Plots</b> must be pressed "
                                         "to update the plots. Automatic update is often undesirable "
                                         "when large maps are displayed and multiple changes to parameters "
                                         "are made.")
        set_tooltip(self.pb_update_plots,
                    "<b>Update plots</b> based on currently selected parameters.")
        set_tooltip(self.le_distance_to_sample,
                    "Distance between <b>the sample and the detector</b>. The ratio between of the distances "
                    "during calibration and measurement is used to scale computed concentrations. "
                    "If distance-to-sample is 0 for calibration or measurement, then no scaling is performed.")
        set_tooltip(self.combo_set_table_header,
                    "Use <b>Serial Number</b> or <b>Name</b> of the calibration standard "
                    "in the header of the table")
        set_tooltip(self.table,
                    "Use Radio Buttons to select the <b>source of calibration data</b> for each emission line. "
                    "This feature is needed if multiple loaded calibration files have data on the same "
                    "emission line.")

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

    def cb_auto_update_state_changed(self, state):
        self._auto_update = state
        self.pb_update_plots.setEnabled(not state)
        # If changes were made, apply the changes while switching to 'auto' mode
        if state and self._changes_exist:
            self._update_maps_auto()

    def pb_update_plots_clicked(self):
        self._update_maps()

    def pb_load_calib_clicked(self):
        current_dir = self.gpc.get_current_working_directory()
        file_name = QFileDialog.getOpenFileName(self, "Select File with Quantitative Calibration Data",
                                                current_dir,
                                                "JSON (*.json);; All (*)")
        file_name = file_name[0]
        if file_name:
            logger.debug(f"Loading quantitative calibration from file: '{file_name}'")
            self.gpc.load_quantitative_calibration_data(file_name)
            self.update_all_data()
            self._update_maps_auto()

    def pb_view_clicked(self, button):
        try:
            n_standard = self.pbs_view.index(button)
            calib_settings = self.gpc.get_quant_calibration_settings()
            file_path = calib_settings[n_standard]['file_path']
            calib_preview = self.gpc.get_quant_calibration_text_preview(file_path)
            dlg = DialogViewCalibStandard(None,
                                          file_path=file_path,
                                          calib_preview=calib_preview)
            dlg.exec()
        except ValueError:
            logger.error("'View' button was pressed, but not found in the list of buttons")

    def pb_remove_clicked(self, button):
        try:
            n_standard = self.pbs_remove.index(button)
            calib_settings = self.gpc.get_quant_calibration_settings()
            file_path = calib_settings[n_standard]['file_path']
            self.gpc.quant_calibration_remove_entry(file_path)
            self.update_all_data()
            self._update_maps_auto()
        except ValueError:
            logger.error("'Remove' button was pressed, but not found in the list of buttons")

    def rb_selection_toggled(self, button, checked):
        if checked:
            # Find the button in 2D list 'self.eline_rb_lists'
            button_found = False
            for nr, rb_list in enumerate(self.eline_rb_lists):
                try:
                    nc = rb_list.index(button)
                    button_found = True
                    break
                except ValueError:
                    pass

            if button_found:
                eline = self.eline_list[nr]
                n_standard = nc
                file_path = self._quant_file_paths[n_standard]
                self.gpc.set_quant_calibration_select_eline(eline, file_path)
                self._update_maps_auto()
            else:
                # This should never happen
                logger.error("Selection radio button was pressed, but not found in the list")

    def combo_set_table_header_index_changed(self, index):
        self.table_header_display_names = bool(index)
        self.display_table_header()

    def le_distance_to_sample_editing_finished(self):
        distance_to_sample = float(self.le_distance_to_sample.text())
        if distance_to_sample != self._distance_to_sample:
            self._distance_to_sample = distance_to_sample
            self.gpc.set_quant_calibration_distance_to_sample(distance_to_sample)
            self._update_maps_auto()

    def le_distance_to_sample_focus_out(self):
        try:
            float(self.le_distance_to_sample.text())
        except ValueError:
            # If the text can not be interpreted to float, then replace the text with the old value
            self._set_le_distance_to_sample(self._distance_to_sample)

    def _set_le_distance_to_sample(self, distance_to_sample):
        self.le_distance_to_sample.setText(f"{distance_to_sample:.12g}")

    def _update_maps_auto(self):
        """Update maps only if 'auto' update is ON. Used as a 'filter'
        to prevent extra plot updates."""
        self._changes_exist = True
        if self._auto_update:
            self._update_maps()

    def _update_maps(self):
        """Upload the selections (limit table) and update plot"""
        self._changes_exist = False
        self._redraw_maps()
        # Emit signal only after the maps are redrawn. This should change
        #   ranges in the respective controls for the plots
        self.signal_quantitative_calibration_changed.emit()

    def _redraw_maps(self):
        # We don't emit any signals here, but we don't really need to.
        logger.debug("Redrawing RGB XRF Maps")
        self.gpc.compute_map_ranges()
        self.gpc.redraw_maps()
        self.gpc.compute_rgb_map_ranges()
        self.gpc.redraw_rgb_maps()


class DialogViewCalibStandard(QDialog):

    def __init__(self, parent=None, *, file_path="", calib_preview=""):

        super().__init__(parent)

        self.setWindowTitle("View Calibration Standard")

        self.setMinimumSize(300, 400)
        self.resize(700, 700)

        # Displayed data (must be set before the dialog is shown
        self.file_path = file_path
        self.calib_preview = calib_preview

        vbox = QVBoxLayout()
        hbox = QHBoxLayout()
        hbox.addWidget(QLabel("<b>Source file:</b> "), 0, Qt.AlignTop)
        file_path = textwrap.fill(self.file_path, width=80)
        hbox.addWidget(QLabel(file_path), 0, Qt.AlignTop)
        hbox.addStretch(1)
        vbox.addLayout(hbox)

        te = QTextEdit()
        te.setReadOnly(True)
        te.setText(self.calib_preview)
        vbox.addWidget(te)

        button_box = QDialogButtonBox(QDialogButtonBox.Close)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        vbox.addWidget(button_box)

        self.setLayout(vbox)


class DialogSaveCalibration(QDialog):

    def __init__(self, parent=None, *, file_path=""):

        super().__init__(parent)

        self.__file_path = ""

        self.setWindowTitle("Save Quantitative Calibration")
        self.setMinimumHeight(600)
        self.setMinimumWidth(600)
        self.resize(600, 600)

        self.text_edit = QTextEdit()
        set_tooltip(self.text_edit,
                    "Preview the <b>quantitative calibration data</b> to be saved. The displayed "
                    "warnings will not be saved, but need to be addressed in order to keep "
                    "data integrity. The parameter <b>distance-to-sample</b> is optional, "
                    "but desirable. If <b>distance-to-sample</b> is zero then no scaling will be "
                    "applied to data to compensate for changing distance.")
        self.text_edit.setReadOnly(True)

        self.le_file_path = LineEditReadOnly()
        set_tooltip(self.le_file_path,
                    "Full path to the file used to <b>save the calibration data</b>. The path "
                    "can be changed in file selection dialog box.")
        self.pb_file_path = PushButtonMinimumWidth("..")
        set_tooltip(self.pb_file_path,
                    "Change <b>file path</b> for saving the calibration data.")
        self.pb_file_path.clicked.connect(self.pb_file_path_clicked)

        self.le_distance_to_sample = QLineEdit()
        set_tooltip(self.le_distance_to_sample,
                    "<b>Distance between the detector and the sample during calibration. If the value "
                    "is 0, then no scaling is applied to data to correct the data if distance-to-sample "
                    "is changed between calibration and measurement.")

        self.cb_overwrite = QCheckBox("OverwriteExisting")
        set_tooltip(self.cb_overwrite,
                    "Overwrite the <b>existin</b> file. This is a safety feature implemented to protect "
                    "valueable results from accidental deletion.")

        vbox = QVBoxLayout()

        hbox = QHBoxLayout()
        hbox.addWidget(QLabel("The following data will be saved to JSON file:"))
        hbox.addStretch(1)
        vbox.addLayout(hbox)

        vbox.addWidget(self.text_edit)

        hbox = QHBoxLayout()
        hbox.addWidget(QLabel("Path: "))
        hbox.addWidget(self.pb_file_path)
        hbox.addWidget(self.le_file_path)
        vbox.addLayout(hbox)

        hbox = QHBoxLayout()
        hbox.addWidget(QLabel("Distance-to-sample:"))
        hbox.addWidget(self.le_distance_to_sample)
        hbox.addStretch(1)
        hbox.addWidget(self.cb_overwrite)
        vbox.addLayout(hbox)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        vbox.addWidget(button_box)

        self.setLayout(vbox)

        # Set and display file path
        file_path = os.path.expanduser(file_path)
        self.file_path = file_path

    @property
    def file_path(self):
        return self.__file_path

    @file_path.setter
    def file_path(self, file_path):
        self.__file_path = file_path
        self.le_file_path.setText(file_path)

    def pb_file_path_clicked(self):
        file_path = QFileDialog.getSaveFileName(self, "Select File to Save Quantitative Calibration",
                                                self.file_path,
                                                "JSON (*.json);; All (*)",
                                                options=QFileDialog.DontConfirmOverwrite)
        file_path = file_path[0]
        if file_path:
            self.file_path = file_path
            print(f"Selected file path for saving calibration standard: '{file_path}'")


class DialogExportToTiffAndTxt(QDialog):

    def __init__(self, parent=None, *, dir_path=""):

        super().__init__(parent)

        self.__dir_path = ""

        self.setWindowTitle("Export XRF Maps to TIFF and/or TXT Files")
        self.setMinimumHeight(600)
        self.setMinimumWidth(600)
        self.resize(600, 600)

        self.combo_select_dataset = QComboBox()
        sample_datasets = ["scan2D_28844_amk_fit", "scan2D_28844_amk_roi",
                           "scan2D_28844_amk_scaler", "positions"]
        # datasets = ["Select Dataset ..."] + sample_datasets
        datasets = sample_datasets
        self.combo_select_dataset.addItems(datasets)
        set_tooltip(self.combo_select_dataset,
                    "Select <b>dataset</b>. Initially, the selection matches the dataset activated "
                    "in <b>XRF Maps</b> tab, but the selection may be changed if different dataset "
                    "needs to be saved.")

        self.combo_normalization = QComboBox()
        set_tooltip(self.combo_normalization,
                    "Select <b>scaler</b> used for data normalization. Initially, the selection matches "
                    "the scaler activated in <b>XRF Maps</b> tab, but the selection may be changed "
                    "if needed")
        sample_scalers = ["i0", "i0_time", "time", "time_diff"]
        scalers = ["Normalize by ..."] + sample_scalers
        self.combo_normalization.addItems(scalers)

        self.cb_interpolate = QCheckBox("Interpolate to uniform grid")
        set_tooltip(self.cb_interpolate,
                    "Interpolate pixel coordinates to <b>uniform grid</b>. The initial choice is "
                    "copied from <b>XRF Maps</b> tab.")

        self.cb_quantitative = QCheckBox("Quantitative normalization")
        set_tooltip(self.cb_quantitative,
                    "Apply <b>quantitative normalization</b> before saving the maps. "
                    "The initial choice is copied from <b>XRF Maps</b> tab.")

        self.group_settings = QGroupBox("Settings (selections from XRF Maps tab)")
        grid = QGridLayout()
        grid.addWidget(self.combo_select_dataset, 0, 0)
        grid.addWidget(self.combo_normalization, 0, 1)
        grid.addWidget(self.cb_interpolate, 1, 0)
        grid.addWidget(self.cb_quantitative, 1, 1)
        self.group_settings.setLayout(grid)

        self.te_saved_files = QTextEdit()
        set_tooltip(self.te_saved_files,
                    "The list of <b>data files</b> about to be created.")
        self.te_saved_files.setReadOnly(True)

        self.le_dir_path = LineEditReadOnly()
        set_tooltip(self.le_dir_path,
                    "<b>Root directory</b> for saving TIFF and TXT files. The files will be saved "
                    "in subdirectories inside the root directory.")
        self.pb_dir_path = PushButtonMinimumWidth("..")
        set_tooltip(self.pb_dir_path,
                    "Change to <b>root directory</b> for TIFF and TXT files.")
        self.pb_dir_path.clicked.connect(self.pb_dir_path_clicked)

        self.cb_save_tiff = QCheckBox("Save TIFF")
        set_tooltip(self.cb_save_tiff,
                    "Save XRF Maps as <b>TIFF</b> files.")
        self.cb_save_tiff.setChecked(True)
        self.cb_save_tiff.toggled.connect(self.cb_save_tiff_toggled)
        self.cb_save_txt = QCheckBox("Save TXT")
        set_tooltip(self.cb_save_txt,
                    "Save XRF Maps as <b>TXT</b> files.")
        self.cb_save_txt.toggled.connect(self.cb_save_txt_toggled)

        vbox = QVBoxLayout()

        hbox = QHBoxLayout()
        hbox.addWidget(self.group_settings)
        hbox.addStretch(1)
        vbox.addLayout(hbox)

        hbox = QHBoxLayout()
        hbox.addWidget(QLabel("The following files will be created:"))
        hbox.addStretch(1)
        vbox.addLayout(hbox)
        vbox.addWidget(self.te_saved_files)

        hbox = QHBoxLayout()
        hbox.addWidget(QLabel("Directory: "))
        hbox.addWidget(self.pb_dir_path)
        hbox.addWidget(self.le_dir_path)
        vbox.addLayout(hbox)

        self.button_box = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        hbox = QHBoxLayout()
        hbox.addStretch(1)
        hbox.addWidget(self.cb_save_tiff)
        hbox.addWidget(self.cb_save_txt)
        hbox.addStretch(1)
        hbox.addWidget(self.button_box)
        vbox.addLayout(hbox)

        self.setLayout(vbox)

        # Set and display file path
        dir_path = os.path.expanduser(dir_path)
        self.dir_path = dir_path

    @property
    def dir_path(self):
        return self.__dir_path

    @dir_path.setter
    def dir_path(self, dir_path):
        self.__dir_path = dir_path
        self.le_dir_path.setText(dir_path)

    def pb_dir_path_clicked(self):
        # Note: QFileDialog.ShowDirsOnly is not set on purpose, so that the dialog
        #   could be used to inspect directory contents. Files can not be selected anyway.
        dir_path = QFileDialog.getExistingDirectory(self, "Select Root Directory for TIFF and TXT Files",
                                                    self.dir_path,
                                                    options=QFileDialog.DontResolveSymlinks)
        if dir_path:
            self.dir_path = dir_path
            print(f"Selected directory: '{self.dir_path}'")

    def enable_save_button(self):

        btn_ok = self.button_box.button(QDialogButtonBox.Save)

        if self.cb_save_tiff.isChecked() or self.cb_save_txt.isChecked():
            btn_ok.setEnabled(True)
            print("Enable Save button")
        else:
            btn_ok.setEnabled(False)
            print("Disable Save button")

    def cb_save_tiff_toggled(self, state):
        self.enable_save_button()

    def cb_save_txt_toggled(self, state):
        self.enable_save_button()
