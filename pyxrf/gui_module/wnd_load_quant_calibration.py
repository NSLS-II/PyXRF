import textwrap

from qtpy.QtWidgets import (
    QPushButton,
    QHBoxLayout,
    QVBoxLayout,
    QGroupBox,
    QCheckBox,
    QLabel,
    QComboBox,
    QFileDialog,
    QRadioButton,
    QButtonGroup,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QWidget,
    QScrollArea,
    QTabWidget,
    QFrame,
    QMessageBox,
)
from qtpy.QtGui import QBrush, QColor, QDoubleValidator
from qtpy.QtCore import Qt, Slot, Signal

from .useful_widgets import get_background_css, SecondaryWindow, set_tooltip, LineEditExtended
from .dlg_view_calib_standard import DialogViewCalibStandard

import logging

logger = logging.getLogger(__name__)


class WndLoadQuantitativeCalibration(SecondaryWindow):

    signal_quantitative_calibration_changed = Signal()

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
        self.combo_set_table_header.currentIndexChanged.connect(self.combo_set_table_header_index_changed)

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

        self.table.setStyleSheet("QTableWidget::item{color: black;}")

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

        class _LabelBlack(QLabel):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.setStyleSheet("color: black")

        for cdata, csettings in zip(calib_data, calib_settings):
            frame = QFrame()
            frame.setFrameStyle(QFrame.StyledPanel)
            frame.setStyleSheet(get_background_css((200, 255, 200), widget="QFrame"))

            _vbox = QVBoxLayout()

            name = cdata["name"]  # Standard name (can be arbitrary string
            # If name is long, then print it in a separate line
            _name_is_long = len(name) > 30

            pb_view = QPushButton("View ...")
            self.group_view.addButton(pb_view)
            pb_remove = QPushButton("Remove")
            self.group_remove.addButton(pb_remove)

            # Row 1: serial, name
            serial = cdata["serial"]
            _hbox = QHBoxLayout()
            _hbox.addWidget(_LabelBlack(f"<b>Standard</b> #{serial}"))
            if not _name_is_long:
                _hbox.addWidget(_LabelBlack(f"'{name}'"))
            _hbox.addStretch(1)
            _hbox.addWidget(pb_view)
            _hbox.addWidget(pb_remove)
            _vbox.addLayout(_hbox)

            # Optional row
            if _name_is_long:
                # Wrap name if it is extemely long
                name = textwrap.fill(name, width=80)
                _hbox = QHBoxLayout()
                _hbox.addWidget(_LabelBlack("<b>Name:</b> "), 0, Qt.AlignTop)
                _hbox.addWidget(_LabelBlack(name), 0, Qt.AlignTop)
                _hbox.addStretch(1)
                _vbox.addLayout(_hbox)

            # Row 2: description
            description = textwrap.fill(cdata["description"], width=80)
            _hbox = QHBoxLayout()
            _hbox.addWidget(_LabelBlack("<b>Description:</b>"), 0, Qt.AlignTop)
            _hbox.addWidget(_LabelBlack(f"{description}"), 0, Qt.AlignTop)
            _hbox.addStretch(1)
            _vbox.addLayout(_hbox)

            # Row 3:
            incident_energy = cdata["incident_energy"]
            scaler = cdata["scaler_name"]
            detector_channel = cdata["detector_channel"]
            distance_to_sample = cdata["distance_to_sample"]
            _hbox = QHBoxLayout()
            _hbox.addWidget(_LabelBlack(f"<b>Incident energy, keV:</b> {incident_energy}"))
            _hbox.addWidget(_LabelBlack(f"  <b>Scaler:</b> {scaler}"))
            _hbox.addWidget(_LabelBlack(f"  <b>Detector channel:</b> {detector_channel}"))
            _hbox.addWidget(_LabelBlack(f"  <b>Distance-to-sample:</b> {distance_to_sample}"))
            _hbox.addStretch(1)
            _vbox.addLayout(_hbox)

            # Row 4: file name
            fln = textwrap.fill(csettings["file_path"], width=80)
            _hbox = QHBoxLayout()
            _hbox.addWidget(_LabelBlack("<b>Source file:</b>"), 0, Qt.AlignTop)
            _hbox.addWidget(_LabelBlack(fln), 0, Qt.AlignTop)
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
                txt = cdata["name"]
            else:
                txt = cdata["serial"]
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
                ks = list(cdata["element_lines"].keys())
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
                    if eline in cdata["element_lines"]:
                        rb = QRadioButton()
                        if self.gpc.get_quant_calibration_is_eline_selected(eline, q_file_path):
                            rb.setChecked(True)

                        rb.setStyleSheet("color: black")

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
                        density = cdata["element_lines"][eline]["density"]
                        fluorescence = cdata["element_lines"][eline]["fluorescence"]
                        ttip = f"Fluorescence (F): {fluorescence:12g}\nDensity (D): {density:12g}\n"
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

    @Slot()
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
            self.le_distance_to_sample,
            "Distance between <b>the sample and the detector</b>. The ratio between of the distances "
            "during calibration and measurement is used to scale computed concentrations. "
            "If distance-to-sample is 0 for calibration or measurement, then no scaling is performed.",
        )
        set_tooltip(
            self.combo_set_table_header,
            "Use <b>Serial Number</b> or <b>Name</b> of the calibration standard in the header of the table",
        )
        set_tooltip(
            self.table,
            "Use Radio Buttons to select the <b>source of calibration data</b> for each emission line. "
            "This feature is needed if multiple loaded calibration files have data on the same "
            "emission line.",
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
        file_name = QFileDialog.getOpenFileName(
            self, "Select File with Quantitative Calibration Data", current_dir, "JSON (*.json);; All (*)"
        )
        file_name = file_name[0]
        if file_name:
            try:
                logger.debug(f"Loading quantitative calibration from file: '{file_name}'")
                self.gpc.load_quantitative_calibration_data(file_name)
                self.update_all_data()
                self._update_maps_auto()
            except Exception:
                msg = "The selected JSON file has incorrect format. Select a different file."
                msgbox = QMessageBox(QMessageBox.Critical, "Data Loading Error", msg, QMessageBox.Ok, parent=self)
                msgbox.exec()

    def pb_view_clicked(self, button):
        try:
            n_standard = self.pbs_view.index(button)
            calib_settings = self.gpc.get_quant_calibration_settings()
            file_path = calib_settings[n_standard]["file_path"]
            calib_preview = self.gpc.get_quant_calibration_text_preview(file_path)
            dlg = DialogViewCalibStandard(None, file_path=file_path, calib_preview=calib_preview)
            dlg.exec()
        except ValueError:
            logger.error("'View' button was pressed, but not found in the list of buttons")

    def pb_remove_clicked(self, button):
        try:
            n_standard = self.pbs_remove.index(button)
            calib_settings = self.gpc.get_quant_calibration_settings()
            file_path = calib_settings[n_standard]["file_path"]
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
