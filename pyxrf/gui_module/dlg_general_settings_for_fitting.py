import copy

from qtpy.QtWidgets import (QHBoxLayout, QVBoxLayout, QGroupBox,
                            QCheckBox, QLabel, QDialog, QDialogButtonBox,
                            QGridLayout)
from qtpy.QtCore import Qt
from .useful_widgets import (set_tooltip, LineEditExtended,
                             IntValidatorStrict, DoubleValidatorStrict)

import logging
logger = logging.getLogger(__name__)


class DialogGeneralFittingSettings(QDialog):

    def __init__(self, parent=None):

        super().__init__(parent)

        self.setWindowTitle("General Settings for Fitting Algorithm")

        self._dialog_data = {}

        self._validator_int = IntValidatorStrict()
        self._validator_float = DoubleValidatorStrict()

        vbox_left = QVBoxLayout()

        # ===== Top-left section of the dialog box =====
        self.le_max_iterations = LineEditExtended()
        set_tooltip(self.le_max_iterations,
                    "<b>Maximum number of iterations</b> used for total spectrum fitting.")
        self.le_tolerance_stopping = LineEditExtended()
        set_tooltip(self.le_tolerance_stopping,
                    "<b>Tolerance</b> setting for total spectrum fitting.")
        self.le_escape_ratio = LineEditExtended()
        set_tooltip(self.le_escape_ratio,
                    "Parameter for total spectrum fitting: <b>escape ration</b>")
        grid = QGridLayout()
        grid.addWidget(QLabel("Iterations (max):"), 0, 0)
        grid.addWidget(self.le_max_iterations, 0, 1)
        grid.addWidget(QLabel("Tolerance (stopping):"), 1, 0)
        grid.addWidget(self.le_tolerance_stopping, 1, 1)
        grid.addWidget(QLabel("Escape peak ratio:"), 2, 0)
        grid.addWidget(self.le_escape_ratio, 2, 1)
        self.group_total_spectrum_fitting = QGroupBox(
            "Fitting of Total Spectrum (Model)")
        self.group_total_spectrum_fitting.setLayout(grid)
        vbox_left.addWidget(self.group_total_spectrum_fitting)

        # ===== Bottom-left section of the dialog box =====

        # Incident energy and the selected range
        self.le_incident_energy = LineEditExtended()
        set_tooltip(self.le_incident_energy,
                    "<b>Incident energy</b> in keV")
        self.le_range_low = LineEditExtended()
        set_tooltip(self.le_range_low,
                    "<b>Lower boundary</b> of the selected range in keV.")
        self.le_range_high = LineEditExtended()
        set_tooltip(self.le_range_high,
                    "<b>Upper boundary</b> of the selected range in keV.")
        self.group_energy_range = QGroupBox("Incident Energy and Selected Range")
        grid = QGridLayout()
        grid.addWidget(QLabel("Incident energy, keV"), 0, 0)
        grid.addWidget(self.le_incident_energy, 0, 1)
        grid.addWidget(QLabel("Range (low), keV"), 1, 0)
        grid.addWidget(self.le_range_low, 1, 1)
        grid.addWidget(QLabel("Range (high), keV"), 2, 0)
        grid.addWidget(self.le_range_high, 2, 1)
        self.group_energy_range.setLayout(grid)
        vbox_left.addWidget(self.group_energy_range)

        vbox_right = QVBoxLayout()

        # ===== Top-right section of the dialog box =====

        self.cb_linear_baseline = QCheckBox("Subtract linear baseline")
        set_tooltip(self.cb_linear_baseline,
                    "Subtract baseline as represented as a constant. <b>XRF Map generation</b>. "
                    "Baseline subtraction is performed as part of NNLS fitting.")
        self.cb_snip_baseline = QCheckBox("Subtract baseline using SNIP")
        set_tooltip(self.cb_snip_baseline,
                    "Subtract baseline using SNIP method. <b>XRF Map generation</b>. "
                    "This is a separate step of processing and can be used together with "
                    "'linear' baseline subtraction if needed.")

        # This option is not supported. In the future it may be removed
        #   if not needed or implemented.
        self.cb_add_const_to_data = QCheckBox("Add const. bias to data")
        self.cb_add_const_to_data.setEnabled(False)
        self.lb_add_const_to_data = QLabel("Constant bias:")
        self.lb_add_const_to_data.setEnabled(False)
        self.le_add_const_to_data = LineEditExtended()
        self.le_add_const_to_data.setEnabled(False)

        vbox = QVBoxLayout()
        vbox.addWidget(self.cb_linear_baseline)
        vbox.addWidget(self.cb_snip_baseline)
        vbox.addWidget(self.cb_add_const_to_data)
        hbox = QHBoxLayout()
        hbox.addWidget(self.lb_add_const_to_data)
        hbox.addWidget(self.le_add_const_to_data)
        vbox.addLayout(hbox)

        self.group_pixel_fitting = QGroupBox("Fitting of Single Spectra (XRF Maps)")
        self.group_pixel_fitting.setLayout(vbox)

        vbox_right.addWidget(self.group_pixel_fitting)

        # ===== Bottom-right section of the dialog box =====

        self.le_snip_window_size = LineEditExtended()
        set_tooltip(self.le_snip_window_size,
                    "Window size for <b>SNIP</b> algorithm. Used both for total spectrum fitting "
                    "and XRF Map generation. SNIP baseline subtraction is always performed while "
                    "fitting total spectrum, but its effect may be reduced or eliminated by setting "
                    "the window size to some large value.")

        vbox = QVBoxLayout()
        hbox = QHBoxLayout()
        hbox.addWidget(QLabel("SNIP window size(*):"))
        hbox.addWidget(self.le_snip_window_size)
        vbox.addLayout(hbox)
        vbox.addWidget(QLabel("*Total spectrum fitting always includes \n"
                              "    SNIP baseline subtraction"))

        self.group_all_fitting = QGroupBox("All Fitting")
        self.group_all_fitting.setLayout(vbox)

        vbox_right.addWidget(self.group_all_fitting)
        vbox_right.addStretch(1)

        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.button(QDialogButtonBox.Cancel).setDefault(True)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        self.pb_ok = button_box.button(QDialogButtonBox.Ok)

        hbox = QHBoxLayout()
        hbox.addLayout(vbox_left)
        hbox.addLayout(vbox_right)

        vbox = QVBoxLayout()
        vbox.addLayout(hbox)
        vbox.addWidget(button_box)

        self.setLayout(vbox)

        self.le_max_iterations.textChanged.connect(self.le_max_iterations_text_changed)
        self.le_max_iterations.editingFinished.connect(self.le_max_iterations_editing_finished)
        self.le_tolerance_stopping.textChanged.connect(self.le_tolerance_stopping_text_changed)
        self.le_tolerance_stopping.editingFinished.connect(self.le_tolerance_stopping_editing_finished)
        self.le_escape_ratio.textChanged.connect(self.le_escape_ratio_text_changed)
        self.le_escape_ratio.editingFinished.connect(self.le_escape_ratio_editing_finished)

        self.le_incident_energy.textChanged.connect(self.le_incident_energy_text_changed)
        self.le_incident_energy.editingFinished.connect(self.le_incident_energy_editing_finished)
        self.le_range_low.textChanged.connect(self.le_range_low_text_changed)
        self.le_range_low.editingFinished.connect(self.le_range_low_editing_finished)
        self.le_range_high.textChanged.connect(self.le_range_high_text_changed)
        self.le_range_high.editingFinished.connect(self.le_range_high_editing_finished)

        self.le_snip_window_size.textChanged.connect(self.le_snip_window_size_text_changed)
        self.le_snip_window_size.editingFinished.connect(self.le_snip_window_size_editing_finished)

        self.cb_linear_baseline.stateChanged.connect(self.cb_linear_baseline_state_changed)
        self.cb_snip_baseline.stateChanged.connect(self.cb_snip_baseline_state_changed)

    def le_max_iterations_text_changed(self, text):
        self._validate_all()

    def le_max_iterations_editing_finished(self):
        if self._validate_max_interations():
            self._dialog_data["max_iterations"] = int(self._read_le_value(self.le_max_iterations))
        else:
            self._show_max_iterations()

    def le_tolerance_stopping_text_changed(self, text):
        self._validate_all()

    def le_tolerance_stopping_editing_finished(self):
        if self._validate_tolerance_stopping():
            self._dialog_data["tolerance"] = self._read_le_value(self.le_tolerance_stopping)
        else:
            self._show_tolerance_stopping()

    def le_escape_ratio_text_changed(self, text):
        self._validate_all()

    def le_escape_ratio_editing_finished(self):
        if self._validate_escape_ratio():
            self._dialog_data["escape_peak_ratio"] = self._read_le_value(self.le_escape_ratio)
        else:
            self._show_escape_ratio()

    def le_incident_energy_text_changed(self, text):
        if self._validate_incident_energy(text):
            val = float(text)
            val_range_high = val + 0.8
            self._show_range_high(val_range_high)
        self._validate_all()

    def le_incident_energy_editing_finished(self):
        if self._validate_incident_energy():
            self._dialog_data["incident_energy"] = self._read_le_value(self.le_incident_energy)
        else:
            self._show_incident_energy()
        if self._validate_range():
            self._dialog_data["range_low"] = self._read_le_value(self.le_range_low)
            self._dialog_data["range_high"] = self._read_le_value(self.le_range_high)
        else:
            self._show_range_low()
            self._show_range_high()

    def le_range_low_text_changed(self, text):
        self._validate_all()

    def le_range_low_editing_finished(self):
        if self._validate_range():
            self._dialog_data["range_low"] = self._read_le_value(self.le_range_low)
            self._dialog_data["range_high"] = self._read_le_value(self.le_range_high)
        else:
            self._show_range_low()
            self._show_range_high()

    def le_range_high_text_changed(self, text):
        self._validate_all()

    def le_range_high_editing_finished(self):
        if self._validate_range():
            self._dialog_data["range_low"] = self._read_le_value(self.le_range_low)
            self._dialog_data["range_high"] = self._read_le_value(self.le_range_high)
        else:
            self._show_range_low()
            self._show_range_high()

    def le_snip_window_size_text_changed(self, text):
        self._validate_all()

    def le_snip_window_size_editing_finished(self):
        if self._validate_snip_window_size():
            self._dialog_data["snip_window_size"] = self._read_le_value(self.le_snip_window_size)
        else:
            self._show_snip_window_size()

    def cb_linear_baseline_state_changed(self, state):
        self._dialog_data["subtract_baseline_linear"] = state == Qt.Checked

    def cb_snip_baseline_state_changed(self, state):
        self._dialog_data["subtract_baseline_snip"] = state == Qt.Checked

    def set_dialog_data(self, dialog_data):
        self._dialog_data = copy.deepcopy(dialog_data)
        self._show_all()

    def get_dialog_data(self):
        return self._dialog_data

    def _show_all(self):
        self._show_max_iterations()
        self._show_tolerance_stopping()
        self._show_escape_ratio()
        self._show_incident_energy()
        self._show_range_high()
        self._show_range_low()
        self._show_linear_baseline()
        self._show_snip_baseline()
        self._show_snip_window_size()

    def _show_max_iterations(self):
        val = self._dialog_data["max_iterations"]
        self.le_max_iterations.setText(f"{val}")

    def _show_tolerance_stopping(self):
        val = self._dialog_data["tolerance"]
        self.le_tolerance_stopping.setText(self._format_float(val))

    def _show_escape_ratio(self):
        val = self._dialog_data["escape_peak_ratio"]
        self.le_escape_ratio.setText(self._format_float(val))

    def _show_incident_energy(self):
        val = self._dialog_data["incident_energy"]
        self.le_incident_energy.setText(self._format_float(val))

    def _show_range_high(self, val=None):
        val = self._dialog_data["range_high"] if val is None else val
        self.le_range_high.setText(self._format_float(val))

    def _show_range_low(self):
        val = self._dialog_data["range_low"]
        self.le_range_low.setText(self._format_float(val))

    def _show_linear_baseline(self):
        val = self._dialog_data["subtract_baseline_linear"]
        self.cb_linear_baseline.setChecked(Qt.Checked if val else Qt.Unchecked)

    def _show_snip_baseline(self):
        val = self._dialog_data["subtract_baseline_snip"]
        self.cb_snip_baseline.setChecked(Qt.Checked if val else Qt.Unchecked)

    def _show_snip_window_size(self):
        val = self._dialog_data["snip_window_size"]
        self.le_snip_window_size.setText(self._format_float(val))

    def _validate_all(self):
        valid = (self._validate_max_interations() and
                 self._validate_tolerance_stopping() and
                 self._validate_escape_ratio() and
                 self._validate_incident_energy() and
                 self._validate_range() and
                 self._validate_snip_window_size())

        self.pb_ok.setEnabled(valid)

    def _validate_max_interations(self, text=None):
        if text is None:
            text = self.le_max_iterations.text()

        valid = self._validate_int(text, v_min=1)
        self.le_max_iterations.setValid(valid)

        return valid

    def _validate_tolerance_stopping(self, text=None):
        if text is None:
            text = self.le_tolerance_stopping.text()

        valid = self._validate_float(text, v_min=1e-30)
        self.le_tolerance_stopping.setValid(valid)

        return valid

    def _validate_escape_ratio(self, text=None):
        if text is None:
            text = self.le_escape_ratio.text()

        valid = self._validate_float(text, v_min=0)
        self.le_escape_ratio.setValid(valid)

        return valid

    def _validate_incident_energy(self, text=None):
        if text is None:
            text = self.le_incident_energy.text()

        valid = self._validate_float(text, v_min=0)
        self.le_incident_energy.setValid(valid)

        return valid

    def _validate_range(self, low_text=None, high_text=None):
        if low_text is None:
            low_text = self.le_range_low.text()
        if high_text is None:
            high_text = self.le_range_high.text()

        valid = False
        if self._validate_float(low_text, v_min=0) and self._validate_float(high_text, v_min=0):
            v_low = float(low_text)
            v_high = float(high_text)
            if v_low < v_high:
                valid = True

        self.le_range_high.setValid(valid)
        self.le_range_low.setValid(valid)

        return valid

    def _validate_snip_window_size(self, text=None):
        if text is None:
            text = self.le_snip_window_size.text()

        valid = self._validate_float(text, v_min=1e-30)
        self.le_snip_window_size.setValid(valid)

        return valid

    def _validate_int(self, text, *, v_min=None, v_max=None):
        valid = False
        if self._validator_int.validate(text, 0)[0] == IntValidatorStrict.Acceptable:
            valid = True
            v_int = int(text)
            if (v_min is not None) and (v_int < v_min):
                valid = False
            if (v_max is not None) and (v_int > v_max):
                valid = False
        return valid

    def _validate_float(self, text, *, v_min=None, v_max=None):
        valid = False
        if self._validator_float.validate(text, 0)[0] == DoubleValidatorStrict.Acceptable:
            valid = True
            v_float = float(text)
            if (v_min is not None) and (v_float < v_min):
                valid = False
            if (v_max is not None) and (v_float > v_max):
                valid = False
        return valid

    def _format_float(self, val):
        return f"{val:.10g}"

    def _read_le_value(self, line_edit):
        """It is assumed that the value is validated before the function is called"""
        return float(line_edit.text())
