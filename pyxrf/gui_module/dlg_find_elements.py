import copy

from qtpy.QtWidgets import QPushButton, QVBoxLayout, QGroupBox, QLabel, QDialog, QDialogButtonBox, QGridLayout
from qtpy.QtGui import QDoubleValidator

from .useful_widgets import set_tooltip, LineEditExtended


class DialogFindElements(QDialog):
    def __init__(self, parent=None):

        super().__init__(parent)

        self._dialog_data = {}

        self.setWindowTitle("Find Elements in Sample")

        # Check this flag after the dialog is exited with 'True' value
        #   If this flag is True, then run element search, if False,
        #   then simply save the changed parameter values.
        self.find_elements_requested = False

        self.validator = QDoubleValidator()

        self.le_e_calib_a0 = LineEditExtended()
        self.le_e_calib_a0.setValidator(self.validator)
        self.pb_e_calib_a0_default = QPushButton("Default")
        self.pb_e_calib_a0_default.setAutoDefault(False)

        self.le_e_calib_a1 = LineEditExtended()
        self.le_e_calib_a1.setValidator(self.validator)
        self.pb_e_calib_a1_default = QPushButton("Default")
        self.pb_e_calib_a1_default.setAutoDefault(False)

        self.le_e_calib_a2 = LineEditExtended()
        self.le_e_calib_a2.setValidator(self.validator)
        self.pb_e_calib_a2_default = QPushButton("Default")
        self.pb_e_calib_a2_default.setAutoDefault(False)

        self.group_energy_axis_calib = QGroupBox("Polynomial Approximation of Energy Axis")
        set_tooltip(
            self.group_energy_axis_calib,
            "Parameters of polynomial approximation of <b>energy axis</b>. "
            "The values of the bins for photon energies are approximated "
            "using 2nd degree polynomial <b>E(n) = a0 + a1 * n + a2 * n^2</b>, "
            "where <b>n</b> is bin number (typically in the range 0..4096).",
        )
        grid = QGridLayout()
        grid.addWidget(QLabel("Bias (a0):"), 0, 0)
        grid.addWidget(self.le_e_calib_a0, 0, 1)
        grid.addWidget(self.pb_e_calib_a0_default, 0, 2)
        grid.addWidget(QLabel("Linear (a1):"), 1, 0)
        grid.addWidget(self.le_e_calib_a1, 1, 1)
        grid.addWidget(self.pb_e_calib_a1_default, 1, 2)
        grid.addWidget(QLabel("Quadratic (a2):"), 2, 0)
        grid.addWidget(self.le_e_calib_a2, 2, 1)
        grid.addWidget(self.pb_e_calib_a2_default, 2, 2)
        self.group_energy_axis_calib.setLayout(grid)

        self.le_fwhm_b1 = LineEditExtended()
        self.le_fwhm_b1.setValidator(self.validator)
        self.pb_fwhm_b1_default = QPushButton("Default")
        self.pb_fwhm_b1_default.setAutoDefault(False)

        self.le_fwhm_b2 = LineEditExtended()
        self.le_fwhm_b2.setValidator(self.validator)
        self.pb_fwhm_b2_default = QPushButton("Default")
        self.pb_fwhm_b2_default.setAutoDefault(False)

        self.group_fwhm = QGroupBox("Peak FWHM Settings")
        set_tooltip(
            self.group_fwhm,
            "Parameters used to estimate <b>FWHM</b> of peaks based on energy: "
            "<b>b1</b> - 'offset', <b>b2</b> - 'fanoprime'",
        )
        grid = QGridLayout()
        grid.addWidget(QLabel("Coefficient b1:"), 0, 0)
        grid.addWidget(self.le_fwhm_b1, 0, 1)
        grid.addWidget(self.pb_fwhm_b1_default, 0, 2)
        grid.addWidget(QLabel("Coefficient b2:"), 1, 0)
        grid.addWidget(self.le_fwhm_b2, 1, 1)
        grid.addWidget(self.pb_fwhm_b2_default, 1, 2)
        self.group_fwhm.setLayout(grid)

        self.le_incident_energy = LineEditExtended()
        self.le_incident_energy.setValidator(self.validator)
        set_tooltip(self.le_incident_energy, "<b>Incident energy</b> in keV.")
        self.pb_incident_energy_default = QPushButton("Default")
        self.pb_incident_energy_default.setAutoDefault(False)
        self.le_range_low = LineEditExtended()
        self.le_range_low.setValidator(self.validator)
        set_tooltip(self.le_range_low, "<b>Lower boundary</b> of the selected range in keV.")
        self.pb_range_low_default = QPushButton("Default")
        self.pb_range_low_default.setAutoDefault(False)
        self.le_range_high = LineEditExtended()
        self.le_range_high.setValidator(self.validator)
        set_tooltip(self.le_range_high, "<b>Upper boundary</b> of the selected range in keV.")
        self.pb_range_high_default = QPushButton("Default")
        self.pb_range_high_default.setAutoDefault(False)
        self.group_energy_range = QGroupBox("Incident Energy and Selected Range")
        grid = QGridLayout()
        grid.addWidget(QLabel("Incident energy, keV"), 0, 0)
        grid.addWidget(self.le_incident_energy, 0, 1)
        grid.addWidget(self.pb_incident_energy_default, 0, 2)
        grid.addWidget(QLabel("Range (low), keV"), 1, 0)
        grid.addWidget(self.le_range_low, 1, 1)
        grid.addWidget(self.pb_range_low_default, 1, 2)
        grid.addWidget(QLabel("Range (high), keV"), 2, 0)
        grid.addWidget(self.le_range_high, 2, 1)
        grid.addWidget(self.pb_range_high_default, 2, 2)
        self.group_energy_range.setLayout(grid)

        self.pb_find_elements = QPushButton("Find &Elements")
        self.pb_find_elements.clicked.connect(self.pb_find_elements_clicked)
        self.pb_apply_settings = QPushButton("&Apply Settings")

        # 'Close' button box
        button_box = QDialogButtonBox(QDialogButtonBox.Cancel)
        button_box.addButton(self.pb_find_elements, QDialogButtonBox.YesRole)
        button_box.addButton(self.pb_apply_settings, QDialogButtonBox.AcceptRole)
        button_box.button(QDialogButtonBox.Cancel).setDefault(True)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        self.pb_cancel = button_box.button(QDialogButtonBox.Cancel)
        self.pb_cancel.setAutoDefault(False)
        self.pb_apply_settings.setAutoDefault(True)

        vbox = QVBoxLayout()
        vbox.addWidget(self.group_energy_axis_calib)
        vbox.addWidget(self.group_fwhm)
        vbox.addWidget(self.group_energy_range)
        vbox.addWidget(button_box)

        self.setLayout(vbox)

        self.le_e_calib_a0.editingFinished.connect(self.le_e_calib_a0_editing_finished)
        self.le_e_calib_a1.editingFinished.connect(self.le_e_calib_a1_editing_finished)
        self.le_e_calib_a2.editingFinished.connect(self.le_e_calib_a2_editing_finished)
        self.le_fwhm_b1.editingFinished.connect(self.le_fwhm_b1_editing_finished)
        self.le_fwhm_b2.editingFinished.connect(self.le_fwhm_b2_editing_finished)
        self.le_incident_energy.editingFinished.connect(self.le_incident_energy_editing_finished)
        self.le_range_low.editingFinished.connect(self.le_range_low_editing_finished)
        self.le_range_high.editingFinished.connect(self.le_range_high_editing_finished)

        self.le_e_calib_a0.focusOut.connect(self.le_e_calib_a0_focus_out)
        self.le_e_calib_a1.focusOut.connect(self.le_e_calib_a1_focus_out)
        self.le_e_calib_a2.focusOut.connect(self.le_e_calib_a2_focus_out)
        self.le_fwhm_b1.focusOut.connect(self.le_fwhm_b1_focus_out)
        self.le_fwhm_b2.focusOut.connect(self.le_fwhm_b2_focus_out)
        self.le_incident_energy.focusOut.connect(self.le_incident_energy_focus_out)
        self.le_range_low.focusOut.connect(self.le_range_low_focus_out)
        self.le_range_high.focusOut.connect(self.le_range_high_focus_out)

        self.le_e_calib_a0.textChanged.connect(self.le_e_calib_a0_text_changed)
        self.le_e_calib_a1.textChanged.connect(self.le_e_calib_a1_text_changed)
        self.le_e_calib_a2.textChanged.connect(self.le_e_calib_a2_text_changed)
        self.le_fwhm_b1.textChanged.connect(self.le_fwhm_b1_text_changed)
        self.le_fwhm_b2.textChanged.connect(self.le_fwhm_b2_text_changed)
        self.le_incident_energy.textChanged.connect(self.le_incident_energy_text_changed)
        self.le_range_low.textChanged.connect(self.le_range_low_text_changed)
        self.le_range_high.textChanged.connect(self.le_range_high_text_changed)

        self.pb_e_calib_a0_default.clicked.connect(self.pb_e_calib_a0_default_clicked)
        self.pb_e_calib_a1_default.clicked.connect(self.pb_e_calib_a1_default_clicked)
        self.pb_e_calib_a2_default.clicked.connect(self.pb_e_calib_a2_default_clicked)
        self.pb_fwhm_b1_default.clicked.connect(self.pb_fwhm_b1_default_clicked)
        self.pb_fwhm_b2_default.clicked.connect(self.pb_fwhm_b2_default_clicked)
        self.pb_incident_energy_default.clicked.connect(self.pb_incident_energy_default_clicked)
        self.pb_range_low_default.clicked.connect(self.pb_range_low_default_clicked)
        self.pb_range_high_default.clicked.connect(self.pb_range_high_default_clicked)

    def _format_float(self, value):
        return f"{value:.10g}"

    def set_dialog_data(self, dialog_data):
        self._dialog_data = copy.deepcopy(dialog_data)
        self.le_e_calib_a0.setText(self._format_float(self._dialog_data["e_offset"]["value"]))
        self.le_e_calib_a1.setText(self._format_float(self._dialog_data["e_linear"]["value"]))
        self.le_e_calib_a2.setText(self._format_float(self._dialog_data["e_quadratic"]["value"]))
        self.le_fwhm_b1.setText(self._format_float(self._dialog_data["fwhm_offset"]["value"]))
        self.le_fwhm_b2.setText(self._format_float(self._dialog_data["fwhm_fanoprime"]["value"]))
        self.le_incident_energy.setText(self._format_float(self._dialog_data["coherent_sct_energy"]["value"]))
        self.le_range_low.setText(self._format_float(self._dialog_data["energy_bound_low"]["value"]))
        self.le_range_high.setText(self._format_float(self._dialog_data["energy_bound_high"]["value"]))

    def get_dialog_data(self):
        return self._dialog_data

    def pb_find_elements_clicked(self):
        self.find_elements_requested = True

    def _read_le_value(self, line_edit, value_ref):
        value_ref["value"] = float(line_edit.text())

    def le_e_calib_a0_editing_finished(self):
        self._read_le_value(self.le_e_calib_a0, self._dialog_data["e_offset"])

    def le_e_calib_a1_editing_finished(self):
        self._read_le_value(self.le_e_calib_a1, self._dialog_data["e_linear"])

    def le_e_calib_a2_editing_finished(self):
        self._read_le_value(self.le_e_calib_a2, self._dialog_data["e_quadratic"])

    def le_fwhm_b1_editing_finished(self):
        self._read_le_value(self.le_fwhm_b1, self._dialog_data["fwhm_offset"])

    def le_fwhm_b2_editing_finished(self):
        self._read_le_value(self.le_fwhm_b2, self._dialog_data["fwhm_fanoprime"])

    def le_incident_energy_editing_finished(self):
        self._read_le_value(self.le_incident_energy, self._dialog_data["coherent_sct_energy"])
        self._read_le_value(self.le_range_high, self._dialog_data["energy_bound_high"])

    def le_range_low_editing_finished(self):
        self._read_le_value(self.le_range_low, self._dialog_data["energy_bound_low"])

    def le_range_high_editing_finished(self):
        self._read_le_value(self.le_range_high, self._dialog_data["energy_bound_high"])

    def _validate_as_float(self, text):
        return self.validator.validate(text, 0)[0] == QDoubleValidator.Acceptable

    def _validate_text(self, line_edit, text):
        line_edit.setValid(self._validate_as_float(text))
        self._update_exit_buttons_states()

    def _update_exit_buttons_states(self):
        if (
            self.le_e_calib_a0.isValid()
            and self.le_e_calib_a1.isValid()
            and self.le_e_calib_a2.isValid()
            and self.le_fwhm_b1.isValid()
            and self.le_fwhm_b2.isValid()
            and self.le_incident_energy.isValid()
            and self.le_range_low.isValid()
            and self.le_range_high.isValid()
        ):
            all_valid = True
        else:
            all_valid = False

        self.pb_find_elements.setEnabled(all_valid)
        self.pb_apply_settings.setEnabled(all_valid)

    def _range_high_update(self, text, margin=0.8):
        """Update the range 'high' limit based on incident energy"""
        v = float(text) + margin
        self.le_range_high.setText(self._format_float(v))

    def le_e_calib_a0_text_changed(self, text):
        self._validate_text(self.le_e_calib_a0, text)

    def le_e_calib_a1_text_changed(self, text):
        self._validate_text(self.le_e_calib_a1, text)

    def le_e_calib_a2_text_changed(self, text):
        self._validate_text(self.le_e_calib_a2, text)

    def le_fwhm_b1_text_changed(self, text):
        self._validate_text(self.le_fwhm_b1, text)

    def le_fwhm_b2_text_changed(self, text):
        self._validate_text(self.le_fwhm_b2, text)

    def le_incident_energy_text_changed(self, text):
        if self._validate_as_float(text) and float(text) > 0:
            self.le_incident_energy.setValid(True)
            self._range_high_update(text)
        else:
            self.le_incident_energy.setValid(False)
        self._update_exit_buttons_states()

    def le_range_low_text_changed(self, text):
        text_valid = False
        if self._validate_as_float(text):
            v = float(text)
            if 0 <= v < self._dialog_data["energy_bound_high"]["value"]:
                text_valid = True
        self.le_range_low.setValid(text_valid)
        self.le_range_high.setValid(text_valid)
        self._update_exit_buttons_states()

    def le_range_high_text_changed(self, text):
        text_valid = False
        if self._validate_as_float(text):
            v = float(text)
            if v > self._dialog_data["energy_bound_low"]["value"]:
                text_valid = True
        self.le_range_high.setValid(text_valid)
        self.le_range_low.setValid(text_valid)
        self._update_exit_buttons_states()

    def _recover_last_valid_le_text(self, line_edit, last_value):
        if not self._validate_as_float(line_edit.text()):
            line_edit.setText(self._format_float(last_value["value"]))
            return False
        else:
            return True

    def le_e_calib_a0_focus_out(self):
        self._recover_last_valid_le_text(self.le_e_calib_a0, self._dialog_data["e_offset"])

    def le_e_calib_a1_focus_out(self):
        self._recover_last_valid_le_text(self.le_e_calib_a1, self._dialog_data["e_linear"])

    def le_e_calib_a2_focus_out(self):
        self._recover_last_valid_le_text(self.le_e_calib_a2, self._dialog_data["e_quadratic"])

    def le_fwhm_b1_focus_out(self):
        self._recover_last_valid_le_text(self.le_fwhm_b1, self._dialog_data["fwhm_offset"])

    def le_fwhm_b2_focus_out(self):
        self._recover_last_valid_le_text(self.le_fwhm_b2, self._dialog_data["fwhm_fanoprime"])

    def le_incident_energy_focus_out(self):
        if not self._recover_last_valid_le_text(self.le_incident_energy, self._dialog_data["coherent_sct_energy"]):
            self.le_range_high.setText(self._format_float(self._dialog_data["energy_bound_high"]["value"]))

    def le_range_low_focus_out(self):
        self._recover_last_valid_le_text(self.le_range_low, self._dialog_data["energy_bound_low"])

    def le_range_high_focus_out(self):
        self._recover_last_valid_le_text(self.le_range_high, self._dialog_data["energy_bound_high"])

    def _reset_le_to_default(self, line_edit, data_ref):
        data_ref["value"] = data_ref["default"]
        line_edit.setText(self._format_float(data_ref["value"]))

    def pb_e_calib_a0_default_clicked(self):
        self._reset_le_to_default(self.le_e_calib_a0, self._dialog_data["e_offset"])

    def pb_e_calib_a1_default_clicked(self):
        self._reset_le_to_default(self.le_e_calib_a1, self._dialog_data["e_linear"])

    def pb_e_calib_a2_default_clicked(self):
        self._reset_le_to_default(self.le_e_calib_a2, self._dialog_data["e_quadratic"])

    def pb_fwhm_b1_default_clicked(self):
        self._reset_le_to_default(self.le_fwhm_b1, self._dialog_data["fwhm_offset"])

    def pb_fwhm_b2_default_clicked(self):
        self._reset_le_to_default(self.le_fwhm_b2, self._dialog_data["fwhm_fanoprime"])

    def pb_incident_energy_default_clicked(self):
        self._reset_le_to_default(self.le_incident_energy, self._dialog_data["coherent_sct_energy"])

    def pb_range_low_default_clicked(self):
        self._reset_le_to_default(self.le_range_low, self._dialog_data["energy_bound_low"])

    def pb_range_high_default_clicked(self):
        self._reset_le_to_default(self.le_range_high, self._dialog_data["energy_bound_high"])
