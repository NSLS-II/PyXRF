from qtpy.QtWidgets import QVBoxLayout, QLabel, QDialog, QDialogButtonBox, QGridLayout
from qtpy.QtGui import QDoubleValidator

from .useful_widgets import LineEditReadOnly, set_tooltip, LineEditExtended

import logging

logger = logging.getLogger(__name__)


class DialogEditUserPeakParameters(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self._data = {"name": "", "maxv": 0, "energy": 0, "fwhm": 0}

        self.setWindowTitle("Edit User-Defined Peak Parameters")
        self.setMinimumWidth(400)

        self.le_name = LineEditReadOnly()
        set_tooltip(self.le_name, "<b>Name</b> of the user-defined peak.")
        self.le_intensity = LineEditExtended()
        set_tooltip(
            self.le_intensity,
            "Peak <b>intensity</b>. Typically it is not necessary to set the "
            "intensity precisely, since it is refined during fitting of total "
            "spectrum.",
        )
        self.le_energy = LineEditExtended()
        set_tooltip(self.le_energy, "<b>Energy</b> (keV) of the center of the user-defined peak.")
        self.le_fwhm = LineEditExtended()
        set_tooltip(self.le_fwhm, "<b>FWHM</b> (in keV) of the user-defined peak.")

        self._validator = QDoubleValidator()

        vbox = QVBoxLayout()

        grid = QGridLayout()
        grid.addWidget(QLabel("Peak name:"), 0, 0)
        grid.addWidget(self.le_name, 0, 1)
        grid.addWidget(QLabel("Energy, keV"), 1, 0)
        grid.addWidget(self.le_energy, 1, 1)
        grid.addWidget(QLabel("Intensity:"), 2, 0)
        grid.addWidget(self.le_intensity, 2, 1)
        grid.addWidget(QLabel("FWHM, keV"), 3, 0)
        grid.addWidget(self.le_fwhm, 3, 1)
        vbox.addLayout(grid)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.button(QDialogButtonBox.Cancel).setDefault(True)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        self.pb_ok = button_box.button(QDialogButtonBox.Ok)

        vbox.addWidget(button_box)

        self.setLayout(vbox)

        self.le_intensity.editingFinished.connect(self.le_intensity_editing_finished)
        self.le_energy.editingFinished.connect(self.le_energy_editing_finished)
        self.le_fwhm.editingFinished.connect(self.le_fwhm_editing_finished)

        self.le_intensity.textChanged.connect(self.le_intensity_text_changed)
        self.le_energy.textChanged.connect(self.le_energy_text_changed)
        self.le_fwhm.textChanged.connect(self.le_fwhm_text_changed)

    def set_parameters(self, data):
        self._data = data.copy()
        self._show_data()

    def get_parameters(self):
        return self._data

    def _format_float(self, v):
        return f"{v:.12g}"

    def _show_data(self, existing=None):
        self.le_name.setText(self._data["name"])
        self.le_energy.setText(self._format_float(self._data["energy"]))
        self.le_intensity.setText(self._format_float(self._data["maxv"]))
        self.le_fwhm.setText(self._format_float(self._data["fwhm"]))

    # The following validation functions are identical, but they will be different,
    #   because the relationships
    def _validate_le_intensity(self, text=None):
        return self._validate_le(self.le_intensity, text, lambda v: v > 0)

    def _validate_le_energy(self, text=None):
        return self._validate_le(self.le_energy, text, lambda v: v > 0)

    def _validate_le_fwhm(self, text=None):
        return self._validate_le(self.le_fwhm, text, lambda v: v > 0)

    def _validate_all(self):
        valid = self._validate_le_intensity() and self._validate_le_energy() and self._validate_le_fwhm()
        self.pb_ok.setEnabled(valid)

    def _validate_le(self, le_widget, text, condition):
        if text is None:
            text = le_widget.text()
        valid = True
        if self._validator.validate(text, 0)[0] != QDoubleValidator.Acceptable:
            valid = False
        elif not condition(float(text)):
            valid = False
        le_widget.setValid(valid)
        return valid

    def le_intensity_editing_finished(self):
        if self._validate_le_intensity():
            self._data["maxv"] = float(self.le_intensity.text())
        else:
            self.le_intensity.setText(self._format_float(self._data["maxv"]))
        self._validate_all()

    def le_energy_editing_finished(self):
        if self._validate_le_energy():
            self._data["energy"] = float(self.le_energy.text())
        else:
            self.le_energy.setText(self._format_float(self._data["energy"]))
        self._validate_all()

    def le_fwhm_editing_finished(self):
        if self._validate_le_fwhm():
            self._data["fwhm"] = float(self.le_fwhm.text())
        else:
            self.le_fwhm.setText(self._format_float(self._data["fwhm"]))
        self._validate_all()

    def le_intensity_text_changed(self, text):
        self._validate_all()

    def le_energy_text_changed(self, text):
        self._validate_all()

    def le_fwhm_text_changed(self, text):
        self._validate_all()
