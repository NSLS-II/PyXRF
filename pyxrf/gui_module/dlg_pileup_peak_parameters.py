from qtpy.QtWidgets import QVBoxLayout, QLabel, QDialog, QDialogButtonBox, QGridLayout
from qtpy.QtGui import QDoubleValidator, QRegExpValidator
from qtpy.QtCore import QRegExp

from .useful_widgets import LineEditReadOnly, set_tooltip, LineEditExtended

import logging

logger = logging.getLogger(__name__)


class DialogPileupPeakParameters(QDialog):
    def __init__(self, parent=None):

        super().__init__(parent)

        self._data = {"element1": "", "element2": "", "energy": 0}
        # Reference to function that computes pileup energy based on two emission lines
        self._compute_energy = None

        self.setWindowTitle("Pileup Peak Parameters")

        self.le_element1 = LineEditExtended()
        set_tooltip(self.le_element1, "The <b>name</b> of the emission line #1")
        self.le_element2 = LineEditExtended()
        set_tooltip(self.le_element2, "The <b>name</b> of the emission line #2")
        self.peak_energy = LineEditReadOnly()
        set_tooltip(
            self.peak_energy,
            "The <b>energy</b> (location) of the pileup peak center. The energy can not"
            "be edited: it is set based on the selected emission lines",
        )

        self._validator_eline = QRegExpValidator()
        # TODO: the following regex is too broad: [a-z] should be narrowed down
        self._validator_eline.setRegExp(QRegExp(r"^[A-Z][a-z]?_[KLM][a-z]\d$"))

        instructions = QLabel("Specify two emission lines, e.g. Si_Ka1 and Fe_Ka1")

        grid = QGridLayout()
        grid.addWidget(QLabel("Emission line 1:"), 0, 0)
        grid.addWidget(self.le_element1, 0, 1)
        grid.addWidget(QLabel("Emission line 2:"), 1, 0)
        grid.addWidget(self.le_element2, 1, 1)
        grid.addWidget(QLabel("Peak energy, keV:"), 2, 0)
        grid.addWidget(self.peak_energy, 2, 1)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.button(QDialogButtonBox.Cancel).setDefault(True)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        self.pb_ok = button_box.button(QDialogButtonBox.Ok)

        vbox = QVBoxLayout()
        vbox.addWidget(instructions)
        vbox.addSpacing(10)
        vbox.addLayout(grid)
        vbox.addWidget(button_box)

        self.setLayout(vbox)

        self.le_element1.editingFinished.connect(self.le_element1_editing_finished)
        self.le_element2.editingFinished.connect(self.le_element2_editing_finished)

        self.le_element1.textChanged.connect(self.le_element1_text_changed)
        self.le_element2.textChanged.connect(self.le_element2_text_changed)

    def set_compute_energy_function(self, func):
        self._compute_energy = func

    def set_parameters(self, data):
        self._data = data.copy()
        self._show_data()

    def get_parameters(self):
        return self._data

    def _format_float(self, v):
        return f"{v:.12g}"

    def _show_data(self):
        self.le_element1.setText(self._data["element1"])
        self.le_element2.setText(self._data["element2"])
        self._show_energy()

    def _show_energy(self, energy=None):
        if energy is None:
            energy = self._data["energy"]
        text = self._format_float(energy)
        self.peak_energy.setText(text)
        self._validate_all()

    def _validate_le_element1(self, text=None):
        return self._validate_eline(self.le_element1, text)

    def _validate_le_element2(self, text=None):
        return self._validate_eline(self.le_element2, text)

    def _validate_peak_energy(self, energy=None):
        # Peak energy is not edited, so it is always has a valid floating point number
        #   The only problem is that it may be out of range.
        if energy is None:
            energy = self._data["energy"]
        valid = self._data["range_low"] < energy < self._data["range_high"]
        self.peak_energy.setValid(valid)
        return valid

    def _validate_all(self):
        valid = self._validate_le_element1() and self._validate_le_element2()
        # Energy doesn't influence the success of validation, but the Ok button
        #   should be disabled if energy if out of range.
        valid_energy = self._validate_peak_energy()
        self.pb_ok.setEnabled(valid and valid_energy)
        return valid

    def _validate_eline(self, le_widget, text):
        if text is None:
            text = le_widget.text()

        valid = True
        if self._validator_eline.validate(text, 0)[0] != QDoubleValidator.Acceptable:
            valid = False
        else:
            # Try to compute energy for the pileup peak
            if self._compute_energy(text, text) == 0:
                valid = False
        le_widget.setValid(valid)
        return valid

    def _refresh_energy(self):
        eline1 = self._data["element1"]
        eline2 = self._data["element2"]
        try:
            energy = self._compute_energy(eline1, eline2)
            self._data["energy"] = energy
            self._show_energy()
        except Exception:
            self._data["energy"] = 0
            self._show_energy()

    def le_element1_editing_finished(self):
        if self._validate_le_element1():
            self._data["element1"] = self.le_element1.text()
            self._refresh_energy()
        else:
            self.le_element1.setText(self._data["element1"])
        self._validate_all()

    def le_element2_editing_finished(self):
        if self._validate_le_element2():
            self._data["element2"] = self.le_element2.text()
            self._refresh_energy()
        else:
            self.le_element2.setText(self._data["element2"])
        self._validate_all()

    def le_element1_text_changed(self, text):
        if self._validate_all():
            self._data["element1"] = self.le_element1.text()
            print(f"element1 - {self._data['element1']}")
            self._refresh_energy()

    def le_element2_text_changed(self, text):
        if self._validate_all():
            self._data["element2"] = self.le_element2.text()
            print(f"element2 - {self._data['element2']}")
            self._refresh_energy()
