from qtpy.QtWidgets import QVBoxLayout, QLabel, QDialog, QDialogButtonBox, QGridLayout

from .useful_widgets import LineEditReadOnly, set_tooltip

import logging

logger = logging.getLogger(__name__)


class DialogNewUserPeak(QDialog):
    def __init__(self, parent=None):

        super().__init__(parent)

        self._data = {"name": "", "energy": 0}

        self.setWindowTitle("Add New User-Defined Peak")
        self.setMinimumWidth(400)

        self.le_name = LineEditReadOnly()
        set_tooltip(self.le_name, "<b>Name</b> of the user-defined peak.")
        self.le_energy = LineEditReadOnly()
        set_tooltip(self.le_energy, "<b>Energy</b> (keV) of the center of the user-defined peak.")

        vbox = QVBoxLayout()

        grid = QGridLayout()
        grid.addWidget(QLabel("Peak name:"), 0, 0)
        grid.addWidget(self.le_name, 0, 1)
        grid.addWidget(QLabel("Energy, keV"), 1, 0)
        grid.addWidget(self.le_energy, 1, 1)
        vbox.addLayout(grid)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.button(QDialogButtonBox.Cancel).setDefault(True)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        vbox.addWidget(button_box)

        self.setLayout(vbox)

    def set_parameters(self, data):
        self._data = data.copy()
        self._show_data()

    def _format_float(self, v):
        return f"{v:.12g}"

    def _show_data(self, existing=None):
        self.le_name.setText(self._data["name"])
        self.le_energy.setText(self._format_float(self._data["energy"]))
