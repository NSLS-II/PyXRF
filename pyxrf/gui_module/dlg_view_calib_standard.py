import textwrap

from qtpy.QtWidgets import QHBoxLayout, QVBoxLayout, QDialog, QDialogButtonBox, QTextEdit, QLabel
from qtpy.QtCore import Qt

import logging

logger = logging.getLogger(__name__)


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
