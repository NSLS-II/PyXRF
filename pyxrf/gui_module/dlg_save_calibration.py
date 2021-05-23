import os

from qtpy.QtWidgets import (
    QHBoxLayout,
    QVBoxLayout,
    QCheckBox,
    QLabel,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QTextEdit,
)
from qtpy.QtGui import QDoubleValidator
from qtpy.QtCore import Qt

from .useful_widgets import (
    LineEditReadOnly,
    PushButtonMinimumWidth,
    set_tooltip,
    LineEditExtended,
    DoubleValidatorStrict,
)

import logging

logger = logging.getLogger(__name__)


class DialogSaveCalibration(QDialog):
    def __init__(self, parent=None, *, file_path=None):

        super().__init__(parent)

        self.__file_path = ""
        self.__distance_to_sample = 0.0
        self.__overwrite_existing = False
        self.__preview = ("", {})  # str - information, dict - warnings

        self.setWindowTitle("Save Quantitative Calibration")
        self.setMinimumHeight(600)
        self.setMinimumWidth(600)
        self.resize(600, 600)

        self.text_edit = QTextEdit()
        set_tooltip(
            self.text_edit,
            "Preview the <b>quantitative calibration data</b> to be saved. The displayed "
            "warnings will not be saved, but need to be addressed in order to keep "
            "data integrity. The parameter <b>distance-to-sample</b> is optional, "
            "but desirable. If <b>distance-to-sample</b> is zero then no scaling will be "
            "applied to data to compensate for changing distance.",
        )
        self.text_edit.setReadOnly(True)

        self.le_file_path = LineEditReadOnly()
        set_tooltip(
            self.le_file_path,
            "Full path to the file used to <b>save the calibration data</b>. The path "
            "can be changed in file selection dialog box.",
        )
        self.pb_file_path = PushButtonMinimumWidth("..")
        set_tooltip(self.pb_file_path, "Change <b>file path</b> for saving the calibration data.")
        self.pb_file_path.clicked.connect(self.pb_file_path_clicked)
        self.pb_file_path.setDefault(False)
        self.pb_file_path.setAutoDefault(False)

        self.le_distance_to_sample = LineEditExtended()
        self.le_distance_to_sample.textChanged.connect(self.le_distance_to_sample_text_changed)
        self.le_distance_to_sample.editingFinished.connect(self.le_distance_to_sample_editing_finished)
        self._le_distance_to_sample_validator = DoubleValidatorStrict()
        self._le_distance_to_sample_validator.setBottom(0.0)
        set_tooltip(
            self.le_distance_to_sample,
            "<b>Distance</b> between the detector and the sample during calibration. If the value "
            "is 0, then no scaling is applied to data to correct the data if distance-to-sample "
            "is changed between calibration and measurement.",
        )

        self.cb_overwrite = QCheckBox("Overwrite Existing")
        self.cb_overwrite.stateChanged.connect(self.cb_overwrite_state_changed)
        set_tooltip(
            self.cb_overwrite,
            "Overwrite the <b>existing</b> file. This is a safety feature implemented to protect "
            "valuable results from accidental deletion.",
        )

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
        self.pb_ok = button_box.button(QDialogButtonBox.Ok)
        self.pb_ok.setDefault(False)
        self.pb_ok.setAutoDefault(False)
        self.pb_cancel = button_box.button(QDialogButtonBox.Cancel)
        self.pb_cancel.setDefault(True)
        self.pb_cancel.setAutoDefault(True)

        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        vbox.addWidget(button_box)

        self.setLayout(vbox)

        self._show_distance_to_sample()
        self._show_preview()
        self._show_overwrite_existing()

        # Set and display file path
        if file_path is not None:
            self.file_path = file_path

    @property
    def file_path(self):
        return self.__file_path

    @file_path.setter
    def file_path(self, file_path):
        file_path = os.path.expanduser(file_path)
        self.__file_path = file_path
        self.le_file_path.setText(file_path)

    @property
    def distance_to_sample(self):
        return self.__distance_to_sample

    @distance_to_sample.setter
    def distance_to_sample(self, distance_to_sample):
        self.__distance_to_sample = distance_to_sample
        self._show_distance_to_sample()

    @property
    def overwrite_existing(self):
        return self.__overwrite_existing

    @overwrite_existing.setter
    def overwrite_existing(self, overwrite_existing):
        self.__overwrite_existing = overwrite_existing
        self._show_overwrite_existing()

    @property
    def preview(self):
        return self.__preview

    @preview.setter
    def preview(self, preview):
        self.__preview = preview
        self._show_preview()

    def pb_file_path_clicked(self):
        file_path = QFileDialog.getSaveFileName(
            self,
            "Select File to Save Quantitative Calibration",
            self.file_path,
            "JSON (*.json);; All (*)",
            options=QFileDialog.DontConfirmOverwrite,
        )
        file_path = file_path[0]
        if file_path:
            self.file_path = file_path

    def le_distance_to_sample_text_changed(self, text):
        valid = self._le_distance_to_sample_validator.validate(text, 0)[0] == QDoubleValidator.Acceptable
        self.le_distance_to_sample.setValid(valid)
        self.pb_ok.setEnabled(valid)

    def le_distance_to_sample_editing_finished(self):
        text = self.le_distance_to_sample.text()
        if self._le_distance_to_sample_validator.validate(text, 0)[0] == QDoubleValidator.Acceptable:
            self.__distance_to_sample = float(text)
        self._show_distance_to_sample()
        self._show_preview()  # Show/hide warning on zero distance-to-sample value

    def cb_overwrite_state_changed(self, state):
        state = state == Qt.Checked
        self.__overwrite_existing = state

    def _show_distance_to_sample(self):
        self.le_distance_to_sample.setText(f"{self.__distance_to_sample:.10g}")

    def _show_overwrite_existing(self):
        self.cb_overwrite.setChecked(Qt.Checked if self.__overwrite_existing else Qt.Unchecked)

    def _show_preview(self):
        text = ""
        # First print warnings
        for key, value in self.__preview[1].items():
            if "distance" in key and self.__distance_to_sample > 0:
                continue
            text += value + "\n"
        # Additional space if there are any warinings
        if len(text):
            text += "\n"
        # Then add the main block of text
        text += self.__preview[0]
        self.text_edit.setText(text)
