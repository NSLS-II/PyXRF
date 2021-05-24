from qtpy.QtWidgets import QHBoxLayout, QVBoxLayout, QLabel, QDialog, QDialogButtonBox, QRadioButton, QButtonGroup
from qtpy.QtGui import QIntValidator, QRegExpValidator
from qtpy.QtCore import QRegExp

from .useful_widgets import LineEditExtended, set_tooltip, IntValidatorStrict

import logging

logger = logging.getLogger(__name__)


class DialogSelectScan(QDialog):
    def __init__(self):

        super().__init__()

        self.resize(400, 200)
        self.setWindowTitle("Load Run From Database")

        self._id_uid = None
        self._mode_id_uid = "id"

        label = QLabel("Enter run ID or UID:")
        self.le_id_uid = LineEditExtended()
        self.le_id_uid.textChanged.connect(self.le_id_uid_text_changed)
        self.le_id_uid.editingFinished.connect(self.le_id_uid_editing_finished)
        set_tooltip(self.le_id_uid, "Enter <b>Run ID</b> or <b>Run UID</b>.")

        self._validator_id = IntValidatorStrict()
        # Short UID example: "04c9afa7"
        self._validator_uid_short = QRegExpValidator(QRegExp(r"[0-9a-f]{8}"))
        # Full UID example: "04c9afa7-a43a-4af1-8e55-2034384d4a77"
        self._validator_uid_full = QRegExpValidator(
            QRegExp(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
        )

        self.rb_id = QRadioButton("Run ID")
        set_tooltip(self.rb_id, "The value in the line edit box is <b>Run ID</b> (e.g. <b>34235</b> or <b>-1</b>)")
        self.rb_id.setChecked(self._mode_id_uid == "id")
        self.rb_uid = QRadioButton("Run UID")
        self.rb_uid.setChecked(self._mode_id_uid == "uid")
        set_tooltip(
            self.rb_uid,
            "The value in the line edit box is <b>Run UID</b> "
            "(e.g. <b>04c9afb7-a43a-4af1-8e55-2034384d4a77</b> or <b>04c9afb7</b>)",
        )

        self.btn_group = QButtonGroup()
        self.btn_group.addButton(self.rb_id)
        self.btn_group.addButton(self.rb_uid)
        self.btn_group.buttonToggled.connect(self.btn_group_button_toggled)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        self.pb_ok = button_box.button(QDialogButtonBox.Ok)

        vbox = QVBoxLayout()

        vbox.addStretch(1)

        hbox = QHBoxLayout()
        hbox.addWidget(label)
        hbox.addStretch(1)
        vbox.addLayout(hbox)
        vbox.addWidget(self.le_id_uid)

        hbox = QHBoxLayout()
        hbox.addStretch(1)
        hbox.addWidget(self.rb_id)
        hbox.addStretch(1)
        hbox.addWidget(self.rb_uid)
        hbox.addStretch(1)
        vbox.addLayout(hbox)

        vbox.addStretch(1)
        vbox.addWidget(button_box)
        self.setLayout(vbox)

        # This is how the button from QDialogButtonBox can be disabled
        # button_box.button(QDialogButtonBox.Ok).setEnabled(False)

        self.le_id_uid.setText("")
        self._validate_id_uid()

    def btn_group_button_toggled(self, button, state):
        if state:
            if button == self.rb_id:
                self._mode_id_uid = "id"
            elif button == self.rb_uid:
                self._mode_id_uid = "uid"

            text = self.le_id_uid.text()
            if self._validate_id_uid(text):
                self._read_id_uid(text)

    def le_id_uid_text_changed(self, text):
        self._validate_id_uid(text)

    def le_id_uid_editing_finished(self):
        text = self.le_id_uid.text()
        if self._validate_id_uid(text):
            self._read_id_uid(text)

    def get_id_uid(self):
        """
        Read the selected scan ID or UID

        Returns
        -------
        (str, int) or (str, str)
            keyword "id" or "uid" depending on whether the second element is scan ID or UID,
            scan ID (int), scan UID (str) or None if no scan ID or UID is selected
        """
        return self._mode_id_uid, self._id_uid

    def _validate_id_uid(self, text=None):
        valid = False
        if text is None:
            text = self.le_id_uid.text()
        if self._mode_id_uid == "id":
            if self._validator_id.validate(text, 0)[0] == QIntValidator.Acceptable:
                valid = True
        elif self._mode_id_uid == "uid":
            if (
                self._validator_uid_short.validate(text, 0)[0] == QIntValidator.Acceptable
                or self._validator_uid_full.validate(text, 0)[0] == QIntValidator.Acceptable
            ):
                valid = True

        self.le_id_uid.setValid(valid)
        self.pb_ok.setEnabled(valid)

        return valid

    def _read_id_uid(self, text):
        # It is assumed that the entered text is valid for the selected mode
        if text is None:
            text = self.le_id_uid.text()
        if self._mode_id_uid == "id":
            self._id_uid = int(text)
        elif self._mode_id_uid == "uid":
            self._id_uid = text

    def _set_mode(self, mode):
        if mode == "id":
            self._mode_id_uid = mode
            self.rb_id.setChecked(True)
        elif mode == "uid":
            self._mode_id_uid = mode
            self.rb_uid.setChecked(True)
