from qtpy.QtWidgets import QVBoxLayout, QDialog, QDialogButtonBox, QTextEdit

import logging

logger = logging.getLogger(__name__)


class DialogViewMetadata(QDialog):
    def __init__(self):

        super().__init__()

        self.resize(500, 500)
        self.setWindowTitle("Run Metadata")

        self.te_meta = QTextEdit()
        self.te_meta.setReadOnly(True)

        # 'Close' button box
        button_box = QDialogButtonBox(QDialogButtonBox.Close)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        vbox = QVBoxLayout()
        vbox.addWidget(self.te_meta)
        vbox.addWidget(button_box)
        self.setLayout(vbox)

    def setText(self, text):
        self.te_meta.setText(text)
