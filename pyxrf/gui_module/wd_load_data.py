from PyQt5.QtWidgets import QPushButton, QHBoxLayout, QVBoxLayout, QWidget

from .form_base_widget import FormBaseWidget


class LoadDataWidget(FormBaseWidget):

    def __init__(self):
        super().__init__()
        self.initialize()

    def initialize(self):

        vbox = QVBoxLayout()

        for i in range(30):
            btn_text = f"Button {i}"
            btn = QPushButton(btn_text)
            vbox.addWidget(btn)

        vbox.addStretch(1)

        self.setLayout(vbox)
