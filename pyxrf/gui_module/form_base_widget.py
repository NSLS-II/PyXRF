from PyQt5.QtWidgets import QWidget


class FormBaseWidget(QWidget):

    def __init__(self):
        super().__init__()
        self.setMinimumSize(400, 100)
