from PyQt5.QtWidgets import QTabWidget
from .form_base_widget import FormBaseWidget


class RightPanel(QTabWidget):

    def __init__(self):
        super().__init__()

        self.addTab(FormBaseWidget(), "Right 1")
        self.addTab(FormBaseWidget(), "Right 2")
