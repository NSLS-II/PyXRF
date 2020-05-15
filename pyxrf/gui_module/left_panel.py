from PyQt5.QtWidgets import QTabWidget
from .form_base_widget import FormBaseWidget


class LeftPanel(QTabWidget):

    def __init__(self):
        super().__init__()

        self.setTabPosition(QTabWidget.West)

        self.addTab(FormBaseWidget(), "Left 1")
        self.addTab(FormBaseWidget(), "Left 2")
