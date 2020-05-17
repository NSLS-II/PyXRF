from PyQt5.QtWidgets import QTabWidget
from .form_base_widget import FormBaseWidget


class RightPanel(QTabWidget):

    def __init__(self):
        super().__init__()

        self.addTab(FormBaseWidget(), "Preview")
        self.addTab(FormBaseWidget(), "Fitting Model")
        self.addTab(FormBaseWidget(), "XRF Maps")
        self.addTab(FormBaseWidget(), "RGB")
