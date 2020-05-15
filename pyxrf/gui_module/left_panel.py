from PyQt5.QtWidgets import QTabWidget, QScrollArea

from .form_base_widget import FormBaseWidget
from .wd_load_data import LoadDataWidget

class LeftPanel(QTabWidget):

    def __init__(self):
        super().__init__()

        self.setTabPosition(QTabWidget.West)

        _scroll = QScrollArea()
        _scroll.setWidget(LoadDataWidget())
        self.addTab(_scroll, "Load")

        _scroll = QScrollArea()
        _scroll.setWidget(FormBaseWidget())
        self.addTab(_scroll, "Model")

        _scroll = QScrollArea()
        _scroll.setWidget(FormBaseWidget())
        self.addTab(_scroll, "Fit")
