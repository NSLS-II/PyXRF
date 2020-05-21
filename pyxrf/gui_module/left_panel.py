from PyQt5.QtWidgets import QTabWidget, QScrollArea

from .form_base_widget import FormBaseWidget
from .wd_load_data import LoadDataWidget
from .wd_model import ModelWidget
from .wd_fit_maps import FitMapsWidget


class LeftPanel(QTabWidget):

    def __init__(self):
        super().__init__()

        self.setTabPosition(QTabWidget.West)

        _scroll = QScrollArea()
        _scroll.setWidget(LoadDataWidget())
        self.addTab(_scroll, "Data")

        _scroll = QScrollArea()
        _scroll.setWidget(ModelWidget())
        self.addTab(_scroll, "Model")

        _scroll = QScrollArea()
        _scroll.setWidget(FitMapsWidget())
        self.addTab(_scroll, "Maps")
