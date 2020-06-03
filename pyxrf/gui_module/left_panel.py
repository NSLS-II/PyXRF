from PyQt5.QtWidgets import QTabWidget, QScrollArea
from PyQt5.QtCore import pyqtSlot

from .form_base_widget import FormBaseWidget
from .wd_load_data import LoadDataWidget
from .wd_model import ModelWidget
from .wd_fit_maps import FitMapsWidget
from .useful_widgets import global_gui_variables


class LeftPanel(QTabWidget):

    def __init__(self):
        super().__init__()

        self.setTabPosition(QTabWidget.West)

        self.load_data_widget = LoadDataWidget()
        _scroll = QScrollArea()
        _scroll.setWidget(self.load_data_widget)
        self.addTab(_scroll, "Data")

        self.model_widget = ModelWidget()
        _scroll = QScrollArea()
        _scroll.setWidget(self.model_widget)
        self.addTab(_scroll, "Model")

        self.fit_maps_widget = FitMapsWidget()
        _scroll = QScrollArea()
        _scroll.setWidget(self.fit_maps_widget)
        self.addTab(_scroll, "Maps")

        self.update_widget_state()

    def update_widget_state(self):
        # TODO: this function has to enable tabs and widgets based on the current program state
        state = not global_gui_variables["gui_state"]["running_computations"]
        for i in range(self.count()):
            if state or (i != self.currentIndex()):
                self.setTabEnabled(i, state)
            self.widget(i).setEnabled(state)

