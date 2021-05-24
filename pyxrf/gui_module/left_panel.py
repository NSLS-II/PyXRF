from qtpy.QtWidgets import QTabWidget, QScrollArea
from qtpy.QtCore import Slot

from .tab_wd_load_data import LoadDataWidget
from .tab_wd_model import ModelWidget
from .tab_wd_fit_maps import FitMapsWidget


class LeftPanel(QTabWidget):
    def __init__(self, *, gpc, gui_vars):
        super().__init__()

        # Global processing classes
        self.gpc = gpc
        # Global GUI variables (used for control of GUI state)
        self.gui_vars = gui_vars

        self.setTabPosition(QTabWidget.West)

        self.load_data_widget = LoadDataWidget(gpc=self.gpc, gui_vars=self.gui_vars)
        self.load_data_tab = QScrollArea()
        self.load_data_tab.setWidget(self.load_data_widget)
        self.addTab(self.load_data_tab, "Data")

        self.model_widget = ModelWidget(gpc=self.gpc, gui_vars=self.gui_vars)
        self.model_tab = QScrollArea()
        self.model_tab.setWidget(self.model_widget)
        self.addTab(self.model_tab, "Model")

        self.fit_maps_widget = FitMapsWidget(gpc=self.gpc, gui_vars=self.gui_vars)
        self.fit_maps_tab = QScrollArea()
        self.fit_maps_tab.setWidget(self.fit_maps_widget)
        self.addTab(self.fit_maps_tab, "Maps")

        self.update_widget_state()

    def update_widget_state(self, condition=None):
        # TODO: this function has to enable tabs and widgets based on the current program state
        state = not self.gui_vars["gui_state"]["running_computations"]
        for i in range(self.count()):
            if state or (i != self.currentIndex()):
                self.setTabEnabled(i, state)
            self.widget(i).setEnabled(state)

        # Propagate the function call downstream (since the actual tab widget is 'QScrollArea')
        self.load_data_widget.update_widget_state(condition)
        self.model_widget.update_widget_state(condition)
        self.fit_maps_widget.update_widget_state(condition)

    @Slot(bool)
    def slot_activate_load_data_tab(self):
        self.setCurrentWidget(self.load_data_tab)
