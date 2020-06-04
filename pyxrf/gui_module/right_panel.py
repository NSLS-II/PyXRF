from PyQt5.QtWidgets import QTabWidget

from .wd_plots_preview import PreviewPlots
from .wd_plots_fitting_model import PlotFittingModel
from .wd_plots_xrf_maps import PlotXrfMaps
from .wd_plots_rgb_maps import PlotRgbMaps
from .useful_widgets import global_gui_variables


class RightPanel(QTabWidget):

    def __init__(self):
        super().__init__()

        self.addTab(PreviewPlots(), "Preview")
        self.addTab(PlotFittingModel(), "Fitting Model")
        self.addTab(PlotXrfMaps(), "XRF Maps")
        self.addTab(PlotRgbMaps(), "RGB")

    def update_widget_state(self):
        # TODO: this function has to enable tabs and widgets based on the current program state
        state = not global_gui_variables["gui_state"]["running_computations"]
        for i in range(self.count()):
            if state or (i != self.currentIndex()):
                self.setTabEnabled(i, state)
            self.widget(i).setEnabled(state)
