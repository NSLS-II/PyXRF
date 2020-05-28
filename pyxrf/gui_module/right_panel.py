from PyQt5.QtWidgets import QTabWidget

from .wd_plots_preview import PreviewPlots
from .wd_plots_fitting_model import PlotFittingModel
from .wd_plots_xrf_maps import PlotXrfMaps
from .wd_plots_rgb_maps import PlotRgbMaps


class RightPanel(QTabWidget):

    def __init__(self):
        super().__init__()

        self.addTab(PreviewPlots(), "Preview")
        self.addTab(PlotFittingModel(), "Fitting Model")
        self.addTab(PlotXrfMaps(), "XRF Maps")
        self.addTab(PlotRgbMaps(), "RGB")
