from PyQt5.QtWidgets import QTabWidget
from .form_base_widget import FormBaseWidget

from .wd_preview import PreviewPlots
from .wd_fitting_model import PlotFittingModel

class RightPanel(QTabWidget):

    def __init__(self):
        super().__init__()

        self.addTab(PreviewPlots(), "Preview")
        self.addTab(PlotFittingModel(), "Fitting Model")
        self.addTab(FormBaseWidget(), "XRF Maps")
        self.addTab(FormBaseWidget(), "RGB")
