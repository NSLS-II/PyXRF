from qtpy.QtWidgets import QTabWidget
from qtpy.QtCore import Slot

from .wd_preview_plot_count import PreviewPlotCount
from .wd_preview_plot_spectrum import PreviewPlotSpectrum

import logging

logger = logging.getLogger(__name__)


class PreviewPlots(QTabWidget):
    def __init__(self, *, gpc, gui_vars):
        super().__init__()

        # Global processing classes
        self.gpc = gpc
        # Global GUI variables (used for control of GUI state)
        self.gui_vars = gui_vars

        self.preview_plot_spectrum = PreviewPlotSpectrum(gpc=self.gpc, gui_vars=self.gui_vars)
        self.addTab(self.preview_plot_spectrum, "Total Spectrum")
        self.preview_plot_count = PreviewPlotCount(gpc=self.gpc, gui_vars=self.gui_vars)
        self.addTab(self.preview_plot_count, "Total Count")

    def update_widget_state(self, condition=None):
        self.preview_plot_spectrum.update_widget_state(condition)
        self.preview_plot_count.update_widget_state(condition)

        state = self.gui_vars["gui_state"]["state_file_loaded"]
        for i in range(self.count()):
            self.setTabEnabled(i, state)

    @Slot(bool)
    def activate_preview_plot_spectrum(self):
        self.setCurrentWidget(self.preview_plot_spectrum)
