from qtpy.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QRadioButton, QButtonGroup, QComboBox
from qtpy.QtCore import Slot

from .useful_widgets import set_tooltip, global_gui_variables
from ..model.lineplot import PlotTypes, EnergyRangePresets

from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)

import logging

logger = logging.getLogger(__name__)


class PreviewPlotSpectrum(QWidget):
    def __init__(self, *, gpc, gui_vars):
        super().__init__()

        # Global processing classes
        self.gpc = gpc
        # Global GUI variables (used for control of GUI state)
        self.gui_vars = gui_vars

        self.cb_plot_type = QComboBox()
        self.cb_plot_type.addItems(["LinLog", "Linear"])
        self.cb_plot_type.setCurrentIndex(self.gpc.get_preview_plot_type())
        self.cb_plot_type.currentIndexChanged.connect(self.cb_plot_type_current_index_changed)

        self.rb_selected_region = QRadioButton("Selected region")
        self.rb_selected_region.setChecked(True)
        self.rb_full_spectrum = QRadioButton("Full spectrum")
        if self.gpc.get_preview_energy_range() == EnergyRangePresets.SELECTED_RANGE:
            self.rb_selected_region.setChecked(True)
        elif self.gpc.get_preview_energy_range() == EnergyRangePresets.FULL_SPECTRUM:
            self.rb_full_spectrum.setChecked(True)
        else:
            logger.error(
                "Spectrum preview: incorrect Enum value for energy range was used:\n"
                "    Report the error to the development team."
            )

        self.btn_group_region = QButtonGroup()
        self.btn_group_region.addButton(self.rb_selected_region)
        self.btn_group_region.addButton(self.rb_full_spectrum)
        self.btn_group_region.buttonToggled.connect(self.btn_group_region_button_toggled)

        self.mpl_canvas = FigureCanvas(self.gpc.plot_model._fig_preview)
        self.mpl_toolbar = NavigationToolbar(self.mpl_canvas, self)

        # Keep layout without change when canvas is hidden (invisible)
        sp_retain = self.mpl_canvas.sizePolicy()
        sp_retain.setRetainSizeWhenHidden(True)
        self.mpl_canvas.setSizePolicy(sp_retain)

        vbox = QVBoxLayout()

        hbox = QHBoxLayout()
        hbox.addWidget(self.cb_plot_type)
        hbox.addStretch(1)
        hbox.addWidget(self.rb_selected_region)
        hbox.addWidget(self.rb_full_spectrum)
        vbox.addLayout(hbox)

        vbox.addWidget(self.mpl_toolbar)
        vbox.addWidget(self.mpl_canvas)
        self.setLayout(vbox)

        self._set_tooltips()

    def _set_tooltips(self):
        set_tooltip(self.cb_plot_type, "Use <b>Linear</b> or <b>LinLog</b> axes to plot spectra")
        set_tooltip(
            self.rb_selected_region,
            "Plot spectrum in the <b>selected range</b> of energies. The range may be set "
            "in the 'Model' tab. Click the button <b>'Find Automatically ...'</b> "
            "to set the range of energies before finding the emission lines. The range "
            "may be changed in General Settings dialog (button <b>'General ...'</b>) at any time.",
        )
        set_tooltip(self.rb_full_spectrum, "Plot full spectrum over <b>all available eneriges</b>.")

    def update_widget_state(self, condition=None):
        if condition == "tooltips":
            self._set_tooltips()
        self.mpl_toolbar.setVisible(self.gui_vars["show_matplotlib_toolbar"])

        # Hide Matplotlib canvas during computations
        state_compute = global_gui_variables["gui_state"]["running_computations"]
        self.mpl_canvas.setVisible(not state_compute)

    @Slot()
    @Slot(bool)
    def redraw_preview_plot(self):
        # It is assumed that the plot is visible
        self.gpc.update_preview_spectrum_plot()

    def btn_group_region_button_toggled(self, button, checked):
        if checked:
            if button == self.rb_selected_region:
                self.gpc.set_preview_energy_range(EnergyRangePresets.SELECTED_RANGE)
                self.gpc.update_preview_spectrum_plot()
                logger.debug("GUI: Display only selected region")
            elif button == self.rb_full_spectrum:
                self.gpc.set_preview_energy_range(EnergyRangePresets.FULL_SPECTRUM)
                self.gpc.update_preview_spectrum_plot()
                logger.debug("GUI: Display full spectrum")
            else:
                logger.error(
                    "Spectrum preview: unknown button was toggled. "
                    "Please, report the error to the development team."
                )

    def cb_plot_type_current_index_changed(self, index):
        try:
            self.gpc.set_preview_plot_type(PlotTypes(index))
            self.gpc.plot_model.update_preview_spectrum_plot()
        except ValueError:
            logger.error(
                "Spectrum preview: incorrect index for energy range preset was detected.\n"
                "Please report the error to the development team."
            )
