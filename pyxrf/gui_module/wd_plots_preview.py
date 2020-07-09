from PyQt5.QtWidgets import (QWidget, QTabWidget, QLabel, QVBoxLayout, QHBoxLayout,
                             QRadioButton, QButtonGroup, QComboBox)
from PyQt5.QtCore import pyqtSlot

from .useful_widgets import RangeManager, set_tooltip
from ..model.lineplot import PlotTypes, EnergyRangePresets, MapTypes, MapAxesUnits

from matplotlib.backends.backend_qt5agg import \
    FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar

import logging
logger = logging.getLogger("pyxrf")


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
        if not state:
            # Select 'Plot spectrum' tab
            self.setCurrentIndex(0)
        for i in range(self.count()):
            self.setTabEnabled(i, state)


class PreviewPlotSpectrum(QWidget):

    def __init__(self, *, gpc, gui_vars):
        super().__init__()

        # Global processing classes
        self.gpc = gpc
        # Global GUI variables (used for control of GUI state)
        self.gui_vars = gui_vars

        self.cb_plot_type = QComboBox()
        self.cb_plot_type.addItems(["LinLog", "Linear"])
        self.cb_plot_type.setCurrentIndex(self.gpc.plot_model.plot_type_preview.value)
        self.cb_plot_type.currentIndexChanged.connect(self.cb_plot_type_current_index_changed)

        self.rb_selected_region = QRadioButton("Selected region")
        self.rb_selected_region.setChecked(True)
        self.rb_full_spectrum = QRadioButton("Full spectrum")
        if self.gpc.plot_model.energy_range_preview == EnergyRangePresets.SELECTED_RANGE:
            self.rb_selected_region.setChecked(True)
        elif self.gpc.plot_model.energy_range_preview == EnergyRangePresets.FULL_SPECTRUM:
            self.rb_full_spectrum.setChecked(True)
        else:
            logger.error("Spectrum preview: incorrect Enum value for energy range was used:\n"
                         "    Report the error to the development team.")

        self.btn_group_region = QButtonGroup()
        self.btn_group_region.addButton(self.rb_selected_region)
        self.btn_group_region.addButton(self.rb_full_spectrum)
        self.btn_group_region.buttonToggled.connect(self.btn_group_region_button_toggled)

        self.mpl_canvas = FigureCanvas(self.gpc.plot_model._fig_preview)
        self.mpl_toolbar = NavigationToolbar(self.mpl_canvas, self)

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
        set_tooltip(self.cb_plot_type,
                    "Use <b>Linear</b> or <b>LinLog</b> axes to plot spectra")
        set_tooltip(
            self.rb_selected_region,
            "Plot spectrum in the <b>selected range</b> of energies. The range may be set "
            "in the 'Model' tab. Click the button <b>'Find Automatically ...'</b> "
            "to set the range of energies before finding the emission lines. The range "
            "may be changed in General Settings dialog (button <b>'General ...'</b>) at any time.")
        set_tooltip(self.rb_full_spectrum,
                    "Plot full spectrum over <b>all available eneriges</b>.")

    def update_widget_state(self, condition=None):
        if condition == "tooltips":
            self._set_tooltips()
        self.mpl_toolbar.setVisible(self.gui_vars["show_matplotlib_toolbar"])

    def btn_group_region_button_toggled(self, button, checked):
        if checked:
            if button == self.rb_selected_region:
                self.gpc.plot_model.energy_range_preview = EnergyRangePresets.SELECTED_RANGE
                self.gpc.plot_model.update_preview_spectrum_plot()
                logger.debug("GUI: Display only selected region")
            elif button == self.rb_full_spectrum:
                self.gpc.plot_model.energy_range_preview = EnergyRangePresets.FULL_SPECTRUM
                self.gpc.plot_model.update_preview_spectrum_plot()
                logger.debug("GUI: Display full spectrum")
            else:
                logger.error("Spectrum preview: unknown button was toggled. "
                             "Please, report the error to the development team.")

    def cb_plot_type_current_index_changed(self, index):
        try:
            self.gpc.plot_model.plot_type_preview = PlotTypes(index)
            self.gpc.plot_model.update_preview_spectrum_plot()
            print(f"Selected index: {index}")
        except ValueError:
            logger.error("Spectrum preview: incorrect index for energy range preset was detected.\n"
                         "Please report the error to the development team.")


class PreviewPlotCount(QWidget):

    def __init__(self, *, gpc, gui_vars):
        super().__init__()

        # Global processing classes
        self.gpc = gpc
        # Global GUI variables (used for control of GUI state)
        self.gui_vars = gui_vars

        self.cb_color_scheme = QComboBox()
        # TODO: make color schemes global
        self._color_schemes = ("viridis", "jet", "bone", "gray", "oranges", "hot")
        self.cb_color_scheme.addItems(self._color_schemes)
        self.cb_color_scheme.currentIndexChanged.connect(self.cb_color_scheme_current_index_changed)

        self.combo_linear_log = QComboBox()
        self.combo_linear_log.addItems(["Linear", "Log"])
        self.combo_linear_log.currentIndexChanged.connect(self.combo_linear_log_current_index_changed)

        self.combo_pixels_positions = QComboBox()
        self.combo_pixels_positions.addItems(["Pixels", "Positions"])
        self.combo_pixels_positions.currentIndexChanged.connect(
            self.combo_pixels_positions_current_index_changed)

        self.range = RangeManager(add_sliders=False)
        self.range.setMaximumWidth(200)
        self.range.selection_changed.connect(self.range_selection_changed)

        self.mpl_canvas = FigureCanvas(self.gpc.plot_model._fig_maps)
        self.mpl_toolbar = NavigationToolbar(self.mpl_canvas, self)

        vbox = QVBoxLayout()

        hbox = QHBoxLayout()
        hbox.addWidget(self.cb_color_scheme)
        hbox.addWidget(self.combo_linear_log)
        hbox.addWidget(self.combo_pixels_positions)
        hbox.addStretch(1)
        hbox.addWidget(QLabel("Range (counts):"))
        hbox.addWidget(self.range)
        vbox.addLayout(hbox)

        vbox.addWidget(self.mpl_toolbar)
        vbox.addWidget(self.mpl_canvas)

        self.setLayout(vbox)

        self._set_tooltips()

    def cb_color_scheme_current_index_changed(self, index):
        logger.debug(f"Color scheme is changed to {self._color_schemes[index]}")
        self.gpc.plot_model.map_preview_color_scheme = self._color_schemes[index]

    def combo_linear_log_current_index_changed(self, index):
        logger.debug(f"Map type is changed to {MapTypes(index)}")
        self.gpc.plot_model.map_type_preview = MapTypes(index)

    def combo_pixels_positions_current_index_changed(self, index):
        logger.debug(f"Map axes are changed to {MapAxesUnits(index)}")
        self.gpc.plot_model.map_axes_units_preview = MapAxesUnits(index)

    def range_selection_changed(self, sel_low, sel_high):
        logger.debug(f"Range selection is changed to ({sel_low:.10g}, {sel_high:.10g})")
        self.gpc.plot_model.map_preview_range_low = sel_low
        self.gpc.plot_model.map_preview_range_high = sel_high

    @pyqtSlot(str)
    def update_map_range(self, mode):
        if mode not in ("reset", "update", "expand", "clear"):
            logger.error(f"PreviewPlotCount.update_map_range: incorrect mode '{mode}'")
            return
        range_low, range_high = self.gpc.io_model.get_dataset_preview_count_map_range()

        if (range_low is None) or (range_high is None) or mode == "clear":
            range_low, range_high = 0.0, 1.0  # No datasets are available
            self.range.set_range(range_low, range_high)
            self.range.reset()
        elif mode == "expand":
            # The range may be only expanded. Keep the selection.
            range_low_old, range_high_old = self.range.get_range()
            # sel_low, sel_high = self.range.get_selection()
            range_low = min(range_low, range_low_old)
            range_high = max(range_high, range_high_old)
            self.range.set_range(range_low, range_high)
            # self.range.set_selection(value_low=sel_low, value_high=sel_high)
        elif mode == "update":
            # The selection will expand proportionally to the change in range.
            self.range.set_range(range_low, range_high)
        elif mode == "reset":
            self.range.set_range(range_low, range_high)
            self.range.reset()
        logger.debug(f"Total Count Preview range is updated: mode='{mode}' range=({range_low}, {range_high})")

    def _set_tooltips(self):
        set_tooltip(self.cb_color_scheme,
                    "Select <b>color scheme</b> for the plotted maps.")
        set_tooltip(
            self.range,
            "<b>Lower and upper limits</b> for the displayed range of intensities. The pixels with "
            "intensities outside the range are <b>clipped</b>.")

    def update_widget_state(self, condition=None):
        if condition == "tooltips":
            self._set_tooltips()
        self.mpl_toolbar.setVisible(self.gui_vars["show_matplotlib_toolbar"])
