from PyQt5.QtWidgets import (QWidget, QTabWidget, QLabel, QVBoxLayout, QHBoxLayout,
                             QRadioButton, QButtonGroup, QComboBox)
from PyQt5.QtCore import Qt

from .useful_widgets import RangeManager, set_tooltip


class PreviewPlots(QTabWidget):

    def __init__(self):
        super().__init__()

        self.preview_plot_spectrum = PreviewPlotSpectrum()
        self.addTab(self.preview_plot_spectrum, "Total Spectrum")
        self.preview_plot_count = PreviewPlotCount()
        self.addTab(self.preview_plot_count, "Total Count")

    def update_widget_state(self, condition=None):
        self.preview_plot_spectrum.update_widget_state(condition)
        self.preview_plot_count.update_widget_state(condition)


class PreviewPlotSpectrum(QWidget):

    def __init__(self):
        super().__init__()

        self.cb_plot_type = QComboBox()
        self.cb_plot_type.addItems(["LinLog", "Linear"])

        self.pb_selected_region = QRadioButton("Selected region")
        self.pb_selected_region.setChecked(True)

        self.pb_full_spectrum = QRadioButton("Full spectrum")

        self.bgroup = QButtonGroup()
        self.bgroup.addButton(self.pb_selected_region)
        self.bgroup.addButton(self.pb_full_spectrum)

        # The label will be replaced with the widget that will actually plot the data
        label = QLabel()
        comment = \
            "The widget will plot total spectrum of the loaded data.\n"\
            "The displayed channels are selected using CheckBoxes in 'Data' tab.\n"\
            "When implemented, the data will be presented as in 'Spectrum View' tab "\
            "of the original PyXRF"
        label.setText(comment)
        label.setStyleSheet("QLabel { background-color : white; color : blue; }")
        label.setAlignment(Qt.AlignCenter)

        vbox = QVBoxLayout()

        hbox = QHBoxLayout()
        hbox.addWidget(self.cb_plot_type)
        hbox.addStretch(1)
        hbox.addWidget(self.pb_selected_region)
        hbox.addWidget(self.pb_full_spectrum)
        vbox.addLayout(hbox)

        vbox.addWidget(label)
        self.setLayout(vbox)

        self._set_tooltips()

    def _set_tooltips(self):
        set_tooltip(self.cb_plot_type,
                    "Use <b>Linear</b> or <b>LinLog</b> axes to plot spectra")
        set_tooltip(
            self.pb_selected_region,
            "Plot spectrum in the <b>selected range</b> of energies. The range may be set "
            "in the 'Model' tab. Click the button <b>'Find Automatically ...'</b> "
            "to set the range of energies before finding the emission lines. The range "
            "may be changed in General Settings dialog (button <b>'General ...'</b>) at any time.")
        set_tooltip(self.pb_full_spectrum,
                    "Plot full spectrum over <b>all available eneriges</b>.")

    def update_widget_state(self, condition=None):
        if condition == "tooltips":
            self._set_tooltips()


class PreviewPlotCount(QWidget):

    def __init__(self):
        super().__init__()

        self.cb_color_scheme = QComboBox()
        # TODO: make color schemes global
        color_schemes = ("viridis", "jet", "bone", "gray", "oranges", "hot")
        self.cb_color_scheme.addItems(color_schemes)

        self.range = RangeManager(add_sliders=False)
        self.range.setMaximumWidth(200)

        label = QLabel()
        comment = \
            "The widget will plot 2D image representing total in each pixel of the loaded data.\n"\
            "The image represents total collected fluorescence from the sample.\n"\
            "The feature does not exist in old PyXRF."
        label.setText(comment)
        label.setStyleSheet("QLabel { background-color : white; color : blue; }")
        label.setAlignment(Qt.AlignCenter)

        vbox = QVBoxLayout()

        hbox = QHBoxLayout()
        hbox.addWidget(self.cb_color_scheme)
        hbox.addStretch(1)
        hbox.addWidget(QLabel("Range (counts):"))
        hbox.addWidget(self.range)
        vbox.addLayout(hbox)

        vbox.addWidget(label)
        self.setLayout(vbox)

        self._set_tooltips()

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
