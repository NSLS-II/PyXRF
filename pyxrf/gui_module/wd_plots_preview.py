from PyQt5.QtWidgets import (QWidget, QTabWidget, QLabel, QVBoxLayout, QHBoxLayout,
                             QRadioButton, QButtonGroup, QComboBox)
from PyQt5.QtCore import Qt

from .useful_widgets import RangeManager


class PreviewPlots(QTabWidget):

    def __init__(self):
        super().__init__()

        self.addTab(PreviewPlotSpectrum(), "Total Spectrum")
        self.addTab(PreviewPlotCount(), "Total Count")


class PreviewPlotSpectrum(QWidget):

    def __init__(self):
        super().__init__()

        self.cb_plot_type = QComboBox()
        self.cb_plot_type.addItems(["LinLog", "Linear"])
        self.cb_plot_type.setToolTip(
            "Use <b>Linear</b> or <b>LinLog</b> axes to plot spectra")

        self.pb_selected_region = QRadioButton("Selected region")
        self.pb_selected_region.setToolTip(
            "Plot spectrum in the <b>selected range</b> of energies. The range may be set "
            "in the 'Model' tab. Click the button <b>'Find Automatically ...'</b> "
            "to set the range of energies before finding the emission lines. The range "
            "may be changed in General Settings dialog (button <b>'General ...'</b>) at any time.")
        self.pb_selected_region.setChecked(True)

        self.pb_full_spectrum = QRadioButton("Full spectrum")
        self.pb_full_spectrum.setToolTip(
            "Plot full spectrum over <b>all available eneriges</b>.")

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


class PreviewPlotCount(QWidget):

    def __init__(self):
        super().__init__()

        self.cb_color_scheme = QComboBox()
        # TODO: make color schemes global
        color_schemes = ("viridis", "jet", "bone", "gray", "oranges", "hot")
        self.cb_color_scheme.addItems(color_schemes)
        self.cb_color_scheme.setToolTip(
            "Select <b>color scheme</b> for the plotted maps.")

        self.range = RangeManager(add_sliders=False)
        self.range.setToolTip(
            "<b>Lower and upper limits</b> for the displayed range of intensities. The pixels with "
            "intensities outside the range are <b>clipped</b>.")
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
