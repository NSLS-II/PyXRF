from PyQt5.QtWidgets import (QWidget, QTabWidget, QLabel, QVBoxLayout, QHBoxLayout,
                             QRadioButton, QButtonGroup, QComboBox, QLineEdit, QDial)
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


class PreviewPlotCount(QWidget):

    def __init__(self):
        super().__init__()

        self.cb_color_scheme = QComboBox()
        # TODO: make color schemes global
        color_schemes = ("viridis", "jet", "bone", "gray", "oranges", "hot")
        self.cb_color_scheme.addItems(color_schemes)

        self.le_range_min = QLineEdit()
        self.le_range_min.setMaximumWidth(100)
        self.le_range_min.setAlignment(Qt.AlignCenter)
        self.le_range_max = QLineEdit()
        self.le_range_max.setMaximumWidth(100)
        self.le_range_max.setAlignment(Qt.AlignCenter)

        # Set some sample values
        #self.le_range_min.setText("200.0")
        #self.le_range_max.setText("1500.0")

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
