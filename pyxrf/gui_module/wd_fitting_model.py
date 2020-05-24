from PyQt5.QtWidgets import (QWidget, QTabWidget, QLabel, QVBoxLayout, QHBoxLayout,
                             QRadioButton, QButtonGroup, QComboBox, QCheckBox, QPushButton,
                             QLineEdit, QDial)
from PyQt5.QtCore import Qt


class PlotFittingModel(QWidget):

    def __init__(self):
        super().__init__()

        self.cb_plot_type = QComboBox()
        self.cb_plot_type.addItems(["LinLog", "Linear"])

        self.cbox_show_spectrum = QCheckBox("Show spectrum")
        self.cbox_show_fit = QCheckBox("Show fit")

        self.pb_selected_region = QRadioButton("Selected region")
        self.pb_selected_region.setChecked(True)
        self.pb_full_spectrum = QRadioButton("Full spectrum")

        self.bgroup = QButtonGroup()
        self.bgroup.addButton(self.pb_selected_region)
        self.bgroup.addButton(self.pb_full_spectrum)

        self.pb_escape_peak = QPushButton("Escape Peak ...")

        self.pb_add_line = QPushButton("Add Line")
        self.pb_remove_line = QPushButton("Remove Line")

        self.cb_select_line = QComboBox()
        # Some emission lines to populate the combo box (as a demo)
        eline_sample_list = ["Li_K", "B_K", "C_K", "N_K", "Fe_K"]
        self.cb_select_line.addItems(eline_sample_list)

        self.pb_previous_line = QPushButton("<")
        self.pb_next_line = QPushButton(">")

        def _set_pb_width(push_button):
            pb_size = push_button.sizeHint()
            push_button.setMaximumSize(pb_size.height() * 2 // 3, pb_size.height())
        _set_pb_width(self.pb_previous_line)
        _set_pb_width(self.pb_next_line)

        # The label will be replaced with the widget that will actually plot the data
        label = QLabel()
        comment = \
            "The widget will plot experimental and fitted spectrum of the loaded data.\n"\
            "Data presentation will be similar to 'Spectrum View' tab, except that\n"\
            "preview data is going to be displayed in a separate tab."
        label.setText(comment)
        label.setStyleSheet("QLabel { background-color : white; color : blue; }")
        label.setAlignment(Qt.AlignCenter)

        vbox = QVBoxLayout()

        hbox = QHBoxLayout()
        hbox.addWidget(self.cb_plot_type)
        hbox.addWidget(self.cbox_show_spectrum)
        hbox.addWidget(self.cbox_show_fit)
        hbox.addStretch(1)
        hbox.addWidget(self.pb_selected_region)
        hbox.addWidget(self.pb_full_spectrum)
        vbox.addLayout(hbox)

        hbox = QHBoxLayout()
        hbox.addWidget(self.pb_escape_peak)
        hbox.addStretch(1)
        hbox.addWidget(self.pb_add_line)
        hbox.addWidget(self.pb_remove_line)
        hbox.addSpacing(20)
        hbox.addWidget(self.pb_previous_line)
        hbox.addWidget(self.cb_select_line)
        hbox.addWidget(self.pb_next_line)
        hbox.addStretch(1)
        vbox.addLayout(hbox)

        vbox.addWidget(label)
        self.setLayout(vbox)

