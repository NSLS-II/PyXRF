from PyQt5.QtWidgets import (QWidget, QLabel, QVBoxLayout, QHBoxLayout, QRadioButton, QButtonGroup,
                             QComboBox, QCheckBox, QPushButton, QGridLayout, QDialog, QDialogButtonBox,
                             QGroupBox)
from PyQt5.QtCore import Qt

from matplotlib.backends.backend_qt5agg import \
    FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar

from .useful_widgets import LineEditReadOnly, ElementSelection, set_tooltip


class PlotFittingModel(QWidget):

    def __init__(self, *, gpc, gui_vars):
        super().__init__()

        # Global processing classes
        self.gpc = gpc
        # Global GUI variables (used for control of GUI state)
        self.gui_vars = gui_vars

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
        self.pb_escape_peak.clicked.connect(self.pb_escape_peak_clicked)

        self.pb_add_line = QPushButton("Add Line")

        self.pb_remove_line = QPushButton("Remove Line")

        self.element_selection = ElementSelection()
        eline_sample_list = ["Li_K", "B_K", "C_K", "N_K", "Fe_K", "Userpeak1"]
        self.element_selection.addItems(eline_sample_list)

        self.mpl_canvas = FigureCanvas(self.gpc.plot_model._fig)
        self.mpl_toolbar = NavigationToolbar(self.mpl_canvas, self)

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
        hbox.addWidget(self.element_selection)
        hbox.addStretch(1)
        vbox.addLayout(hbox)

        vbox.addWidget(self.mpl_toolbar)
        vbox.addWidget(self.mpl_canvas)
        self.setLayout(vbox)

        self._set_tooltips()

    def _set_tooltips(self):
        set_tooltip(self.cb_plot_type,
                    "Use <b>Linear</b> or <b>LinLog</b> axes to plot spectra")
        set_tooltip(self.cbox_show_spectrum,
                    "Show <b>raw<b> spectrum")
        set_tooltip(self.cbox_show_fit,
                    "Show <b>fitted</b> spectrum")
        set_tooltip(self.pb_selected_region,
                    "Plot spectrum in the <b>selected range</b> of energies. The range may be set "
                    "in the 'Model' tab. Click the button <b>'Find Automatically ...'</b> "
                    "to set the range of energies before finding the emission lines. The range "
                    "may be changed in General Settings dialog (button <b>'General ...'</b>) at any time."
                    )
        set_tooltip(self.pb_full_spectrum,
                    "Plot full spectrum over <b>all available eneriges</b>.")
        set_tooltip(self.pb_escape_peak,
                    "Select options for displaying the <b>escape peak</b>")
        set_tooltip(self.pb_add_line,
                    "<b>Add</b> the current emission line to the list of selected lines")
        set_tooltip(self.pb_remove_line,
                    "<b>Remove</b> the current emission line from the list of selected lines.")
        set_tooltip(self.element_selection,
                    "<b>Choose</b> the emission line for addition or removal.")

    def update_widget_state(self, condition=None):
        if condition == "tooltips":
            self._set_tooltips()
        self.mpl_toolbar.setVisible(self.gui_vars["show_matplotlib_toolbar"])

    def le_mouse_press(self, event):
        print("Button pressed (line edit)")
        if event.button() == Qt.RightButton:
            print("Right button pressed")

    def slider1_mouse_press(self, event):
        print("Button pressed (slider1)")
        if event.button() == Qt.RightButton:
            print("Right button pressed")

    def pb_escape_peak_clicked(self, event):
        dlg = DialogPlotEscapePeak()
        if dlg.exec() == QDialog.Accepted:
            print("Dialog exit: Ok button")


class DialogPlotEscapePeak(QDialog):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Escape Peak Settings")

        self.grp_plot_escape_peak = QGroupBox("Plot Escape Peak")
        self.grp_plot_escape_peak.setCheckable(True)
        self.grp_plot_escape_peak.setChecked(False)  # Set based on data !!!

        self.le_incident_energy = LineEditReadOnly()
        set_tooltip(self.le_incident_energy,
                    "<b>Incident energy</b>. Use <b>General...</b> button of <b>Model</b> tab "
                    "to change the value if needed.")
        self.le_incident_energy.setText("12.0")  # Set based on data !!!
        self.cbb_detector_type = QComboBox()
        set_tooltip(self.cbb_detector_type,
                    "Select <b>detector</b> material. The typical choice is <b>Si</b>")
        self.cbb_detector_type.addItems(["Si", "Ge"])

        grid = QGridLayout()
        grid.addWidget(QLabel("Incident energy, kev:"), 0, 0)
        grid.addWidget(self.le_incident_energy, 0, 1)
        grid.addWidget(QLabel("Detectory type:"), 1, 0)
        grid.addWidget(self.cbb_detector_type, 1, 1)
        self.grp_plot_escape_peak.setLayout(grid)

        # Yes/No button box
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        vbox = QVBoxLayout()
        vbox.addWidget(self.grp_plot_escape_peak)
        vbox.addWidget(button_box)
        self.setLayout(vbox)
