from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QRadioButton,
    QButtonGroup,
    QComboBox,
    QCheckBox,
    QPushButton,
    QDialog,
)
from qtpy.QtCore import Qt, Slot, Signal

from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)

from .useful_widgets import ElementSelection, set_tooltip  # , global_gui_variables
from .dlg_plot_escape_peak import DialogPlotEscapePeak


class PlotFittingModel(QWidget):

    signal_selected_element_changed = Signal(str)
    signal_add_line = Signal()
    signal_remove_line = Signal()

    def __init__(self, *, gpc, gui_vars):
        super().__init__()

        self._enable_events = False

        # Global processing classes
        self.gpc = gpc
        # Global GUI variables (used for control of GUI state)
        self.gui_vars = gui_vars

        self.combo_plot_type = QComboBox()
        self.combo_plot_type.addItems(["LinLog", "Linear"])
        # Values are received and sent to the model, the don't represent the displayed text
        self.combo_plot_type_values = ["linlog", "linear"]

        self.cb_show_spectrum = QCheckBox("Show spectrum")

        self.cb_show_fit = QCheckBox("Show fit")

        self.rb_selected_region = QRadioButton("Selected region")
        self.rb_selected_region.setChecked(True)

        self.rb_full_spectrum = QRadioButton("Full spectrum")

        self.bgroup = QButtonGroup()
        self.bgroup.addButton(self.rb_selected_region)
        self.bgroup.addButton(self.rb_full_spectrum)

        self.pb_escape_peak = QPushButton("Escape Peak ...")
        self.pb_escape_peak.clicked.connect(self.pb_escape_peak_clicked)

        self.pb_add_line = QPushButton("Add Line")
        self.pb_add_line.clicked.connect(self.pb_add_line_clicked)

        self.pb_remove_line = QPushButton("Remove Line")
        self.pb_remove_line.clicked.connect(self.pb_remove_line_clicked)

        self.element_selection = ElementSelection()
        eline_sample_list = ["Li_K", "B_K", "C_K", "N_K", "Fe_K", "Userpeak1"]
        self.element_selection.set_item_list(eline_sample_list)
        self.element_selection.signal_current_item_changed.connect(self.element_selection_item_changed)

        self.mpl_canvas = FigureCanvas(self.gpc.plot_model._fig)
        self.mpl_toolbar = NavigationToolbar(self.mpl_canvas, self)

        # Keep layout without change when canvas is hidden (invisible)
        sp_retain = self.mpl_canvas.sizePolicy()
        sp_retain.setRetainSizeWhenHidden(True)
        self.mpl_canvas.setSizePolicy(sp_retain)

        self.widgets_enable_events(True)

        vbox = QVBoxLayout()

        hbox = QHBoxLayout()
        hbox.addWidget(self.combo_plot_type)
        hbox.addWidget(self.cb_show_spectrum)
        hbox.addWidget(self.cb_show_fit)
        hbox.addStretch(1)
        hbox.addWidget(self.rb_selected_region)
        hbox.addWidget(self.rb_full_spectrum)
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

    def widgets_enable_events(self, enable):
        if enable:
            if not self._enable_events:
                self.cb_show_spectrum.toggled.connect(self.cb_show_spectrum_toggled)
                self.cb_show_fit.toggled.connect(self.cb_show_fit_toggled)
                self.bgroup.buttonToggled.connect(self.bgroup_button_toggled)
                self.combo_plot_type.currentIndexChanged.connect(self.combo_plot_type_current_index_changed)
                self._enable_events = True
        else:
            if self._enable_events:
                self.cb_show_spectrum.toggled.disconnect(self.cb_show_spectrum_toggled)
                self.cb_show_fit.toggled.disconnect(self.cb_show_fit_toggled)
                self.bgroup.buttonToggled.disconnect(self.bgroup_button_toggled)
                self.combo_plot_type.currentIndexChanged.disconnect(self.combo_plot_type_current_index_changed)
                self._enable_events = False

    def _set_tooltips(self):
        set_tooltip(self.combo_plot_type, "Use <b>Linear</b> or <b>LinLog</b> axes to plot spectra")
        set_tooltip(self.cb_show_spectrum, "Show <b>raw<b> spectrum")
        set_tooltip(self.cb_show_fit, "Show <b>fitted</b> spectrum")
        set_tooltip(
            self.rb_selected_region,
            "Plot spectrum in the <b>selected range</b> of energies. The range may be set "
            "in the 'Model' tab. Click the button <b>'Find Automatically ...'</b> "
            "to set the range of energies before finding the emission lines. The range "
            "may be changed in General Settings dialog (button <b>'General ...'</b>) at any time.",
        )
        set_tooltip(self.rb_full_spectrum, "Plot full spectrum over <b>all available eneriges</b>.")
        set_tooltip(
            self.pb_escape_peak,
            "Select options for displaying the <b>escape peak</b>. "
            "If activated, the location of the escape peak is shown "
            "for the emission line, which is currently selected for adding, "
            "removal or editing.",
        )
        set_tooltip(self.pb_add_line, "<b>Add</b> the current emission line to the list of selected lines")
        set_tooltip(
            self.pb_remove_line, "<b>Remove</b> the current emission line from the list of selected lines."
        )
        set_tooltip(self.element_selection, "<b>Choose</b> the emission line for addition or removal.")

    def update_widget_state(self, condition=None):
        if condition == "tooltips":
            self._set_tooltips()
        self.mpl_toolbar.setVisible(self.gui_vars["show_matplotlib_toolbar"])

        # Hide Matplotlib canvas during computations
        # state_compute = global_gui_variables["gui_state"]["running_computations"]
        # self.mpl_canvas.setVisible(not state_compute)

    @Slot()
    def update_controls(self):
        self.widgets_enable_events(False)
        plot_spectrum, plot_fit = self.gpc.get_line_plot_state()
        self.cb_show_spectrum.setChecked(plot_spectrum)
        self.cb_show_fit.setChecked(plot_fit)
        if self.gpc.get_plot_fit_energy_range() == "selected":
            self.rb_selected_region.setChecked(True)
        else:
            self.rb_full_spectrum.setChecked(True)

        try:
            index = self.combo_plot_type_values.index(self.gpc.get_plot_fit_linlog())
            self.combo_plot_type.setCurrentIndex(index)
        except ValueError:
            self.combo_plot_type.setCurrentIndex(-1)

        self.widgets_enable_events(True)

    def le_mouse_press(self, event):
        print("Button pressed (line edit)")
        if event.button() == Qt.RightButton:
            print("Right button pressed")

    def pb_escape_peak_clicked(self, event):
        plot_escape_peak, detector_material = self.gpc.get_escape_peak_params()
        incident_energy = self.gpc.get_incident_energy()
        dlg = DialogPlotEscapePeak()
        dlg.set_parameters(plot_escape_peak, incident_energy, detector_material)
        if dlg.exec() == QDialog.Accepted:
            plot_escape_peak, detector_material = dlg.get_parameters()
            self.gpc.set_escape_peak_params(plot_escape_peak, detector_material)
            print("Dialog exit: Ok button")

    def cb_show_spectrum_toggled(self, state):
        self.gpc.show_plot_spectrum(state)

    def cb_show_fit_toggled(self, state):
        self.gpc.show_plot_fit(state)

    def bgroup_button_toggled(self, button, checked):
        if checked:
            if button == self.rb_selected_region:
                self.gpc.set_plot_fit_energy_range("selected")
            else:
                self.gpc.set_plot_fit_energy_range("full")

    def combo_plot_type_current_index_changed(self, index):
        self.gpc.set_plot_fit_linlog(self.combo_plot_type_values[index])

    def pb_add_line_clicked(self):
        self.signal_add_line.emit()

    def pb_remove_line_clicked(self):
        self.signal_remove_line.emit()

    def element_selection_item_changed(self, index, eline):
        self.signal_selected_element_changed.emit(eline)

    @Slot(str)
    def slot_selection_item_changed(self, eline):
        self.element_selection.set_current_item(eline)

    @Slot()
    def slot_update_eline_selection_list(self):
        eline_list = self.gpc.get_full_eline_list()
        self.element_selection.set_item_list(eline_list)

    @Slot(bool, bool)
    def slot_update_add_remove_btn_state(self, add_enabled, remove_enabled):
        self.pb_add_line.setEnabled(add_enabled)
        self.pb_remove_line.setEnabled(remove_enabled)

    @Slot()
    @Slot(bool)
    def redraw_plot_fit(self):
        self.gpc.update_plot_fit()
