from qtpy.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QCheckBox, QPushButton
from qtpy.QtCore import Signal, Slot

from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)

from .useful_widgets import set_tooltip, global_gui_variables

import logging

logger = logging.getLogger(__name__)


class PlotXrfMaps(QWidget):

    signal_maps_dataset_selection_changed = Signal()
    signal_maps_norm_changed = Signal()

    def __init__(self, *, gpc, gui_vars):
        super().__init__()

        self._dataset_list = []  # The list of datasets ('combo_select_dataset')
        self._scaler_list = []  # The list of scalers ('combo_normalization')

        # Global processing classes
        self.gpc = gpc
        # Global GUI variables (used for control of GUI state)
        self.gui_vars = gui_vars

        # Reference to the main window. The main window will hold
        #   references to all non-modal windows that could be opened
        #   from multiple places in the program.
        self.ref_main_window = self.gui_vars["ref_main_window"]

        self.combo_select_dataset = QComboBox()
        self.combo_select_dataset.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.combo_select_dataset.currentIndexChanged.connect(self.combo_select_dataset_current_index_changed)

        self.combo_normalization = QComboBox()
        self.combo_normalization.currentIndexChanged.connect(self.combo_normalization_current_index_changed)

        self.pb_image_wizard = QPushButton("Image Wizard ...")
        self.pb_image_wizard.clicked.connect(self.pb_image_wizard_clicked)

        # self.pb_quant_settings = QPushButton("Quantitative ...")

        self.cb_interpolate = QCheckBox("Interpolate")
        self.cb_interpolate.setChecked(self.gpc.get_maps_grid_interpolate())
        self.cb_interpolate.toggled.connect(self.cb_interpolate_toggled)

        self.cb_scatter_plot = QCheckBox("Scatter plot")
        self.cb_scatter_plot.setChecked(self.gpc.get_maps_show_scatter_plot())
        self.cb_scatter_plot.toggled.connect(self.cb_scatter_plot_toggled)

        self.cb_quantitative = QCheckBox("Quantitative")
        self.cb_quantitative.setChecked(self.gpc.get_maps_quant_norm_enabled())
        self.cb_quantitative.toggled.connect(self.cb_quantitative_toggled)

        self.combo_color_scheme = QComboBox()
        # TODO: make color schemes global
        self._color_schemes = ("viridis", "jet", "bone", "gray", "Oranges", "hot")
        self.combo_color_scheme.addItems(self._color_schemes)
        self.combo_color_scheme.setCurrentIndex(self._color_schemes.index(self.gpc.get_maps_color_opt()))
        self.combo_color_scheme.currentIndexChanged.connect(self.combo_color_scheme_current_index_changed)

        self.combo_linear_log = QComboBox()
        self._linear_log_values = ["Linear", "Log"]
        self.combo_linear_log.addItems(["Linear", "Log"])
        scale_opt = self.gpc.get_maps_scale_opt()
        ind = self._linear_log_values.index(scale_opt)
        self.combo_linear_log.setCurrentIndex(ind)
        self.combo_linear_log.currentIndexChanged.connect(self.combo_linear_log_current_index_changed)

        self.combo_pixels_positions = QComboBox()
        self._pix_pos_values = ["Pixels", "Positions"]
        self.combo_pixels_positions.addItems(self._pix_pos_values)
        self.combo_pixels_positions.setCurrentIndex(self._pix_pos_values.index(self.gpc.get_maps_pixel_or_pos()))
        self.combo_pixels_positions.currentIndexChanged.connect(self.combo_pixels_positions_current_index_changed)

        self.mpl_canvas = FigureCanvas(self.gpc.img_model_adv.fig)
        self.mpl_toolbar = NavigationToolbar(self.mpl_canvas, self)

        # Keep layout without change when canvas is hidden (invisible)
        sp_retain = self.mpl_canvas.sizePolicy()
        sp_retain.setRetainSizeWhenHidden(True)
        self.mpl_canvas.setSizePolicy(sp_retain)

        vbox = QVBoxLayout()
        hbox = QHBoxLayout()
        hbox.addWidget(self.combo_select_dataset)
        hbox.addStretch(4)
        hbox.addWidget(self.pb_image_wizard)
        hbox.addStretch(1)
        hbox.addWidget(self.cb_interpolate)
        hbox.addWidget(self.cb_scatter_plot)
        vbox.addLayout(hbox)
        hbox = QHBoxLayout()
        hbox.addWidget(self.combo_normalization)
        hbox.addStretch(3)
        hbox.addWidget(self.cb_quantitative)
        hbox.addStretch(1)
        hbox.addWidget(self.combo_linear_log)
        hbox.addWidget(self.combo_pixels_positions)
        hbox.addWidget(self.combo_color_scheme)
        vbox.addLayout(hbox)
        vbox.addWidget(self.mpl_toolbar)
        vbox.addWidget(self.mpl_canvas)
        self.setLayout(vbox)

        self._set_tooltips()

    def _set_tooltips(self):
        set_tooltip(self.combo_select_dataset, "Select <b>dataset</b>.")
        set_tooltip(self.combo_normalization, "Select <b>scaler</b> for normalization of displayed XRF maps.")
        set_tooltip(
            self.pb_image_wizard,
            "Open the window with tools for <b>selection and configuration</b> the displayed XRF maps.",
        )
        set_tooltip(self.cb_interpolate, "Interpolate coordinates to <b>uniform grid</b>.")
        set_tooltip(self.cb_scatter_plot, "Display <b>scatter plot</b>.")
        set_tooltip(
            self.cb_quantitative,
            "Normalize the displayed XRF maps using loaded <b>Quantitative Calibration</b> data.",
        )
        set_tooltip(self.combo_color_scheme, "Select <b>color scheme</b>")
        set_tooltip(
            self.combo_linear_log,
            "Switch between <b>linear</b> and <b>logarithmic</b> scale "
            "for plotting of XRF Map pixel <b>intensity</b>.",
        )
        set_tooltip(
            self.combo_pixels_positions, "Switch axes units between <b>pixels</b> and <b>positional units</b>."
        )

    def update_widget_state(self, condition=None):
        if condition == "tooltips":
            self._set_tooltips()
        self.mpl_toolbar.setVisible(self.gui_vars["show_matplotlib_toolbar"])

        # Hide Matplotlib canvas during computations
        state_compute = global_gui_variables["gui_state"]["running_computations"]
        self.mpl_canvas.setVisible(not state_compute)

    def pb_image_wizard_clicked(self):
        # Position the window in relation ot the main window (only when called once)
        pos = self.ref_main_window.pos()
        self.ref_main_window.wnd_image_wizard.position_once(pos.x(), pos.y())

        if not self.ref_main_window.wnd_image_wizard.isVisible():
            self.ref_main_window.wnd_image_wizard.show()
        self.ref_main_window.wnd_image_wizard.activateWindow()

    def cb_scatter_plot_toggled(self, state):
        self.gpc.set_maps_show_scatter_plot(state)
        self.cb_interpolate.setVisible(not state)
        self.combo_pixels_positions.setVisible(not state)

    def cb_quantitative_toggled(self, state):
        self.gpc.set_maps_quant_norm_enabled(state)
        self.signal_maps_norm_changed.emit()

    def combo_select_dataset_current_index_changed(self, index):
        self.gpc.set_maps_selected_dataset(index + 1)
        self.signal_maps_dataset_selection_changed.emit()

    @Slot()
    def combo_select_dataset_update_current_index(self):
        index = self.gpc.get_maps_selected_dataset()
        self.combo_select_dataset.setCurrentIndex(index - 1)

    def combo_normalization_current_index_changed(self, index):
        self.gpc.set_maps_scaler_index(index)
        self.signal_maps_norm_changed.emit()

    def combo_linear_log_current_index_changed(self, index):
        self.gpc.set_maps_scale_opt(self._linear_log_values[index])

    def combo_color_scheme_current_index_changed(self, index):
        self.gpc.set_maps_color_opt(self._color_schemes[index])

    def combo_pixels_positions_current_index_changed(self, index):
        self.gpc.set_maps_pixel_or_pos(self._pix_pos_values[index])

    def cb_interpolate_toggled(self, state):
        self.gpc.set_maps_grid_interpolate(state)

    @Slot()
    def slot_update_dataset_info(self):
        self._update_datasets()
        self._update_scalers()
        self.cb_quantitative.setChecked(self.gpc.get_maps_quant_norm_enabled())

    def _update_datasets(self):
        dataset_list, dset_sel = self.gpc.get_maps_dataset_list()
        self._dataset_list = dataset_list.copy()
        self.combo_select_dataset.clear()
        self.combo_select_dataset.addItems(self._dataset_list)
        # No item should be selected if 'dset_sel' is 0
        self.combo_select_dataset.setCurrentIndex(dset_sel - 1)

    def _update_scalers(self):
        scalers, scaler_sel = self.gpc.get_maps_scaler_list()
        self._scaler_list = ["Normalize by ..."] + scalers
        self.combo_normalization.clear()
        self.combo_normalization.addItems(self._scaler_list)
        self.combo_normalization.setCurrentIndex(scaler_sel)
