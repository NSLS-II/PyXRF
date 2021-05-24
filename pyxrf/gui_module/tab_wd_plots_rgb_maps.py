from qtpy.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QCheckBox, QSpacerItem
from qtpy.QtCore import Signal, Slot

from .useful_widgets import set_tooltip, global_gui_variables

from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)

from .wd_rgb_selection import RgbSelectionWidget

import logging

logger = logging.getLogger(__name__)


class PlotRgbMaps(QWidget):

    signal_rgb_maps_dataset_selection_changed = Signal()
    signal_rgb_maps_norm_changed = Signal()
    signal_redraw_rgb_maps = Signal()

    def __init__(self, *, gpc, gui_vars):
        super().__init__()

        self._enable_plot_updates = True
        self._changes_exist = False
        self._enable_events = False

        # Global processing classes
        self.gpc = gpc
        # Global GUI variables (used for control of GUI state)
        self.gui_vars = gui_vars

        self.combo_select_dataset = QComboBox()
        self.combo_select_dataset.setSizeAdjustPolicy(QComboBox.AdjustToContents)

        self.combo_normalization = QComboBox()

        self.cb_interpolate = QCheckBox("Interpolate")
        self.cb_interpolate.setChecked(self.gpc.get_rgb_maps_grid_interpolate())
        self.cb_interpolate.toggled.connect(self.cb_interpolate_toggled)

        self.cb_quantitative = QCheckBox("Quantitative")
        self.cb_quantitative.setChecked(self.gpc.get_maps_quant_norm_enabled())
        self.cb_quantitative.toggled.connect(self.cb_quantitative_toggled)

        self.combo_pixels_positions = QComboBox()
        self._pix_pos_values = ["Pixels", "Positions"]
        self.combo_pixels_positions.addItems(self._pix_pos_values)
        self.combo_pixels_positions.setCurrentIndex(self._pix_pos_values.index(self.gpc.get_maps_pixel_or_pos()))
        self.combo_pixels_positions.currentIndexChanged.connect(self.combo_pixels_positions_current_index_changed)

        self.mpl_canvas = FigureCanvas(self.gpc.img_model_rgb.fig)
        self.mpl_toolbar = NavigationToolbar(self.mpl_canvas, self)

        # Keep layout without change when canvas is hidden (invisible)
        sp_retain = self.mpl_canvas.sizePolicy()
        sp_retain.setRetainSizeWhenHidden(True)
        self.mpl_canvas.setSizePolicy(sp_retain)

        self.rgb_selection = RgbSelectionWidget()
        self.slot_update_dataset_info()

        self.rgb_selection.signal_update_map_selections.connect(self._update_map_selections)

        self.widgets_enable_events(True)

        vbox = QVBoxLayout()
        hbox = QHBoxLayout()
        hbox.addWidget(self.combo_select_dataset)
        hbox.addWidget(self.combo_normalization)
        hbox.addStretch(1)
        hbox.addWidget(self.cb_quantitative)
        hbox.addWidget(self.cb_interpolate)
        hbox.addWidget(self.combo_pixels_positions)
        vbox.addLayout(hbox)
        vbox.addWidget(self.mpl_toolbar)
        vbox.addWidget(self.mpl_canvas)
        hbox = QHBoxLayout()
        hbox.addSpacerItem(QSpacerItem(0, 0))
        hbox.addWidget(self.rgb_selection)
        hbox.addSpacerItem(QSpacerItem(0, 0))
        vbox.addLayout(hbox)
        self.setLayout(vbox)

        self._set_tooltips()

    def _set_tooltips(self):
        set_tooltip(self.combo_select_dataset, "Select <b>dataset</b>.")
        set_tooltip(self.combo_normalization, "Select <b>scaler</b> for normalization of displayed XRF maps.")
        set_tooltip(self.cb_interpolate, "Interpolate coordinates to <b>uniform grid</b>.")
        set_tooltip(
            self.cb_quantitative,
            "Normalize the displayed XRF maps using loaded <b>Quantitative Calibration</b> data.",
        )
        set_tooltip(
            self.combo_pixels_positions, "Switch axes units between <b>pixels</b> and <b>positional units</b>."
        )
        set_tooltip(
            self.rgb_selection,
            "Select XRF Maps displayed in <b>Red</b>, <b>Green</b> and "
            "<b>Blue</b> colors and adjust the range of <b>intensity</b> for each "
            "displayed map.",
        )

    def widgets_enable_events(self, status):
        if status:
            if not self._enable_events:
                self.combo_select_dataset.currentIndexChanged.connect(
                    self.combo_select_dataset_current_index_changed
                )
                self.combo_normalization.currentIndexChanged.connect(
                    self.combo_normalization_current_index_changed
                )
                self._enable_events = True
        else:
            if self._enable_events:
                self.combo_select_dataset.currentIndexChanged.disconnect(
                    self.combo_select_dataset_current_index_changed
                )
                self.combo_normalization.currentIndexChanged.disconnect(
                    self.combo_normalization_current_index_changed
                )
                self._enable_events = False

    def update_widget_state(self, condition=None):
        if condition == "tooltips":
            self._set_tooltips()
        self.mpl_toolbar.setVisible(self.gui_vars["show_matplotlib_toolbar"])

        # Hide Matplotlib canvas during computations
        state_compute = global_gui_variables["gui_state"]["running_computations"]
        self.mpl_canvas.setVisible(not state_compute)

    def combo_select_dataset_current_index_changed(self, index):
        self.gpc.set_rgb_maps_selected_dataset(index + 1)
        self._update_dataset()
        self.signal_rgb_maps_dataset_selection_changed.emit()

    @Slot()
    def combo_select_dataset_update_current_index(self):
        index = self.gpc.get_rgb_maps_selected_dataset()
        self.combo_select_dataset.setCurrentIndex(index - 1)

    def combo_normalization_current_index_changed(self, index):
        self.gpc.set_rgb_maps_scaler_index(index)
        self.slot_update_ranges()
        self.signal_rgb_maps_norm_changed.emit()

    def combo_pixels_positions_current_index_changed(self, index):
        self.gpc.set_rgb_maps_pixel_or_pos(self._pix_pos_values[index])

    def cb_interpolate_toggled(self, state):
        self.gpc.set_rgb_maps_grid_interpolate(state)

    def cb_quantitative_toggled(self, state):
        self.gpc.set_rgb_maps_quant_norm_enabled(state)
        self.slot_update_ranges()
        self.signal_rgb_maps_norm_changed.emit()

    @Slot()
    def slot_update_dataset_info(self):
        self._update_dataset_list()
        self._update_dataset()
        self.cb_quantitative.setChecked(self.gpc.get_rgb_maps_quant_norm_enabled())

    def _update_dataset(self):
        self._update_scalers()

        # Update ranges in the RGB selection widget
        range_table, limit_table, rgb_dict = self.gpc.get_rgb_maps_info_table()
        self.rgb_selection.set_ranges_and_limits(
            range_table=range_table, limit_table=limit_table, rgb_dict=rgb_dict
        )

    @Slot()
    def slot_update_ranges(self):
        """Update only ranges and selections for the emission lines"""
        range_table, limit_table, _ = self.gpc.get_rgb_maps_info_table()
        self.rgb_selection.set_ranges_and_limits(range_table=range_table, limit_table=limit_table)

    def _update_dataset_list(self):
        self.widgets_enable_events(False)
        dataset_list, dset_sel = self.gpc.get_rgb_maps_dataset_list()
        self._dataset_list = dataset_list.copy()
        self.combo_select_dataset.clear()
        self.combo_select_dataset.addItems(self._dataset_list)
        # No item should be selected if 'dset_sel' is 0
        self.combo_select_dataset.setCurrentIndex(dset_sel - 1)
        self.widgets_enable_events(True)

    def _update_scalers(self):
        self.widgets_enable_events(False)
        scalers, scaler_sel = self.gpc.get_rgb_maps_scaler_list()
        self._scaler_list = ["Normalize by ..."] + scalers
        self.combo_normalization.clear()
        self.combo_normalization.addItems(self._scaler_list)
        self.combo_normalization.setCurrentIndex(scaler_sel)
        self.widgets_enable_events(True)

    @Slot()
    def _update_map_selections(self):
        """Upload the selections (limit table) and update plot"""
        if self._enable_plot_updates:
            self._changes_exist = False
            self.gpc.set_rgb_maps_limit_table(self.rgb_selection._limit_table, self.rgb_selection._rgb_dict)
            self._redraw_maps()

    def _redraw_maps(self):
        logger.debug("Redrawing XRF Maps")
        self.gpc.redraw_rgb_maps()
        self.signal_redraw_rgb_maps.emit()
