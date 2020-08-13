from __future__ import absolute_import

import os
import copy
import math
from ..model.fileio import FileIOModel
from ..model.lineplot import LinePlotModel  # , SettingModel
from ..model.guessparam import GuessParamModel, save_as
from ..model.draw_image import DrawImageAdvanced
from ..model.draw_image_rgb import DrawImageRGB
from ..model.fit_spectrum import Fit1D, get_cs
from ..model.setting import SettingModel
from ..model.param_data import param_data

import logging
logger = logging.getLogger(__name__)


class GlobalProcessingClasses:

    def __init__(self):
        self.defaults = None
        self.io_model = None
        self.param_model = None
        self.plot_model = None
        self.fit_model = None
        self.setting_model = None
        self.img_model_adv = None
        self.img_model_rgb = None

    def _get_defaults(self):

        # Set working directory to current working directory (if PyXRF is started from shell)
        working_directory = os.getcwd()
        logger.info(f"Starting PyXRF in the current working directory '{working_directory}'")

        default_parameters = param_data
        defaults = {'working_directory': working_directory,
                    'default_parameters': default_parameters}
        return defaults

    def initialize(self):
        """
        Run the sequence of actions needed to initialize PyXRF modules.

        """

        defaults = self._get_defaults()
        self.io_model = FileIOModel(**defaults)
        self.param_model = GuessParamModel(**defaults)
        self.plot_model = LinePlotModel(param_model=self.param_model)
        self.fit_model = Fit1D(param_model=self.param_model, io_model=self.io_model, **defaults)
        self.setting_model = SettingModel(**defaults)
        self.img_model_adv = DrawImageAdvanced()
        self.img_model_rgb = DrawImageRGB(img_model_adv=self.img_model_adv)

        # Initialization needed to eliminate program crash
        self.plot_model.roi_dict = self.setting_model.roi_dict

        # send working directory changes to different models
        self.io_model.observe('working_directory', self.fit_model.result_folder_changed)
        self.io_model.observe('working_directory', self.setting_model.result_folder_changed)
        self.io_model.observe('selected_file_name', self.fit_model.data_title_update)
        self.io_model.observe('selected_file_name', self.plot_model.exp_label_update)
        self.io_model.observe('selected_file_name', self.setting_model.data_title_update)

        # send the same file to fit model, as fitting results need to be saved
        self.io_model.observe('file_name', self.fit_model.filename_update)
        self.io_model.observe('file_name', self.plot_model.plot_exp_data_update)
        self.io_model.observe('file_name', self.setting_model.filename_update)
        self.io_model.observe('runid', self.fit_model.runid_update)

        # send exp data to different models
        self.io_model.observe('data', self.plot_model.exp_data_update)
        self.io_model.observe('data', self.param_model.exp_data_update)
        self.io_model.observe('data', self.fit_model.exp_data_update)
        self.io_model.observe('data_all', self.fit_model.exp_data_all_update)
        self.io_model.observe('img_dict', self.fit_model.img_dict_update)
        self.io_model.observe('data_sets', self.fit_model.data_sets_update)
        self.io_model.observe('img_dict', self.setting_model.img_dict_update)

        # send fitting param of summed spectrum to param_model
        self.io_model.observe('param_fit', self.param_model.param_from_db_update)

        # send img dict to img_model for visualization
        self.io_model.observe('img_dict', self.img_model_adv.img_dict_update)
        self.io_model.observe('img_dict', self.img_model_rgb.img_dict_update)
        self.io_model.observe('img_dict', self.plot_model.img_dict_update)

        self.io_model.observe('incident_energy_set', self.plot_model.set_incident_energy)
        self.io_model.observe('incident_energy_set', self.img_model_adv.set_incident_energy)

        self.img_model_adv.observe('scaler_name_index', self.fit_model.scaler_index_update)

        self.img_model_adv.observe('dict_to_plot', self.fit_model.dict_to_plot_update)
        self.img_model_adv.observe('img_title', self.fit_model.img_title_update)
        self.img_model_adv.observe('quantitative_normalization', self.fit_model.quantitative_normalization_update)
        self.img_model_adv.observe('param_quant_analysis', self.fit_model.param_quant_analysis_update)

        self.param_model.observe('energy_bound_high_buf', self.fit_model.energy_bound_high_update)
        self.param_model.observe('energy_bound_low_buf', self.fit_model.energy_bound_low_update)
        self.param_model.observe('energy_bound_high_buf', self.plot_model.energy_bound_high_update)
        self.param_model.observe('energy_bound_low_buf', self.plot_model.energy_bound_low_update)

        logger.info('pyxrf started.')

    def open_data_file(self, file_path):

        self.io_model.data_ready = False
        # only load one file
        # 'temp' is used to reload the same file, otherwise file_name will not update
        self.io_model.file_name = 'temp'
        f_dir, f_name = os.path.split(file_path)
        self.io_model.working_directory = f_dir

        def _update_data():
            self.plot_model.parameters = self.param_model.param_new
            self.plot_model.data_sets = self.io_model.data_sets
            self.setting_model.parameters = self.param_model.param_new
            self.setting_model.data_sets = self.io_model.data_sets
            self.fit_model.data_sets = self.io_model.data_sets
            self.fit_model.fit_img = {}  # clear dict in fitmodel to rm old results
            # This will draw empty (hidden) preview plot, since no channels are selected.
            self.plot_model.update_preview_spectrum_plot()
            self.plot_model.update_total_count_map_preview(new_plot=True)

        # The following statement initiates file loading. It may raise exceptions
        try:
            self.io_model.file_name = f_name

        except Exception:
            _update_data()

            # For plotting purposes, otherwise plot will not update
            self.plot_model.plot_exp_opt = False
            self.plot_model.show_fit_opt = False

            logger.info(f"Failed to load the file '{f_name}'.")
            # Clear file name or scan id from window title. This does not update
            #   the displayed title.
            self.io_model.window_title_clear()
            raise
        else:
            _update_data()

            # For plotting purposes, otherwise plot will not update
            self.plot_model.plot_exp_opt = False
            self.plot_model.plot_exp_opt = True
            self.plot_model.show_fit_opt = False
            self.plot_model.show_fit_opt = True

            # Change window title (include file name). This does not update the visible title,
            #   only the text attribute of 'io_model' class.
            self.io_model.window_title_set_file_name(f_name)

            if not self.io_model.incident_energy_available:
                msg = ("Incident energy is not available in scan metadata and must be set manually. "
                       "Incident energy may be set by changing 'Incident energy, keV' parameter "
                       "in the dialog boxes opened using 'Find Automatically...' ('Find Elements "
                       "in sample' or 'General...' ('General Settings for Fitting Alogirthm') "
                       "buttons in 'Model' tab.")
            else:
                msg = ""

            return msg

    def select_preview_dataset(self, *, dset_name, is_visible):
        """
        Select datasets for preview. A dataset may be selected or deselected by
        providing the name of the dataset (a key in `io_model.datasets` dictionary)
        and `is_visible` flag. Multiple datasets may be visualized by selecting the
        datasets one by one.

        Parameters
        ----------
        dset_name: str
            dataset name (a key in `io_model.datasets` dictionary)
        is_visible: bool
            True - show the dataset, False - hide the dataset
        """
        self.io_model.data_sets[dset_name].selected_for_preview = True if is_visible else False
        self.io_model.update_data_set_buffers()
        self.plot_model.data_sets = self.io_model.data_sets
        self.plot_model.update_preview_spectrum_plot()
        self.plot_model.update_total_count_map_preview()

    def get_current_working_directory(self):
        """
        Return current working directory (defined in 'io_model')
        """
        return self.io_model.working_directory

    def set_data_channel(self, channel_index):
        self.io_model.file_opt = channel_index

        # For plotting purposes, otherwise plot will not update
        self.plot_model.plot_exp_opt = False
        self.plot_model.plot_exp_opt = True
        self.plot_model.show_fit_opt = False
        self.plot_model.show_fit_opt = True

    # ==========================================================================
    #          The following methods are used by widgets XRF Maps tab
    #                  to read/write data from model classes

    def get_maps_info_table(self):
        """
        The function builds and returns two tables: a table of value ranges for each map
        in the dataset; a table of limit values that represent selection for the displayed
        values for each map of the dataset.

        Returns
        -------
        range_table: list(list)
            The table is represented as list of lists. Every element of the outer list is
            a table row. Each row has 3 elements: 0 - key (emission line or map name, str),
            1 - lower boundary (float), 2 - upper boundary (float).
        limit_table: list(list)
            The table is represented as list of lists. Every element of the outer list is
            a table row. Each row has 3 elements: 0 - key (emission line or map name, str),
            1 - lower limit value (float), 2 - upper limit value (float).
        """
        # Check if 'range_dict' and 'limit_dict' have the same set of keys
        ks = self.img_model_adv.map_keys
        ks_range = list(self.img_model_adv.range_dict.keys())
        ks_limit = list(self.img_model_adv.limit_dict.keys())
        ks_show = list(self.img_model_adv.stat_dict.keys())
        if set(ks) != set(ks_limit):
            raise RuntimeError("The list of keys in 'limit_dict' is not as expected: "
                               f"limit_dict.keys(): {ks_limit} expected: {ks}")
        if set(ks) != set(ks_range):
            raise RuntimeError("The list of keys in 'range_dict' is not as expected: "
                               f"range_dict.keys(): {ks_range} expected: {ks}")
        if set(ks) != set(ks_show):
            raise RuntimeError("The list of keys in 'stat_dict' is not as expected: "
                               f"stat_dict.keys(): {ks_show} expected: {ks}")

        range_table = []
        limit_table = []
        show_table = []
        for key in ks:
            rng_low = self.img_model_adv.range_dict[key]['low']
            rng_high = self.img_model_adv.range_dict[key]['high']
            limit_low_norm = self.img_model_adv.limit_dict[key]['low']
            limit_high_norm = self.img_model_adv.limit_dict[key]['high']
            limit_low = rng_low + (rng_high - rng_low) * limit_low_norm / 100.0
            limit_high = rng_low + (rng_high - rng_low) * limit_high_norm / 100.0
            range_table.append([key, rng_low, rng_high])
            limit_table.append([key, limit_low, limit_high])
            show_table.append([key, bool(self.img_model_adv.stat_dict[key])])

        return range_table, limit_table, show_table

    def set_maps_limit_table(self, limit_table, show_table):
        """
        Write updated range limits to 'img_model_adv.limit_dict'. Used by 'Image Wizard'.

        Parameters
        ----------
        limit_table: list(list)
            The table is represented as list of lists. Every element of the outer list is
            a table row. Each row has 3 elements: 0 - key (eline or map name, str),
            1 - low limit (float), 2 - high limit (float).
        """
        # Verify: the keys in both tables must match 'self.img_model_adv.map_keys'
        limit_table_keys = [_[0] for _ in limit_table]
        if set(limit_table_keys) != set(self.img_model_adv.map_keys):
            raise ValueError("GlobalProcessingClasses:set_maps_info_table: keys don't match:"
                             f"limit_table has keys {limit_table_keys}, "
                             f"original keys {self.img_model_adv.map_keys}")

        show_table_keys = [_[0] for _ in show_table]
        if set(show_table_keys) != set(self.img_model_adv.map_keys):
            raise ValueError("GlobalProcessingClasses:set_maps_info_table: keys don't match:"
                             f"show_table has keys {show_table_keys}, "
                             f"original keys {self.img_model_adv.map_keys}")

        # Copy limits
        for row in limit_table:
            key, v_low, v_high = row

            rng_low = self.img_model_adv.range_dict[key]['low']
            rng_high = self.img_model_adv.range_dict[key]['high']

            v_low_norm = (v_low - rng_low) / (rng_high - rng_low) * 100.0
            v_high_norm = (v_high - rng_low) / (rng_high - rng_low) * 100.0

            self.img_model_adv.limit_dict[key]['low'] = v_low_norm
            self.img_model_adv.limit_dict[key]['high'] = v_high_norm

        # Copy 'show' status (whether the map should be shown)
        for row in show_table:
            key, show_status = row
            self.img_model_adv.stat_dict[key] = bool(show_status)

    def get_maps_dataset_list(self):
        dsets = list(self.img_model_adv.img_dict_keys)
        dset_sel = self.get_maps_selected_dataset()  # The index in the list + 1 (0 - nothing is selected)
        return dsets, dset_sel

    def get_maps_scaler_list(self):
        "Return the list of available scalers for maps"
        scalers = list(self.img_model_adv.scaler_items)
        scaler_sel = self.get_maps_scaler_index()  # The index in the list + 1 (0 - nothing is selected)
        return scalers, scaler_sel

    def get_maps_selected_dataset(self):
        """Returns selected dataset (XRF Maps tab). Index: 0 - no dataset is selected,
        1, 2, ... datasets with index 0, 1, ... is selected"""
        return int(self.img_model_adv.data_opt)

    def set_maps_selected_dataset(self, dataset_index):
        """Select dataset (XRF Maps tab)"""
        self.img_model_adv.select_dataset(dataset_index)

    def get_maps_scaler_index(self):
        return self.img_model_adv.scaler_name_index

    def set_maps_scaler_index(self, scaler_index):
        self.img_model_adv.set_scaler_index(scaler_index)

    def get_maps_quant_norm_enabled(self):
        return bool(self.img_model_adv.quantitative_normalization)

    def set_maps_quant_norm_enabled(self, enable):
        self.img_model_adv.enable_quantitative_normalization(enable)

    def get_maps_scale_opt(self):
        """Returns selected plot type: `Linear` or `Log` """
        return self.img_model_adv.scale_opt

    def set_maps_scale_opt(self, scale_opt):
        """Set plot type. Allowed values: `Linear` and `Log` """
        self.img_model_adv.set_scale_opt(scale_opt)

    def get_maps_color_opt(self):
        """Returns selected plot color option: `Linear` or `Log` """
        return self.img_model_adv.color_opt

    def set_maps_color_opt(self, color_opt):
        """Set plot type. Allowed values: `Linear` and `Log` """
        self.img_model_adv.set_color_opt(color_opt)

    def get_maps_pixel_or_pos(self):
        """Get the current axes options for the plotted maps. Returned values:
        `Pixes` or `Positions`."""
        values = ["Pixels", "Positions"]
        return values[self.img_model_adv.pixel_or_pos]

    def set_maps_pixel_or_pos(self, pixel_or_pos):
        values = ["Pixels", "Positions"]
        self.img_model_adv.set_pixel_or_pos(values.index(pixel_or_pos))

    def get_maps_show_scatter_plot(self):
        return self.img_model_adv.scatter_show

    def set_maps_show_scatter_plot(self, is_scatter):
        self.img_model_adv.set_plot_scatter(is_scatter)

    def get_maps_grid_interpolate(self):
        return self.img_model_adv.grid_interpolate

    def set_maps_grid_interpolate(self, grid_interpolate):
        self.img_model_adv.set_grid_interpolate(grid_interpolate)

    def compute_map_ranges(self):
        self.img_model_adv.set_low_high_value()

    def redraw_maps(self):
        """Redraw maps in XRF Maps tab"""
        self.img_model_adv.show_image()

    # ==========================================================================
    #          The following methods are used by widgets RGB tab
    #                  to read/write data from model classes

    def get_rgb_maps_info_table(self):
        """
        The function builds and returns two tables: a table of value ranges for each map
        in the dataset; a table of limit values that represent selection for the displayed
        values for each map of the dataset.

        Returns
        -------
        range_table: list(list)
            The table is represented as list of lists. Every element of the outer list is
            a table row. Each row has 3 elements: 0 - key (emission line or map name, str),
            1 - lower boundary (float), 2 - upper boundary (float).
        limit_table: list(list)
            The table is represented as list of lists. Every element of the outer list is
            a table row. Each row has 3 elements: 0 - key (emission line or map name, str),
            1 - lower limit value (float), 2 - upper limit value (float).
        rgb_dict: dict
            dictionary that hold the displayed items: key - color ("red", "green" or "blue"),
            value - selected map represented by the key of `self.img_model_rgb.dict_to_plot`
            or `self.img_model_rgb.img_dict[<dataset>]` dictionaries.
        """
        # Check if 'range_dict' and 'limit_dict' have the same set of keys
        ks = self.img_model_rgb.map_keys
        ks_range = list(self.img_model_rgb.range_dict.keys())
        ks_limit = list(self.img_model_rgb.limit_dict.keys())
        rgb_dict = self.img_model_rgb.rgb_dict
        if set(ks) != set(ks_limit):
            raise RuntimeError("The list of keys in 'limit_dict' is not as expected: "
                               f"limit_dict.keys(): {ks_limit} expected: {ks}")
        if set(ks) != set(ks_range):
            raise RuntimeError("The list of keys in 'range_dict' is not as expected: "
                               f"range_dict.keys(): {ks_range} expected: {ks}")
        if len(rgb_dict) != len(self.img_model_rgb.rgb_keys) or \
                set(rgb_dict.keys()) != set(self.img_model_rgb.rgb_keys):
            raise ValueError("GlobalProcessingClasses.get_rgb_maps_info_table(): "
                             f"incorrect set of keys in 'rgb_dict': {list(rgb_dict.keys())}, "
                             f"expected: {self.img_model_rgb.rgb_keys}")
        for v in rgb_dict.values():
            if (v is not None) and (v not in ks):
                raise ValueError("GlobalProcessingClasses.get_rgb_maps_info_table(): "
                                 f"Invalid key {v}. Allowed keys: {ks}")

        range_table = []
        limit_table = []
        for key in ks:
            rng_low = self.img_model_rgb.range_dict[key]['low']
            rng_high = self.img_model_rgb.range_dict[key]['high']
            limit_low_norm = self.img_model_rgb.limit_dict[key]['low']
            limit_high_norm = self.img_model_rgb.limit_dict[key]['high']
            limit_low = rng_low + (rng_high - rng_low) * limit_low_norm / 100.0
            limit_high = rng_low + (rng_high - rng_low) * limit_high_norm / 100.0
            range_table.append([key, rng_low, rng_high])
            limit_table.append([key, limit_low, limit_high])

        return range_table, limit_table, rgb_dict

    def set_rgb_maps_limit_table(self, limit_table, rgb_dict):
        """
        Write updated range limits to 'img_model_adv.limit_dict'. Used by 'Image Wizard'.

        Parameters
        ----------
        limit_table: list(list)
            The table is represented as list of lists. Every element of the outer list is
            a table row. Each row has 3 elements: 0 - key (eline or map name, str),
            1 - low limit (float), 2 - high limit (float).
        rgb_dict: dict
            dictionary that hold the displayed items: key - color ("red", "green" or "blue"),
            value - selected map represented by the key of `self.img_model_rgb.dict_to_plot`
            or `self.img_model_rgb.img_dict[<dataset>]` dictionaries.
        """
        # Verify: the keys in both tables must match 'self.img_model_adv.map_keys'
        limit_table_keys = [_[0] for _ in limit_table]
        if set(limit_table_keys) != set(self.img_model_rgb.map_keys):
            raise ValueError("GlobalProcessingClasses:set_maps_info_table: keys don't match:"
                             f"limit_table has keys {limit_table_keys}, "
                             f"original keys {self.img_model_rgb.map_keys}")

        # Copy limits
        for row in limit_table:
            key, v_low, v_high = row

            rng_low = self.img_model_rgb.range_dict[key]['low']
            rng_high = self.img_model_rgb.range_dict[key]['high']

            v_low_norm = (v_low - rng_low) / (rng_high - rng_low) * 100.0
            v_high_norm = (v_high - rng_low) / (rng_high - rng_low) * 100.0

            self.img_model_rgb.limit_dict[key]['low'] = v_low_norm
            self.img_model_rgb.limit_dict[key]['high'] = v_high_norm

        self.img_model_rgb.rgb_dict = rgb_dict.copy()

    def get_rgb_maps_dataset_list(self):
        dsets = list(self.img_model_rgb.img_dict_keys)
        dset_sel = self.get_maps_selected_dataset()  # The index in the list + 1 (0 - nothing is selected)
        return dsets, dset_sel

    def get_rgb_maps_scaler_list(self):
        "Return the list of available scalers for maps"
        scalers = list(self.img_model_rgb.scaler_items)
        scaler_sel = self.get_maps_scaler_index()  # The index in the list + 1 (0 - nothing is selected)
        return scalers, scaler_sel

    def get_rgb_maps_selected_dataset(self):
        """Returns selected dataset (XRF Maps tab). Index: 0 - no dataset is selected,
        1, 2, ... datasets with index 0, 1, ... is selected"""
        return int(self.img_model_rgb.data_opt)

    def set_rgb_maps_selected_dataset(self, dataset_index):
        """Select dataset (XRF Maps tab)"""
        self.img_model_rgb.select_dataset(dataset_index)

    def get_rgb_maps_scaler_index(self):
        return self.img_model_rgb.scaler_name_index

    def set_rgb_maps_scaler_index(self, scaler_index):
        self.img_model_rgb.set_scaler_index(scaler_index)

    def get_rgb_maps_pixel_or_pos(self):
        """Get the current axes options for the plotted maps. Returned values:
        `Pixes` or `Positions`."""
        values = ["Pixels", "Positions"]
        return values[self.img_model_rgb.pixel_or_pos]

    def set_rgb_maps_pixel_or_pos(self, pixel_or_pos):
        values = ["Pixels", "Positions"]
        self.img_model_rgb.set_pixel_or_pos(values.index(pixel_or_pos))

    def get_rgb_maps_grid_interpolate(self):
        return self.img_model_rgb.grid_interpolate

    def set_rgb_maps_grid_interpolate(self, grid_interpolate):
        self.img_model_rgb.set_grid_interpolate(grid_interpolate)

    def get_rgb_maps_quant_norm_enabled(self):
        return bool(self.img_model_rgb.quantitative_normalization)

    def set_rgb_maps_quant_norm_enabled(self, enable):
        self.img_model_rgb.enable_quantitative_normalization(enable)

    def compute_rgb_map_ranges(self):
        self.img_model_rgb.set_low_high_value()

    def redraw_rgb_maps(self):
        """Redraw maps in XRF Maps tab"""
        self.img_model_rgb.show_image()

    # ==========================================================================
    #          The following methods are used by widgets related
    #      to loading and management of quantitative calibration data
    def load_quantitative_calibration_data(self, file_path):
        self.img_model_adv.load_quantitative_calibration_data(file_path)

    def get_quant_calibration_data(self):
        return self.img_model_adv.param_quant_analysis.calibration_data

    def get_quant_calibration_settings(self):
        return self.img_model_adv.param_quant_analysis.calibration_settings

    def get_quant_calibration_text_preview(self, file_path):
        return self.img_model_adv.param_quant_analysis.get_entry_text_preview(file_path)

    def quant_calibration_remove_entry(self, file_path):
        return self.img_model_adv.param_quant_analysis.remove_entry(file_path)

    def get_quant_calibration_file_path_list(self):
        return self.img_model_adv.param_quant_analysis.get_file_path_list()

    def get_quant_calibration_is_eline_selected(self, eline, file_path):
        return self.img_model_adv.param_quant_analysis.is_eline_selected(eline, file_path)

    def set_quant_calibration_select_eline(self, eline, file_path):
        self.img_model_adv.param_quant_analysis.select_eline(eline, file_path)

    def get_quant_calibration_distance_to_sample(self):
        return self.img_model_adv.param_quant_analysis.experiment_distance_to_sample

    def set_quant_calibration_distance_to_sample(self, distance_to_sample):
        self.img_model_adv.param_quant_analysis.set_experiment_distance_to_sample(distance_to_sample)

    # ==========================================================================
    #    The following methods are used by Fitting Model tab (pectrum plots)

    def get_line_plot_state(self):
        plot_spectrum = bool(self.plot_model.plot_exp_opt)
        plot_fit = bool(self.plot_model.show_fit_opt)
        return plot_spectrum, plot_fit

    def show_plot_spectrum(self, state):
        self.plot_model.plot_exp_opt = bool(state)

    def show_plot_fit(self, state):
        self.plot_model.show_fit_opt = bool(state)

    def get_plot_fit_energy_range(self):
        return self.plot_model.energy_range_fitting

    def set_plot_fit_energy_range(self, range_name):
        if range_name not in self.plot_model.energy_range_names:
            raise ValueError(f"Range name {range_name} is not in the list of allowed "
                             f"names {self.plot_model.energy_range_name}")
        self.plot_model.set_energy_range_fitting(range_name)

    def get_plot_fit_linlog(self):
        if self.plot_model.scale_opt == 0:
            return "linlog"
        else:
            return "linear"

    def set_plot_fit_linlog(self, scale):
        """Scale may have values 'linlog' or 'linear'.
        Raises `ValueError` if other value is passed."""
        self.plot_model.scale_opt = ["linlog", "linear"].index(scale)

    def get_incident_energy(self):
        return self.plot_model.incident_energy

    def get_escape_peak_params(self):
        plot_escape_peak = bool(self.plot_model.plot_escape_line)
        materials = ["Si", "Ge"]
        try:
            detector_material = materials[self.plot_model.det_materials]
        except Exception:
            detector_material = ""
        return plot_escape_peak, detector_material

    def set_escape_peak_params(self, plot_escape_peak, detector_material):
        materials = ["Si", "Ge"]
        if detector_material not in materials:
            raise ValueError(f"Detector material '{detector_material}' is unknown: {materials}")
        self.plot_model.change_escape_peak_settings(1 if plot_escape_peak else 0,
                                                    materials.index(detector_material))

    # ==========================================================================
    #          The following methods are used by Model tab
    def get_autofind_elements_params(self):
        """Assemble the parameters managed by 'Find Elements in Sample` dialog box."""
        dialog_params = {}
        keys1 = ["e_offset", "e_linear", "e_quadratic",
                 "fwhm_offset", "fwhm_fanoprime",
                 "coherent_sct_energy"]
        for k in keys1:
            dialog_params[k] = {}
            dialog_params[k]["value"] = self.param_model.param_new[k]["value"]
            dialog_params[k]["default"] = self.param_model.param_new[k]["default"]
        keys2 = ["energy_bound_high", "energy_bound_low"]
        for k in keys2:
            dialog_params[k] = {}
            dialog_params[k]["value"] = self.param_model.param_new["non_fitting_values"][k]["value"]
            dialog_params[k]["default"] = self.param_model.param_new["non_fitting_values"][k]["default_value"]

        # Some values were taken from different places
        dialog_params["coherent_sct_energy"]["value"] = self.plot_model.incident_energy

        return dialog_params

    def set_autofind_elements_params(self, dialog_params):
        """Save the parameters changed in 'Find Elements in Sample` dialog box.

        Returns
        -------
        boolean
            True - selected range or incident energy were changed, False otherwise
        """
        keys = ["e_offset", "e_linear", "e_quadratic",
                "fwhm_offset", "fwhm_fanoprime"]
        for k in keys:
            self.param_model.param_new[k]["value"] = dialog_params[k]["value"]
        # Check if the critical parameters were changed
        if (self.plot_model.incident_energy != dialog_params["coherent_sct_energy"]["value"]) or \
                (self.param_model.energy_bound_high_buf != dialog_params["energy_bound_high"]["value"]) or \
                (self.param_model.energy_bound_low_buf != dialog_params["energy_bound_low"]["value"]):
            return_value = True
        else:
            return_value = False
        self.io_model.incident_energy_set = dialog_params["coherent_sct_energy"]["value"]
        self.param_model.energy_bound_high_buf = dialog_params["energy_bound_high"]["value"]
        self.param_model.energy_bound_low_buf = dialog_params["energy_bound_low"]["value"]
        return return_value

    def get_quant_standard_list(self):
        self.fit_model.param_quant_estimation.clear_standards()
        self.fit_model.param_quant_estimation.load_standards()

        qe_param_built_in = self.fit_model.param_quant_estimation.standards_built_in
        qe_param_custom = self.fit_model.param_quant_estimation.standards_custom

        # The current selection is set to the currently selected standard
        #   held in 'fit_model.qe_standard_selected_copy'
        qe_standard_selected = self.fit_model.qe_standard_selected_copy
        qe_standard_selected = \
            self.fit_model.param_quant_estimation.set_selected_standard(qe_standard_selected)

        return qe_param_built_in, qe_param_custom, qe_standard_selected

    def set_selected_quant_standard(self, selected_standard):
        if selected_standard is not None:
            self.fit_model.param_quant_estimation.set_selected_standard(selected_standard)
            self.fit_model.qe_standard_selected_copy = copy.deepcopy(selected_standard)
        else:
            self.fit_model.param_quant_estimation.clear_standards()
            self.fit_model.qe_standard_selected_copy = None

    def is_quant_standard_custom(self, standard=None):
        return self.fit_model.param_quant_estimation.is_standard_custom(standard)

    def process_peaks_from_quantitative_sample_data(self):
        incident_energy = self.param_model.param_new['coherent_sct_energy']['value']
        # Generate the data structure for the results of processing of standard data
        self.fit_model.param_quant_estimation.gen_fluorescence_data_dict(incident_energy)
        # Obtain the list (dictionary) of elemental lines from the generated structure
        elemental_lines = self.fit_model.param_quant_estimation.fluorescence_data_dict["element_lines"]

        self.param_model.find_peak(elemental_lines=elemental_lines.keys())
        self.param_model.EC.order()
        self.param_model.update_name_list()

        self.param_model.EC.turn_on_all()
        self.param_model.data_for_plot()

        # update experimental plots in case the coefficients change
        self.plot_model.parameters = self.param_model.param_new
        self.plot_model.plot_experiment()

        self.plot_model.plot_fit(self.param_model.prefit_x,
                                 self.param_model.total_y,
                                 self.param_model.auto_fit_all)

        # Update displayed intensity of the selected peak
        self.plot_model.compute_manual_peak_intensity()

        # Show the summed spectrum used for fitting
        self.plot_model.plot_exp_opt = False
        self.plot_model.plot_exp_opt = True
        # For plotting purposes, otherwise plot will not update
        self.plot_model.show_fit_opt = False
        self.plot_model.show_fit_opt = True

    def load_parameters_from_file(self, parameter_file_path, ask_question):

        try:
            self.fit_model.read_param_from_file(parameter_file_path)
        except Exception as ex:
            msg = f"Error occurred while reading parameter file: {ex}"
            logger.error(msg)
            raise IOError(msg)
        else:
            # Make a decision, if the incident energy from metadata should be replaced
            #   with the incident energy from the parameter file
            overwrite_metadata_incident_energy = False

            # Incident energy from the parameter file
            param_incident_energy = self.fit_model.default_parameters['coherent_sct_energy']['value']

            if self.io_model.incident_energy_available:

                # Incident energy from datafile metadata
                mdata_incident_energy = self.io_model.scan_metadata.get_mono_incident_energy()

                # If two energies have very close values (say 1 eV), then the difference doesn't matter
                #   Consider two energies equal (they probably ARE equal) and use the value
                #   from datafile metadata.
                if not math.isclose(param_incident_energy, mdata_incident_energy, abs_tol=0.001):
                    # TODO: the following text is not properly formatted for QMessageBox
                    #       It's not very important now, but should be resolved in the future.
                    msg = f"The values of incident energy from data file metadata " \
                          f"and parameter file are different.\n" \
                          f"Incident energy from metadata: {mdata_incident_energy} keV.\n" \
                          f"Incident energy from the loaded parameter file: {param_incident_energy} keV.\n" \
                          f"Would you prefer to use the incident energy from the parameter file for processing?"

                    question = ask_question(msg)
                    if question():
                        overwrite_metadata_incident_energy = True

            else:

                # If incident energy is not present in file metadata, then
                #   just load the incident energy from the parameter file.
                overwrite_metadata_incident_energy = True

            if overwrite_metadata_incident_energy:
                logger.info(f"Using incident energy from the parameter file: {param_incident_energy} keV")
            else:
                # Keep the incident energy from the file
                logger.info(f"Using incident energy from the datafile metadata: "
                            f"{mdata_incident_energy} keV")
                incident_energy = round(mdata_incident_energy, 6)
                self.fit_model.default_parameters["coherent_sct_energy"]["value"] = incident_energy
                self.fit_model.default_parameters["non_fitting_values"]["energy_bound_high"]["value"] = \
                    incident_energy + 0.8

            self.fit_model.apply_default_param()

            # update experimental plots
            self.plot_model.parameters = self.fit_model.default_parameters
            self.plot_model.plot_experiment()
            self.plot_model.plot_exp_opt = False
            self.plot_model.plot_exp_opt = True

            # update autofit param
            self.param_model.update_new_param(self.fit_model.default_parameters)
            # param_model.get_new_param_from_file(parameter_file_path)

            self.param_model.EC.order()
            self.param_model.update_name_list()
            self.param_model.EC.turn_on_all()
            self.param_model.data_for_plot()

            # update params for roi sum
            self.setting_model.update_parameter(self.fit_model.default_parameters)

            # calculate profile and plot
            self.fit_model.get_profile()

            # update experimental plot with new calibration values
            self.plot_model.plot_fit(self.fit_model.cal_x, self.fit_model.cal_y,
                                     self.fit_model.cal_spectrum,
                                     self.fit_model.residual)

            self.plot_model.plot_exp_opt = False
            self.plot_model.plot_exp_opt = True
            # For plotting purposes, otherwise plot will not update
            self.plot_model.show_fit_opt = False
            self.plot_model.show_fit_opt = True

            # The following statement is necessary mostly to set the correct value of
            #   the upper boundary of the energy range used for emission line search.
            self.plot_model.change_incident_energy(
                self.fit_model.default_parameters["coherent_sct_energy"]["value"])

            # update parameter for fit
            # self.param_model.create_full_param()
            self.fit_model.update_default_param(self.param_model.param_new)
            self.fit_model.apply_default_param()

            # update params for roi sum
            self.setting_model.update_parameter(self.fit_model.param_dict)

            # Update displayed intensity of the selected peak
            self.plot_model.compute_manual_peak_intensity()

    def find_elements_automatically(self):
        self.param_model.find_peak()

        self.param_model.EC.order()
        self.param_model.update_name_list()

        self.param_model.EC.turn_on_all()

        self.plot_model.plot_exp_opt = True
        self.apply_to_fit()

    def get_full_eline_list(self):
        """Returns full list of supported emission lines."""
        return self.param_model.get_user_peak_list()

    def set_marker_reporter(self, callback):
        self.plot_model.report_marker_state = callback

    def select_eline(self, eline):
        self.param_model.e_name = eline
        n_id = 0  # No element is selected
        if eline != "":
            eline_list = self.get_full_eline_list()
            try:
                n_id = eline_list.index(eline) + 1  # Elements are numbered starting with 1.
            except ValueError:
                pass
        self.plot_model.element_id = n_id

    def set_selected_eline(self, eline):
        """Sets and display the location of current emission line"""
        self.select_eline(eline)

        eline_category = self.get_eline_name_category(eline)
        if eline_category == "userpeak":
            self.fit_model.select_index_by_eline_name(eline)
            # This will delete the peak lines (no peak lines are plotted for user peak)
            self.plot_model.plot_current_eline(eline)
            # Show the marker at the center of the userpeak
            energy, _ = self.param_model.get_selected_eline_energy_fwhm(eline)
            self.plot_model.set_plot_vertical_marker(energy)
        elif eline_category == "pileup":
            # The lines for pileup peak are not displayed as 'plot_model.element_id' is changed.
            #   So call plotting function explicitly
            self.plot_model.plot_current_eline(eline)
            self.plot_model.hide_plot_vertical_marker()
        elif eline_category == "eline":
            self.plot_model.hide_plot_vertical_marker()
        else:
            # This will delete the peak lines
            self.plot_model.plot_current_eline(eline)
            self.plot_model.hide_plot_vertical_marker()

    def get_eline_intensity(self, eline):
        line_category = self.get_eline_name_category(eline)
        if line_category == "eline":
            return self.param_model.add_element_intensity
        elif line_category in ("userpeak", "pileup"):
            return self.param_model.EC.element_dict[eline].maxv
        else:
            return None

    def get_selected_eline_table(self):
        """Returns full list of supported emission lines."""
        eline_names = self.param_model.get_sorted_result_dict_names()
        eline_table = []
        for eline in eline_names:
            sel_status = self.param_model.EC.element_dict[eline].status
            z = self.param_model.EC.element_dict[eline].z
            energy = self.param_model.EC.element_dict[eline].energy
            peak_int = self.param_model.EC.element_dict[eline].maxv
            rel_int = self.param_model.EC.element_dict[eline].norm
            cs = get_cs(eline, self.fit_model.param_dict['coherent_sct_energy']['value'])
            row_data = {"eline": eline, "sel_status": sel_status, "z": z,
                        "energy": energy, "peak_int": peak_int, "rel_int": rel_int, "cs": cs}
            eline_table.append(row_data)
        return eline_table

    def get_unused_userpeak_name(self):
        eline_names = self.param_model.get_sorted_result_dict_names()
        for n in range(1, 10000):  # Set some arbitrarily high limit to avoid infinite loops
            userpeak_name = f"Userpeak{n}"
            if userpeak_name not in eline_names:
                return userpeak_name

    def get_eline_name_category(self, eline_name):
        return self.param_model.get_eline_name_category(eline_name)

    def set_checked_emission_lines(self, eline_list, eline_checked_list):
        for eline, checked in zip(eline_list, eline_checked_list):
            self.param_model.EC.element_dict[eline].status = checked
            self.param_model.EC.element_dict[eline].stat_copy = \
                self.param_model.EC.element_dict[eline].status

        self.param_model.data_for_plot()

        self.plot_model.plot_fit(self.param_model.prefit_x,
                                 self.param_model.total_y,
                                 self.param_model.auto_fit_all)
        # For plotting purposes, otherwise plot will not update
        self.plot_model.show_fit_opt = False
        self.plot_model.show_fit_opt = True

    def get_current_userpeak_energy_fwhm(self):
        return self.param_model.get_selected_eline_energy_fwhm(self.param_model.e_name)

    def generate_pileup_peak_name(self, eline1, eline2):
        return self.param_model.generate_pileup_peak_name(eline1, eline2)

    def get_pileup_peak_energy(self, eline):
        return self.param_model.get_pileup_peak_energy(eline)

    def is_peak_already_selected(self, eline):
        """Verifies if emission line with the same name is already selected"""
        line_list = list(self.param_model.EC.element_dict.keys())
        if eline in line_list:
            return True
        else:
            return False

    def convert_full_eline_name(self, eline, to_upper=True):
        """Convert emission lines to the form used for displayed emission line names"""
        try:
            index = eline.index("_")
            eline_as_list = list(eline)
            if to_upper:
                eline_as_list[index + 1] = eline_as_list[index + 1].upper()
            else:
                eline_as_list[index + 1] = eline_as_list[index + 1].lower()
            eline = "".join(eline_as_list)
        except Exception:
            pass
        return eline

    def get_guessed_pileup_peak_components(self, energy, tolerance=0.05):
        """
        Returns tuple (line1, line2, pileup_energy), that contains the best guess
        for the pileup peak placed `energy`.

        Parameters
        ----------
        energy: float
            Approximate (selected) energy of pileup peak location
        tolerance: float
            Allowed deviation of the sum of component energies from the selected energy, keV

        Returns
        -------
        tuple(str, str, float)
            Component emission lines (such as Ca_ka1, K_ka1 etc) and the energy of
            the resulting pileup peak.
        """
        eline_info = self.param_model.guess_pileup_peak_components(energy=energy, tolerance=tolerance)
        if eline_info is not None:
            e1, e2, energy = eline_info
            eline_info = (self.convert_full_eline_name(e1, to_upper=True),
                          self.convert_full_eline_name(e2, to_upper=True),
                          energy)
        return eline_info

    def get_selected_energy_range(self):
        """
        Returns the energy range selected for processing. No peaks are supposed to be selected
        outside this range.

        Returns
        -------
        tuple(float, float)
            Selected range (low, high) in keV
        """
        e_low = self.param_model.param_new['non_fitting_values']['energy_bound_low']['value']
        e_high = self.param_model.param_new['non_fitting_values']['energy_bound_high']['value']
        return e_low, e_high

    def add_peak_manual(self, eline):
        """Manually add a peak (emission line) using 'Add' button"""
        self.select_eline(eline)

        # Verify if we are adding a userpeak
        is_userpeak = self.get_eline_name_category(eline) == "userpeak"

        # The following set of conditions is not complete, but sufficient
        if self.param_model.x0 is None or self.param_model.y0 is None:
            err_msg = "Experimental data is not loaded or initial\n" \
                      "spectrum fitting is not performed"
            raise RuntimeError(err_msg)

        elif is_userpeak and (not self.plot_model.vertical_marker_is_visible):
            err_msg = ("Select position of userpeak by clicking on the spectrum plot.\n"
                       "Note: plot toolbar options, such as Pan and Zoom, \n"
                       "must be turned off before selecting the position.")
            raise RuntimeError(err_msg)

        else:
            try:
                # Energy is used for user-defined peaks only
                energy, _ = self.get_suggested_manual_peak_energy()

                self.param_model.add_peak_manual(userpeak_center=energy)
                self.apply_to_fit()
                self.fit_model.select_index_by_eline_name(eline)
            except Exception as ex:
                raise RuntimeError(str(ex))

    def remove_peak_manual(self, eline):
        """Manually add a peak (emission line) using 'Remove' button"""
        try:
            self.select_eline(eline)
            self.param_model.remove_peak_manual()
            self.apply_to_fit()
        except Exception as ex:
            raise RuntimeError(str(ex))

    def update_userpeak(self, eline, energy, maxv, fwhm):
        """
        Update parameters of the user defined peak. Exception may be raised in
        case of an error. Note, that if energy is changed, FWHM will change to
        adjust for the variation of peak width.

        Parameters
        ----------
        eline: str
            name of the userpeak (e.g. 'Userpeak2')
        energy: float
            new position of the peak center, keV
        maxv: float
            new value of the amplitude
        fwhm: float
            new value of FWHM of the peak.
        """
        self.select_eline(eline)
        self.param_model.modify_userpeak_params(maxv, fwhm, energy)
        self.apply_to_fit()

    def update_eline_peak_height(self, eline, maxv):
        self.select_eline(eline)
        self.param_model.modify_peak_height(maxv)
        self.apply_to_fit()

    def get_suggested_manual_peak_energy(self):
        """
        Returns the suggested (pointed by the vertical marker) energy of the center peak in keV
        and the status of the marker.

        Returns
        -------
        float
            Energy of the manual peak center in keV. The energy is determined
            by vertical marker on the screen.
        bool
            True if the vertical marker is visible, otherwise False.

        """
        return self.plot_model.get_suggested_new_manual_peak_energy()

    def show_marker_at_current_position(self):
        """Force marker to visible state at the current position"""
        energy, marker_visible = self.get_suggested_manual_peak_energy()
        self.plot_model.set_plot_vertical_marker(energy)

    def get_peak_threshold(self):
        return self.param_model.bound_val

    def remove_peaks_below_threshold(self, peak_threshold):
        self.param_model.bound_val = peak_threshold
        self.param_model.remove_elements_below_threshold(threshv=peak_threshold)
        self.param_model.update_name_list()
        self.apply_to_fit()

    def remove_unchecked_peaks(self):
        self.param_model.remove_elements_unselected()
        self.param_model.update_name_list()
        self.apply_to_fit()

    def apply_to_fit(self):
        """
        Update plot, and apply updated parameters to fitting process.
        Note: this is an original function from 'fit.enaml' file.
        """
        self.param_model.EC.update_peak_ratio()
        self.param_model.data_for_plot()

        # update experimental plots in case the coefficients change
        self.plot_model.parameters = self.param_model.param_new
        self.plot_model.plot_experiment()

        self.plot_model.plot_fit(self.param_model.prefit_x,
                                 self.param_model.total_y,
                                 self.param_model.auto_fit_all)

        # For plotting purposes, otherwise plot will not update
        if self.plot_model.plot_exp_opt:
            self.plot_model.plot_exp_opt = False
            self.plot_model.plot_exp_opt = True
        else:
            self.plot_model.plot_exp_opt = True
            self.plot_model.plot_exp_opt = False
        self.plot_model.show_fit_opt = False
        self.plot_model.show_fit_opt = True

        # update parameter for fit
        # self.param_model.create_full_param()
        self.fit_model.update_default_param(self.param_model.param_new)
        self.fit_model.apply_default_param()

        # update params for roi sum
        self.setting_model.update_parameter(self.fit_model.param_dict)

        # Update displayed intensity of the selected peak
        self.plot_model.compute_manual_peak_intensity()

    '''
    def calculate_spectrum_helper(self):
        """
        Calculate spectrum, and update plotting and param_model.
        Note: this is an original function from 'fit.enaml' file.
        """
        if self.fit_model.x0 is None or self.fit_model.y0 is None:
            return

        self.fit_model.get_profile()

        # update experimental plot with new calibration values
        self.plot_model.parameters = self.fit_model.param_dict
        self.plot_model.plot_experiment()

        self.plot_model.plot_fit(self.fit_model.cal_x, self.fit_model.cal_y,
                                 self.fit_model.cal_spectrum,
                                 self.fit_model.residual)

        # For plotting purposes, otherwise plot will not update
        self.plot_model.show_fit_opt = False
        self.plot_model.show_fit_opt = True

        # update autofit param
        self.param_model.update_new_param(self.fit_model.param_dict)
        self.param_model.update_name_list()
        self.param_model.EC.turn_on_all()

        # update params for roi sum
        self.setting_model.update_parameter(self.fit_model.param_dict)
    '''

    def total_spectrum_fitting(self):

        # Update parameter for fit. This may be unnecessary
        self.apply_to_fit()

        self.fit_model.fit_multiple()

        # BUG: line color for pileup is not correct from fit
        self.fit_model.get_profile()

        # update experimental plot with new calibration values
        self.plot_model.parameters = self.fit_model.param_dict
        self.plot_model.plot_experiment()

        self.plot_model.plot_fit(self.fit_model.cal_x, self.fit_model.cal_y,
                                 self.fit_model.cal_spectrum,
                                 self.fit_model.residual)

        # For plotting purposes, otherwise plot will not update
        self.plot_model.plot_exp_opt = False
        self.plot_model.plot_exp_opt = True
        self.plot_model.show_fit_opt = False
        self.plot_model.show_fit_opt = True

        # update autofit param
        self.param_model.update_new_param(self.fit_model.param_dict)
        # param_model.get_new_param_from_file(parameter_file_path)
        self.param_model.update_name_list()
        self.param_model.EC.turn_on_all()

        # update params for roi sum
        self.setting_model.update_parameter(self.fit_model.param_dict)

        # Update displayed intensity of the selected peak
        self.plot_model.compute_manual_peak_intensity()

    def save_param_to_file(self, path):
        save_as(path, self.fit_model.param_dict)

    def save_spectrum(self, dir, save_fit):
        self.fit_model.result_folder = dir
        self.fit_model.output_summed_data_fit(save_fit=save_fit)

    def compute_current_rfactor(self, save_fit):
        return self.fit_model.compute_current_rfactor(save_fit)

    def get_iter_and_var_number(self):
        return {"var_number": self.fit_model.nvar,
                "iter_number": self.fit_model.function_num}

    # ==========================================================================
    #          The following methods are used by Maps tab
    def fit_individual_pixels(self):
        self.apply_to_fit()

        self.fit_model.fit_single_pixel()

        # add scalers to fit dict
        scaler_keys = [v for v in self.img_model_adv.img_dict.keys() if 'scaler' in v]
        if len(scaler_keys) > 0:
            self.fit_model.fit_img[list(self.fit_model.fit_img.keys())[0]].update(
                self.img_model_adv.img_dict[scaler_keys[0]])

        self.img_model_adv.update_img_dict_entries(self.fit_model.fit_img)
        self.img_model_rgb.update_img_dict_entries(self.fit_model.fit_img)
