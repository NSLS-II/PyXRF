from __future__ import absolute_import

import os
from ..model.fileio import FileIOModel
from ..model.lineplot import LinePlotModel  # , SettingModel
from ..model.guessparam import GuessParamModel
from ..model.draw_image import DrawImageAdvanced
from ..model.draw_image_rgb import DrawImageRGB
from ..model.fit_spectrum import Fit1D
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
        self.img_model_rgb = DrawImageRGB()

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
            logger.info(f"Failed to load the file '{f_name}'.")
            # Clear file name or scan id from window title. This does not update
            #   the displayed title.
            self.io_model.window_title_clear()
            raise
        else:
            _update_data()

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

    def redraw_maps(self):
        """Redraw maps in XRF Maps tab"""
        self.img_model_adv.show_image()
