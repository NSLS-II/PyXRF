from __future__ import absolute_import

import os
import copy
import math
from ..model.fileio import FileIOModel
from ..model.lineplot import LinePlotModel
from ..model.parameters import ParamModel, save_as, fit_strategy_list, bound_options
from ..model.draw_image import DrawImageAdvanced
from ..model.draw_image_rgb import DrawImageRGB
from ..model.fit_spectrum import Fit1D, get_cs
from ..model.roi_model import ROIModel
from ..model.param_data import param_data

from ..core.xrf_utils import get_eline_energy

import logging

logger = logging.getLogger(__name__)


class GlobalProcessingClasses:
    def __init__(self):
        self.defaults = None
        self.io_model = None
        self.param_model = None
        self.plot_model = None
        self.fit_model = None
        self.roi_model = None
        self.img_model_adv = None
        self.img_model_rgb = None

    def _get_defaults(self):
        # Set working directory to current working directory (if PyXRF is started from shell)
        working_directory = os.getcwd()
        logger.info(f"Starting PyXRF in the current working directory '{working_directory}'")
        default_parameters = param_data
        return working_directory, default_parameters

    def initialize(self):
        """
        Run the sequence of actions needed to initialize PyXRF modules.
        """
        working_directory, default_parameters = self._get_defaults()
        self.io_model = FileIOModel(working_directory=working_directory)
        self.param_model = ParamModel(default_parameters=default_parameters, io_model=self.io_model)
        self.plot_model = LinePlotModel(param_model=self.param_model, io_model=self.io_model)
        self.fit_model = Fit1D(param_model=self.param_model, io_model=self.io_model)
        self.roi_model = ROIModel(param_model=self.param_model, io_model=self.io_model)
        self.img_model_adv = DrawImageAdvanced(io_model=self.io_model)
        self.img_model_rgb = DrawImageRGB(io_model=self.io_model, img_model_adv=self.img_model_adv)

        # Initialization needed to eliminate program crash
        self.plot_model.roi_dict = self.roi_model.roi_dict

        # send working directory changes to different models
        self.io_model.observe("selected_file_name", self.fit_model.data_title_update)
        self.io_model.observe("selected_file_name", self.plot_model.exp_label_update)
        self.io_model.observe("selected_file_name", self.roi_model.data_title_update)

        # send the same file to fit model, as fitting results need to be saved
        self.io_model.observe("file_path", self.fit_model.filepath_update)
        self.io_model.observe("file_path", self.plot_model.plot_exp_data_update)
        self.io_model.observe("file_path", self.roi_model.filepath_update)
        self.io_model.observe("runid", self.fit_model.runid_update)

        # Perform updates when 'io_model.data' is changed (no data is passed)
        self.io_model.observe("data", self.plot_model.exp_data_update)

        # send img dict to img_model for visualization
        self.io_model.observe("img_dict_is_updated", self.fit_model.img_dict_updated)
        self.io_model.observe("img_dict_is_updated", self.img_model_adv.img_dict_updated)
        self.io_model.observe("img_dict_is_updated", self.img_model_rgb.img_dict_updated)

        self.io_model.observe("incident_energy_set", self.plot_model.set_incident_energy)
        self.io_model.observe("incident_energy_set", self.img_model_adv.set_incident_energy)

        self.img_model_adv.observe("scaler_name_index", self.fit_model.scaler_index_update)

        self.img_model_adv.observe("dict_to_plot", self.fit_model.dict_to_plot_update)
        self.img_model_adv.observe("img_title", self.fit_model.img_title_update)

        self.param_model.observe("energy_bound_high_buf", self.plot_model.energy_bound_high_update)
        self.param_model.observe("energy_bound_low_buf", self.plot_model.energy_bound_low_update)

        logger.info("pyxrf started.")

    def add_parameters_changed_cb(self, cb):
        """
        Add callback to the list of callback function that are called after parameters are updated.
        """
        self.param_model.add_parameters_changed_cb(cb)

    def remove_parameters_changed_cb(self, cb):
        """
        Remove reference from the list of callback functions.
        """
        self.param_model.remove_parameters_changed_cb(cb)

    def fitting_parameters_changed(self):
        self.param_model.parameters_changed()

    def get_window_title(self):
        """
        Returns current title of the Main Window of the application.
        """
        return self.io_model.window_title

    def is_databroker_available(self):
        return self.io_model.is_databroker_available()

    def open_data_file(self, file_path):

        self.io_model.data_ready = False
        # only load one file
        # 'temp' is used to reload the same file, otherwise file_name will not update
        self.io_model.file_path = "temp"
        f_path = file_path

        def _update_data():
            self.fit_model.fit_img = {}  # clear dict in fitmodel to rm old results
            # This will draw empty (hidden) preview plot, since no channels are selected.
            self.plot_model.update_preview_spectrum_plot()
            self.plot_model.update_total_count_map_preview(new_plot=True)

        # The following statement initiates file loading. It may raise exceptions
        try:

            self.io_model.file_path = file_path

        except Exception:
            _update_data()
            self.fitting_parameters_changed()

            # For plotting purposes, otherwise plot will not update
            self.plot_model.plot_exp_opt = False

            self.plot_model.show_fit_opt = False

            logger.info(f"Failed to load the file '{f_path}'.")
            # Clear file name or scan id from window title. This does not update
            #   the displayed title.
            self.io_model.window_title_clear()
            raise
        else:
            _update_data()
            self.fitting_parameters_changed()

            # For plotting purposes, otherwise plot will not update
            try:
                # TODO: permanent fix for the following issue is needed
                # Suppress Matplotlib exception.
                #   draw_glyph_to_bitmap: glyph num is out of range
                # This is a temporary fix, proper debugging is needed so that try..except
                #   could be removed
                self.plot_model.plot_exp_opt = False
            except Exception as ex:
                logger.error(f"Exception was raised while removing experimental data plot: {str(ex)}")
            try:
                self.plot_model.plot_exp_opt = True
            except Exception as ex:
                logger.error(f"Exception was raised while plotting experimental data: {str(ex)}")
            try:
                self.plot_model.show_fit_opt = False
            except Exception as ex:
                logger.error(f"Exception was raised while removing fitted data plot: {str(ex)}")
            try:
                self.plot_model.show_fit_opt = True
            except Exception as ex:
                logger.error(f"Exception was raised while plotting fitted data: {str(ex)}")

            # Change window title (include file name). This does not update the visible title,
            #   only the text attribute of 'io_model' class.
            self.io_model.window_title_set_file_name(os.path.basename(f_path))

            if not self.io_model.incident_energy_available:
                msg = (
                    "Incident energy is not available in scan metadata and must be set manually. "
                    "Incident energy may be set by changing 'Incident energy, keV' parameter "
                    "in the dialog boxes opened using 'Find Automatically...' ('Find Elements "
                    "in sample' or 'General...' ('General Settings for Fitting Alogirthm') "
                    "buttons in 'Model' tab."
                )
            else:
                msg = ""

            logger.info(f"Loading of the file '{file_path}' is completed.")

            return msg

    def load_run_from_db(self, run_id_uid):
        """
        Parameters
        ----------
        run_id_uid: int or str
            int - run ID (e.g. -1, 45326), str - run UID
        """

        # Copy value to 'runid' field ('runid' field should not change
        #   during processing unless new data is loaded)
        self.io_model.data_ready = False

        def _update_data():
            self.fit_model.fit_img = {}  # clear dict in fitmodel to rm old results
            # This will draw empty (hidden) preview plot, since no channels are selected.
            self.plot_model.update_preview_spectrum_plot()
            self.plot_model.update_total_count_map_preview(new_plot=True)

        try:
            self.io_model.load_data_runid(run_id_uid)

        except Exception as ex:
            _update_data()
            self.fitting_parameters_changed()

            # For plotting purposes, otherwise plot will not update
            self.plot_model.plot_exp_opt = False
            self.plot_model.show_fit_opt = False

            logger.info(f"Failed to load the run #{run_id_uid}.")
            logger.exception(ex)

            # Clear file name or scan id from window title. This does not update
            #   the displayed title.
            self.io_model.window_title_clear()
            raise

        else:
            _update_data()
            self.fitting_parameters_changed()

            # For plotting purposes, otherwise plot will not update
            self.plot_model.plot_exp_opt = False
            self.plot_model.plot_exp_opt = True
            self.plot_model.show_fit_opt = False
            self.plot_model.show_fit_opt = True

            # Change window title (include file name). This does not update the visible title,
            #   only the text attribute of 'io_model' class.
            self.io_model.window_title_set_run_id(self.io_model.runid)

            file_name = self.io_model.file_path
            msg = ""

            logger.info("Loading of the run is completed")

            return msg, file_name

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
        self.plot_model.update_preview_spectrum_plot()
        self.plot_model.update_total_count_map_preview()

    def get_current_working_directory(self):
        """
        Return current working directory (defined in 'io_model')
        """
        return self.io_model.current_working_directory

    def get_file_directory(self):
        """
        Return the directory of the last data file that was opened.
        """
        return self.io_model.file_directory

    def set_current_working_directory(self, working_directory):
        """
        Sets current working directory ('io_model')
        """
        self.io_model.current_working_directory = working_directory

    def get_load_each_channel(self):
        """
        Get the value of `load_each_channel` flag, which indicates if data from
        all detector channels will be loaded.
        """
        return self.io_model.load_each_channel

    def set_load_each_channel(self, load_each_channel):
        """
        Get the value of `load_each_channel` flag, which indicates if data from
        all detector channels will be loaded.
        """
        self.io_model.load_each_channel = load_each_channel

    def get_file_channel_list(self):
        """
        Get the list of channels in the currently loaded file (or run)
        """
        return self.io_model.file_channel_list

    def is_dset_item_selected_for_preview(self, item):
        """
        Check if dataset item is selected for preview.
        """
        return bool(self.io_model.data_sets[item].selected_for_preview)

    def get_loaded_file_name(self):
        """
        Returns the name of currently loaded file.
        """
        return self.io_model.file_path

    def is_scan_metadata_available(self):
        """
        Returns the flag that indicates if scan metadata is loaded
        """
        return bool(self.io_model.scan_metadata_available)

    def get_formatted_metadata(self):
        """
        Returns formatted metadata (for displaying or printing)
        """
        return self.io_model.scan_metadata.get_formatted_output()

    def get_metadata_scan_id(self):
        """
        Reads Run ID from metadata. Check if metadata exists using `is_scan_metadata_available()`
        before calling this function.
        """
        if self.is_scan_metadata_available():
            scan_id = self.io_model.scan_metadata["scan_id"]
        else:
            scan_id = 0  # This is mostly for preventing crashes.
        return scan_id

    def get_metadata_scan_uid(self):
        """
        Reads Run UID from metadata. Check if metadata exists using `is_scan_metadata_available()`
        before calling this function.
        """
        if self.is_scan_metadata_available():
            scan_uid = self.io_model.scan_metadata["scan_uid"]
        else:
            scan_uid = ""  # This is mostly for preventing crashes.
        return scan_uid

    def get_current_run_id(self):
        """
        Returns value of currently set Run ID (in `io_model` class)
        """
        return self.io_model.runid

    def is_xrf_maps_available(self):
        """
        Returns the flag that indicates if XRF maps are currently available (loaded or computed).
        """
        return bool(self.io_model.is_xrf_maps_available())

    def set_data_channel(self, channel_index):
        self.io_model.file_opt = channel_index

        # For plotting purposes, otherwise plot will not update
        self.plot_model.plot_exp_opt = False
        self.plot_model.plot_exp_opt = True
        self.plot_model.show_fit_opt = False
        self.plot_model.show_fit_opt = True

    def is_roi_selection_active(self):
        """
        Returns the flag that indicates if spatial ROI is activated.
        """
        return bool(self.io_model.roi_selection_active)

    def set_roi_selection_active(self, active):
        """
        Sets the flag that indicates if spatial ROI is activated.
        """
        self.io_model.roi_selection_active = active

    def get_preview_spatial_roi(self):
        """
        Returns currently selected spatial ROI (for preview)
        """
        spatial_roi = {
            "row_start": self.io_model.roi_row_start,
            "col_start": self.io_model.roi_col_start,
            "row_end": self.io_model.roi_row_end,
            "col_end": self.io_model.roi_col_end,
        }
        return spatial_roi

    def set_preview_spatial_roi(self, spatial_roi):
        """
        Set spatial ROI (for preview)
        """
        self.io_model.roi_row_start = spatial_roi["row_start"]
        self.io_model.roi_col_start = spatial_roi["col_start"]
        self.io_model.roi_row_end = spatial_roi["row_end"]
        self.io_model.roi_col_end = spatial_roi["col_end"]

    def is_mask_selection_active(self):
        """
        Returns the flag that indicates if mask for ROI selection is activated.
        """
        return bool(self.io_model.mask_active)

    def set_mask_selection_active(self, active):
        """
        Sets the flag that indicates if mask for ROI selection is activated.
        """
        self.io_model.mask_active = active

    def get_mask_selection_file_path(self):
        """
        Returns currently selected path to mask file (ROI selection)
        """
        return self.io_model.mask_file_path

    def set_mask_selection_file_path(self, file_path):
        """
        Sets path to mask file (ROI selection). It also sets mask name.
        """
        self.io_model.mask_file_path = file_path
        # At this point the mask name is just the file name
        self.io_model.mask_name = os.path.split(file_path)[-1]

    def apply_mask_to_datasets(self):
        self.io_model.apply_mask_to_datasets()
        self.plot_model.update_preview_spectrum_plot()

    # ==========================================================================
    #        The following methods are used by the preview widgets

    def get_preview_plot_type(self):
        return self.plot_model.plot_type_preview.value

    def set_preview_plot_type(self, plot_type):
        self.plot_model.plot_type_preview = plot_type

    def get_preview_energy_range(self):
        return self.plot_model.energy_range_preview

    def set_preview_energy_range(self, energy_range):
        self.plot_model.energy_range_preview = energy_range

    def update_preview_spectrum_plot(self):
        self.plot_model.update_preview_spectrum_plot()

    def get_preview_map_color_scheme(self):
        return self.plot_model.map_preview_color_scheme

    def set_preview_map_color_scheme(self, color_scheme):
        self.plot_model.map_preview_color_scheme = color_scheme

    def get_preview_map_type(self):
        return self.plot_model.map_type_preview

    def set_preview_map_type(self, map_type):
        self.plot_model.map_type_preview = map_type

    def get_preview_map_axes_units(self):
        return self.plot_model.map_axes_units_preview

    def set_preview_map_axes_units(self, map_axes_units):
        self.plot_model.map_axes_units_preview = map_axes_units

    def get_preview_map_range(self):
        return self.plot_model.map_preview_range_low, self.plot_model.map_preview_range_high

    def set_preview_map_range(self, low, high):
        self.plot_model.set_map_preview_range(low=low, high=high)

    def get_dataset_preview_count_map_range(self):
        return self.io_model.get_dataset_preview_count_map_range()

    def update_preview_total_count_map(self):
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
            raise RuntimeError(
                "The list of keys in 'limit_dict' is not as expected: "
                f"limit_dict.keys(): {ks_limit} expected: {ks}"
            )
        if set(ks) != set(ks_range):
            raise RuntimeError(
                "The list of keys in 'range_dict' is not as expected: "
                f"range_dict.keys(): {ks_range} expected: {ks}"
            )
        if set(ks) != set(ks_show):
            raise RuntimeError(
                "The list of keys in 'stat_dict' is not as expected: "
                f"stat_dict.keys(): {ks_show} expected: {ks}"
            )

        range_table = []
        limit_table = []
        show_table = []
        for key in ks:
            rng_low = self.img_model_adv.range_dict[key]["low"]
            rng_high = self.img_model_adv.range_dict[key]["high"]
            limit_low_norm = self.img_model_adv.limit_dict[key]["low"]
            limit_high_norm = self.img_model_adv.limit_dict[key]["high"]
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
            raise ValueError(
                "GlobalProcessingClasses:set_maps_info_table: keys don't match:"
                f"limit_table has keys {limit_table_keys}, "
                f"original keys {self.img_model_adv.map_keys}"
            )

        show_table_keys = [_[0] for _ in show_table]
        if set(show_table_keys) != set(self.img_model_adv.map_keys):
            raise ValueError(
                "GlobalProcessingClasses:set_maps_info_table: keys don't match:"
                f"show_table has keys {show_table_keys}, "
                f"original keys {self.img_model_adv.map_keys}"
            )

        # Copy limits
        for row in limit_table:
            key, v_low, v_high = row

            rng_low = self.img_model_adv.range_dict[key]["low"]
            rng_high = self.img_model_adv.range_dict[key]["high"]

            v_low_norm = (v_low - rng_low) / (rng_high - rng_low) * 100.0
            v_high_norm = (v_high - rng_low) / (rng_high - rng_low) * 100.0

            self.img_model_adv.limit_dict[key]["low"] = v_low_norm
            self.img_model_adv.limit_dict[key]["high"] = v_high_norm

        # Copy 'show' status (whether the map should be shown)
        for row in show_table:
            key, show_status = row
            self.img_model_adv.stat_dict[key] = bool(show_status)

    def get_maps_dataset_list(self):
        dsets = list(self.io_model.img_dict_keys)
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
        # return int(self.img_model_adv.data_opt)
        return int(self.io_model.img_dict_default_selected_item)

    def set_maps_selected_dataset(self, dataset_index):
        """Select dataset (XRF Maps tab)"""
        # self.img_model_adv.select_dataset(dataset_index)
        self.io_model.select_img_dict_item(dataset_index)

    def get_maps_scaler_index(self):
        return self.img_model_adv.scaler_name_index

    def set_maps_scaler_index(self, scaler_index):
        self.img_model_adv.set_scaler_index(scaler_index)

    def get_maps_quant_norm_enabled(self):
        return bool(self.img_model_adv.quantitative_normalization)

    def set_maps_quant_norm_enabled(self, enable):
        self.img_model_adv.enable_quantitative_normalization(enable)

    def get_maps_scale_opt(self):
        """Returns selected plot type: `Linear` or `Log`"""
        return self.img_model_adv.scale_opt

    def set_maps_scale_opt(self, scale_opt):
        """Set plot type. Allowed values: `Linear` and `Log`"""
        self.img_model_adv.set_scale_opt(scale_opt)

    def get_maps_color_opt(self):
        """Returns selected plot color option: `Linear` or `Log`"""
        return self.img_model_adv.color_opt

    def set_maps_color_opt(self, color_opt):
        """Set plot type. Allowed values: `Linear` and `Log`"""
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
            or `self.io_model.img_dict[<dataset>]` dictionaries.
        """
        # Check if 'range_dict' and 'limit_dict' have the same set of keys
        ks = self.img_model_rgb.map_keys
        ks_range = list(self.img_model_rgb.range_dict.keys())
        ks_limit = list(self.img_model_rgb.limit_dict.keys())
        rgb_dict = self.img_model_rgb.rgb_dict
        if set(ks) != set(ks_limit):
            raise RuntimeError(
                "The list of keys in 'limit_dict' is not as expected: "
                f"limit_dict.keys(): {ks_limit} expected: {ks}"
            )
        if set(ks) != set(ks_range):
            raise RuntimeError(
                "The list of keys in 'range_dict' is not as expected: "
                f"range_dict.keys(): {ks_range} expected: {ks}"
            )
        if len(rgb_dict) != len(self.img_model_rgb.rgb_keys) or set(rgb_dict.keys()) != set(
            self.img_model_rgb.rgb_keys
        ):
            raise ValueError(
                "GlobalProcessingClasses.get_rgb_maps_info_table(): "
                f"incorrect set of keys in 'rgb_dict': {list(rgb_dict.keys())}, "
                f"expected: {self.img_model_rgb.rgb_keys}"
            )
        for v in rgb_dict.values():
            if (v is not None) and (v not in ks):
                raise ValueError(
                    f"GlobalProcessingClasses.get_rgb_maps_info_table(): Invalid key {v}. Allowed keys: {ks}"
                )

        range_table = []
        limit_table = []
        for key in ks:
            rng_low = self.img_model_rgb.range_dict[key]["low"]
            rng_high = self.img_model_rgb.range_dict[key]["high"]
            limit_low_norm = self.img_model_rgb.limit_dict[key]["low"]
            limit_high_norm = self.img_model_rgb.limit_dict[key]["high"]
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
            or `self.io_model.img_dict[<dataset>]` dictionaries.
        """
        # Verify: the keys in both tables must match 'self.img_model_adv.map_keys'
        limit_table_keys = [_[0] for _ in limit_table]
        if set(limit_table_keys) != set(self.img_model_rgb.map_keys):
            raise ValueError(
                "GlobalProcessingClasses:set_maps_info_table: keys don't match:"
                f"limit_table has keys {limit_table_keys}, "
                f"original keys {self.img_model_rgb.map_keys}"
            )

        # Copy limits
        for row in limit_table:
            key, v_low, v_high = row

            rng_low = self.img_model_rgb.range_dict[key]["low"]
            rng_high = self.img_model_rgb.range_dict[key]["high"]

            v_low_norm = (v_low - rng_low) / (rng_high - rng_low) * 100.0
            v_high_norm = (v_high - rng_low) / (rng_high - rng_low) * 100.0

            self.img_model_rgb.limit_dict[key]["low"] = v_low_norm
            self.img_model_rgb.limit_dict[key]["high"] = v_high_norm

        self.img_model_rgb.rgb_dict = rgb_dict.copy()

    def get_rgb_maps_dataset_list(self):
        dsets = list(self.io_model.img_dict_keys)
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
        # return int(self.img_model_rgb.data_opt)
        return int(self.io_model.img_dict_default_selected_item)

    def set_rgb_maps_selected_dataset(self, dataset_index):
        """Select dataset (XRF Maps tab)"""
        # self.img_model_rgb.select_dataset(dataset_index)
        self.io_model.select_img_dict_item(dataset_index)

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
            raise ValueError(
                f"Range name {range_name} is not in the list of allowed "
                f"names {self.plot_model.energy_range_name}"
            )
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

    def update_plot_fit(self):
        self.plot_model.plot_experiment()

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
        self.plot_model.change_escape_peak_settings(
            1 if plot_escape_peak else 0, materials.index(detector_material)
        )

    # ==========================================================================
    #          The following methods are used by Model tab
    def get_autofind_elements_params(self):
        """Assemble the parameters managed by 'Find Elements in Sample` dialog box."""
        dialog_params = {}
        keys1 = ["e_offset", "e_linear", "e_quadratic", "fwhm_offset", "fwhm_fanoprime", "coherent_sct_energy"]
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

    def set_autofind_elements_params(self, dialog_params, *, update_model=True, update_fitting_params=True):
        """Save the parameters changed in 'Find Elements in Sample` dialog box.

        Parameters
        ----------
        dialog_params : dict
            parameters from dialog box
        update_model : boolean
            True - update the spectra based on new parameters. Updating the spectra
            may take long time if model contains a lot of parameters, therefore it
            should not be used for the default model which contains all supported emission
            lines.
        update_fitting_params : bool
            True - execute callback functions after updating fitting parameters

        Returns
        -------
        boolean
            True - selected range or incident energy were changed, False otherwise
        """
        logger.debug("Saving parameters from 'DialogFindElements'")

        # Check if the critical parameters were changed
        if (
            (self.plot_model.incident_energy != dialog_params["coherent_sct_energy"]["value"])
            or (self.param_model.energy_bound_high_buf != dialog_params["energy_bound_high"]["value"])
            or (self.param_model.energy_bound_low_buf != dialog_params["energy_bound_low"]["value"])
            or (self.param_model.param_new["e_offset"]["value"] != dialog_params["e_offset"]["value"])
            or (self.param_model.param_new["e_linear"]["value"] != dialog_params["e_linear"]["value"])
            or (self.param_model.param_new["e_quadratic"]["value"] != dialog_params["e_quadratic"]["value"])
        ):
            return_value = True
        else:
            return_value = False

        keys = ["e_offset", "e_linear", "e_quadratic", "fwhm_offset", "fwhm_fanoprime"]
        for k in keys:
            self.param_model.param_new[k]["value"] = dialog_params[k]["value"]

        self.io_model.incident_energy_set = dialog_params["coherent_sct_energy"]["value"]
        self.param_model.energy_bound_high_buf = dialog_params["energy_bound_high"]["value"]
        self.param_model.energy_bound_low_buf = dialog_params["energy_bound_low"]["value"]

        # Also change incident energy
        self.param_model.param_new["coherent_sct_energy"]["value"] = dialog_params["coherent_sct_energy"]["value"]
        self.param_model.param_new["non_fitting_values"]["energy_bound_low"]["value"] = dialog_params[
            "energy_bound_low"
        ]["value"]
        self.param_model.param_new["non_fitting_values"]["energy_bound_high"]["value"] = dialog_params[
            "energy_bound_high"
        ]["value"]

        if update_model and return_value:
            self.param_model.create_spectrum_from_param_dict(reset=False)

        if update_fitting_params:
            self.fitting_parameters_changed()

        return return_value

    def get_general_fitting_params(self):
        params = dict()

        # Another option is self.param_model.param_new
        source_dict = self.param_model.param_new

        params["max_iterations"] = self.fit_model.fit_num
        params["tolerance"] = self.fit_model.ftol
        params["escape_peak_ratio"] = source_dict["non_fitting_values"]["escape_ratio"]

        params["incident_energy"] = source_dict["coherent_sct_energy"]["value"]
        params["range_low"] = source_dict["non_fitting_values"]["energy_bound_low"]["value"]
        params["range_high"] = source_dict["non_fitting_values"]["energy_bound_high"]["value"]

        params["subtract_baseline_linear"] = self.fit_model.linear_bg
        params["subtract_baseline_snip"] = self.fit_model.use_snip
        params["add_bias"] = self.fit_model.raise_bg

        params["snip_window_size"] = source_dict["non_fitting_values"]["background_width"]

        return params

    def set_general_fitting_params(self, params):

        # Another option is self.param_model.param_new
        dest_dict = self.param_model.param_new

        self.fit_model.fit_num = params["max_iterations"]
        self.fit_model.ftol = params["tolerance"]
        dest_dict["non_fitting_values"]["escape_ratio"] = params["escape_peak_ratio"]

        dest_dict["coherent_sct_energy"]["value"] = params["incident_energy"]
        dest_dict["non_fitting_values"]["energy_bound_low"]["value"] = params["range_low"]
        dest_dict["non_fitting_values"]["energy_bound_high"]["value"] = params["range_high"]
        self.io_model.incident_energy_set = params["incident_energy"]

        self.fit_model.linear_bg = params["subtract_baseline_linear"]
        self.fit_model.use_snip = params["subtract_baseline_snip"]
        self.fit_model.raise_bg = params["add_bias"]

        dest_dict["non_fitting_values"]["background_width"] = params["snip_window_size"]

        self.param_model.create_spectrum_from_param_dict(reset=False)
        self.apply_to_fit()

        self.fitting_parameters_changed()

    def get_fit_strategy_list(self):
        return fit_strategy_list.copy()

    def get_detailed_fitting_params_lines(self):
        # Dictionary of fitting parameters. Create a copy !!!
        param_dict = copy.deepcopy(self.param_model.param_new)
        # Ordered list of emission lines
        eline_list = self.param_model.get_sorted_element_list()
        # Dictionary: emission line -> [list of keys in 'params' dictionary]
        eline_key_dict = dict()
        eline_energy_dict = dict()
        for eline in eline_list:
            key = None
            eline_category = self.get_eline_name_category(eline)
            if eline_category == "eline":
                key = eline[:-1] + eline[-1].lower()
            elif eline_category == "pileup":
                key = f"pileup_{eline.replace('-', '_')}"
            elif eline_category == "userpeak":
                key = eline

            if key:
                e_list = [_ for _ in param_dict.keys() if _.startswith(key)]
            else:
                e_list = []

            energy_list = []
            if eline_category == "eline":
                for key in e_list:
                    el = "_".join(key.split("_")[:2])
                    energy_list.append(get_eline_energy(el))
            elif eline_category == "pileup":
                energy = self.param_model.get_pileup_peak_energy(eline)
                energy_list = [energy] * len(e_list)
            elif eline_category == "userpeak":
                key = eline + "_delta_center"
                energy = param_dict[key]["value"] + 5.0
                energy_list = [energy] * len(e_list)

                # Convert energy for user peaks to absolute values
                for k in ("value", "min", "max"):
                    param_dict[key][k] += 5.0

            eline_key_dict[eline] = e_list
            eline_energy_dict[eline] = energy_list

        # List of other parameters in 'params' dictionary (except non_fitting_values).
        #   Currently all the parameters not related to emission lines are strictly lower-case.
        other_param_list = [_ for _ in param_dict.keys() if (_ == _.lower()) and (_ != "non_fitting_values")]
        other_param_list.sort()
        # eline_list = ["Shared parameters"] + eline_list

        _fit_strategy_list = fit_strategy_list.copy()
        _bound_options = bound_options.copy()

        return {
            "param_dict": param_dict,
            "eline_list": eline_list,
            "eline_key_dict": eline_key_dict,
            "eline_energy_dict": eline_energy_dict,
            "other_param_list": other_param_list,
            "fit_strategy_list": _fit_strategy_list,
            "bound_options": _bound_options,
        }

    def get_detailed_fitting_params_shared(self):
        # Dictionary of fitting parameters. Create a copy !!!
        param_dict = copy.deepcopy(self.param_model.param_new)

        for eline in self.param_model.get_sorted_element_list():
            eline_category = self.get_eline_name_category(eline)
            if eline_category == "userpeak":
                key = eline + "_delta_center"
                # Convert energy for user peaks to absolute values
                for k in ("value", "min", "max"):
                    param_dict[key][k] += 5.0

        # Ordered list of emission lines
        eline_list = ["Shared parameters"]
        # Dictionary: emission line -> [list of keys in 'params' dictionary]
        eline_key_dict = dict()
        eline_energy_dict = dict()

        # List of other parameters in 'params' dictionary (except non_fitting_values).
        #   Currently all the parameters not related to emission lines are strictly lower-case.
        other_param_list = [_ for _ in param_dict.keys() if (_ == _.lower()) and (_ != "non_fitting_values")]
        other_param_list.sort()

        _fit_strategy_list = fit_strategy_list.copy()
        _bound_options = bound_options.copy()

        return {
            "param_dict": param_dict,
            "eline_list": eline_list,
            "eline_key_dict": eline_key_dict,
            "eline_energy_dict": eline_energy_dict,
            "other_param_list": other_param_list,
            "fit_strategy_list": _fit_strategy_list,
            "bound_options": _bound_options,
        }

    def set_detailed_fitting_params(self, dialog_data):
        param_dict = copy.deepcopy(dialog_data["param_dict"])
        # 'param_dict' is expected to have identical structure as 'param_model.param_new'.
        # We don't want to change the reference to the parameters, so we copy references to
        #   dictionary elements (which are also dictionaries).

        eline_list = self.param_model.get_sorted_element_list()
        for eline in eline_list:
            eline_category = self.get_eline_name_category(eline)
            if eline_category == "userpeak":
                key = eline + "_delta_center"
                # Convert energy for user peaks back to relative values
                for k in ("value", "min", "max"):
                    param_dict[key][k] -= 5.0

        # Check if the energy axis parameters were changed
        e_axis_changed = False
        e_axis_keys = ("e_offset", "e_linear", "e_quadratic")
        for key in e_axis_keys:
            if self.param_model.param_new[key]["value"] != param_dict[key]["value"]:
                e_axis_changed = True
                break

        for key, val in param_dict.items():
            self.param_model.param_new[key] = val

        self.io_model.incident_energy_set = param_dict["coherent_sct_energy"]["value"]

        self.param_model.create_spectrum_from_param_dict(reset=False)
        self.apply_to_fit()

        # This is needed in case the coefficients that define energy axis have changed
        if e_axis_changed:
            self.update_preview_spectrum_plot()

        self.fitting_parameters_changed()

    def get_quant_standard_list(self):
        self.fit_model.param_quant_estimation.clear_standards()
        self.fit_model.param_quant_estimation.load_standards()

        qe_param_built_in = self.fit_model.param_quant_estimation.standards_built_in
        qe_param_custom = self.fit_model.param_quant_estimation.standards_custom

        # The current selection is set to the currently selected standard
        #   held in 'fit_model.qe_standard_selected_copy'
        qe_standard_selected = self.fit_model.qe_standard_selected_copy
        qe_standard_selected = self.fit_model.param_quant_estimation.set_selected_standard(qe_standard_selected)

        return qe_param_built_in, qe_param_custom, qe_standard_selected

    def set_selected_quant_standard(self, selected_standard):
        if selected_standard is not None:
            self.fit_model.param_quant_estimation.set_selected_standard(selected_standard)
            self.fit_model.qe_standard_selected_copy = copy.deepcopy(selected_standard)
        else:
            self.fit_model.param_quant_estimation.clear_standards()
            self.fit_model.qe_standard_selected_copy = None

        self.fitting_parameters_changed()

    def is_quant_standard_custom(self, standard=None):
        return self.fit_model.param_quant_estimation.is_standard_custom(standard)

    def process_peaks_from_quantitative_sample_data(self):
        incident_energy = self.param_model.param_new["coherent_sct_energy"]["value"]
        # Generate the data structure for the results of processing of standard data
        self.fit_model.param_quant_estimation.gen_fluorescence_data_dict(incident_energy)
        # Obtain the list (dictionary) of elemental lines from the generated structure
        elemental_lines = self.fit_model.param_quant_estimation.fluorescence_data_dict["element_lines"]

        self.param_model.find_peak(elemental_lines=elemental_lines.keys())
        self.param_model.EC.order()
        self.param_model.update_name_list()

        self.param_model.EC.turn_on_all()
        self.param_model.data_for_plot()

        # update parameter for fit
        self.param_model.create_full_param()  # Not sure it is necessary
        self.fit_model.apply_default_param()

        # update experimental plots in case the coefficients change
        self.plot_model.plot_experiment()

        self.plot_model.plot_fit(
            self.param_model.prefit_x, self.param_model.total_y, self.param_model.auto_fit_all
        )

        # Update displayed intensity of the selected peak
        self.plot_model.compute_manual_peak_intensity()

        # Show the summed spectrum used for fitting
        self.plot_model.plot_exp_opt = False
        self.plot_model.plot_exp_opt = True
        # For plotting purposes, otherwise plot will not update
        self.plot_model.show_fit_opt = False
        self.plot_model.show_fit_opt = True

        self.fitting_parameters_changed()

    def load_parameters_from_file(self, parameter_file_path, incident_energy_from_param_file=None):

        try:
            self.param_model.read_param_from_file(parameter_file_path)
        except Exception as ex:
            msg = f"Error occurred while reading parameter file: {ex}"
            logger.error(msg)
            raise IOError(msg)
        else:
            # Make a decision, if the incident energy from metadata should be replaced
            #   with the incident energy from the parameter file
            overwrite_metadata_incident_energy = False

            # Incident energy from the parameter file
            param_incident_energy = self.param_model.param_new["coherent_sct_energy"]["value"]

            if self.io_model.incident_energy_available:

                # Incident energy from datafile metadata
                mdata_incident_energy = self.io_model.scan_metadata.get_mono_incident_energy()

                # If two energies have very close values (say 1 eV), then the difference doesn't matter
                #   Consider two energies equal (they probably ARE equal) and use the value
                #   from datafile metadata.
                if not math.isclose(param_incident_energy, mdata_incident_energy, abs_tol=0.001):
                    # TODO: the following text is not properly formatted for QMessageBox
                    #       It's not very important now, but should be resolved in the future.
                    msg = (
                        f"The values of incident energy from data file metadata "
                        f"and parameter file are different.\n"
                        f"Incident energy from metadata: {mdata_incident_energy} keV.\n"
                        f"Incident energy from the loaded parameter file: {param_incident_energy} keV.\n"
                        f"Would you prefer to use the incident energy from the parameter file for processing?"
                    )

                    if incident_energy_from_param_file is None:
                        return False, msg
                    else:
                        overwrite_metadata_incident_energy = bool(incident_energy_from_param_file)

            else:

                # If incident energy is not present in file metadata, then
                #   just load the incident energy from the parameter file.
                overwrite_metadata_incident_energy = True

            if overwrite_metadata_incident_energy:
                logger.info(f"Using incident energy from the parameter file: {param_incident_energy} keV")
            else:
                # Keep the incident energy from the file
                logger.info(f"Using incident energy from the datafile metadata: {mdata_incident_energy} keV")
                incident_energy = round(mdata_incident_energy, 6)
                self.param_model.param_new["coherent_sct_energy"]["value"] = incident_energy
                self.param_model.param_new["non_fitting_values"]["energy_bound_high"]["value"] = (
                    incident_energy + 0.8
                )

            self.fit_model.apply_default_param()

            # update experimental plots
            self.plot_model.plot_experiment()
            self.plot_model.plot_exp_opt = False
            self.plot_model.plot_exp_opt = True

            # calculate profile and plot
            self.fit_model.get_profile()

            # update experimental plot with new calibration values
            self.plot_model.plot_fit(
                self.fit_model.cal_x, self.fit_model.cal_y, self.fit_model.cal_spectrum, self.fit_model.residual
            )

            self.plot_model.plot_exp_opt = False
            self.plot_model.plot_exp_opt = True
            # For plotting purposes, otherwise plot will not update
            self.plot_model.show_fit_opt = False
            self.plot_model.show_fit_opt = True

            # The following statement is necessary mostly to set the correct value of
            #   the upper boundary of the energy range used for emission line search.
            self.plot_model.change_incident_energy(self.param_model.param_new["coherent_sct_energy"]["value"])

            # update parameter for fit
            # self.param_model.create_full_param()
            self.fit_model.apply_default_param()

            # Update displayed intensity of the selected peak
            self.plot_model.compute_manual_peak_intensity()

            self.fitting_parameters_changed()

            return True, ""

    def find_elements_automatically(self):
        logger.debug("Starting automated element search")
        self.param_model.find_peak()

        self.param_model.EC.order()
        self.param_model.update_name_list()

        self.param_model.EC.turn_on_all()

        self.plot_model.plot_exp_opt = True
        self.apply_to_fit()

        self.fitting_parameters_changed()

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
            cs = get_cs(eline, self.param_model.param_new["coherent_sct_energy"]["value"])
            row_data = {
                "eline": eline,
                "sel_status": sel_status,
                "z": z,
                "energy": energy,
                "peak_int": peak_int,
                "rel_int": rel_int,
                "cs": cs,
            }
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
            self.param_model.EC.element_dict[eline].stat_copy = self.param_model.EC.element_dict[eline].status

        self.param_model.data_for_plot()

        self.plot_model.plot_fit(
            self.param_model.prefit_x, self.param_model.total_y, self.param_model.auto_fit_all
        )
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
            eline_info = (
                self.convert_full_eline_name(e1, to_upper=True),
                self.convert_full_eline_name(e2, to_upper=True),
                energy,
            )
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
        e_low = self.param_model.param_new["non_fitting_values"]["energy_bound_low"]["value"]
        e_high = self.param_model.param_new["non_fitting_values"]["energy_bound_high"]["value"]
        return e_low, e_high

    def add_peak_manual(self, eline):
        """Manually add a peak (emission line) using 'Add' button"""
        self.select_eline(eline)

        # Verify if we are adding a userpeak
        is_userpeak = self.get_eline_name_category(eline) == "userpeak"

        # The following set of conditions is not complete, but sufficient
        if self.param_model.x0 is None or self.param_model.y0 is None:
            err_msg = "Experimental data is not loaded or initial\nspectrum fitting is not performed"
            raise RuntimeError(err_msg)

        elif is_userpeak and (not self.plot_model.vertical_marker_is_visible):
            err_msg = (
                "Select position of userpeak by clicking on the spectrum plot.\n"
                "Note: plot toolbar options, such as Pan and Zoom, \n"
                "must be turned off before selecting the position."
            )
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

        self.fitting_parameters_changed()

    def remove_peak_manual(self, eline):
        """Manually add a peak (emission line) using 'Remove' button"""
        try:
            self.select_eline(eline)
            self.param_model.remove_peak_manual()
            self.apply_to_fit()
            self.fitting_parameters_changed()
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
        self.fitting_parameters_changed()

    def update_eline_peak_height(self, eline, maxv):
        self.select_eline(eline)
        self.param_model.modify_peak_height(maxv)
        self.apply_to_fit()
        self.fitting_parameters_changed()

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
        self.fitting_parameters_changed()

    def remove_unchecked_peaks(self):
        self.param_model.remove_elements_unselected()
        self.param_model.update_name_list()
        self.apply_to_fit()
        self.fitting_parameters_changed()

    def apply_to_fit(self, *, update_plots=True):
        """
        Update plot, and apply updated parameters to fitting process.
        Note: this is an original function from 'fit.enaml' file.
        """
        self.param_model.EC.update_peak_ratio()
        self.param_model.data_for_plot()

        # update parameter for fit
        self.param_model.create_full_param()  # Not sure this is needed
        self.fit_model.apply_default_param()

        # Update displayed intensity of the selected peak
        self.plot_model.compute_manual_peak_intensity()

        if update_plots:
            # update experimental plots in case the coefficients change
            self.plot_model.plot_experiment()

            self.plot_model.plot_fit(
                self.param_model.prefit_x, self.param_model.total_y, self.param_model.auto_fit_all
            )
            self.plot_model.update_fit_plots()

    def total_spectrum_fitting(self):

        # Update parameter for fit. This may be unnecessary
        self.apply_to_fit(update_plots=False)

        self.fit_model.fit_multiple()

        # BUG: line color for pileup is not correct from fit
        self.fit_model.get_profile()

        # update experimental plot with new calibration values
        self.plot_model.plot_experiment()

        self.plot_model.plot_fit(
            self.fit_model.cal_x, self.fit_model.cal_y, self.fit_model.cal_spectrum, self.fit_model.residual
        )

        # For plotting purposes, otherwise plot will not update
        self.plot_model.plot_exp_opt = False
        self.plot_model.plot_exp_opt = True
        self.plot_model.show_fit_opt = False
        self.plot_model.show_fit_opt = True

        # update autofit param
        self.param_model.update_new_param(self.param_model.param_new)
        # param_model.get_new_param_from_file(parameter_file_path)
        self.param_model.update_name_list()
        self.param_model.EC.turn_on_all()

        # Update displayed intensity of the selected peak
        self.plot_model.compute_manual_peak_intensity()

        self.fitting_parameters_changed()

    def save_param_to_file(self, path):
        save_as(path, self.param_model.param_new)

    def save_spectrum(self, directory, save_fit):
        self.fit_model.output_summed_data_fit(save_fit=save_fit, directory=directory)

    def compute_current_rfactor(self, save_fit):
        return self.fit_model.compute_current_rfactor(save_fit)

    def get_iter_and_var_number(self):
        return {"var_number": self.fit_model.nvar, "iter_number": self.fit_model.function_num}

    # ==========================================================================
    #          The following methods are used by Maps tab
    def fit_individual_pixels(self):
        self.apply_to_fit()

        self.fit_model.fit_single_pixel()

        # add scalers to fit dict
        scaler_keys = [v for v in self.io_model.img_dict.keys() if "scaler" in v]
        if len(scaler_keys) > 0:
            self.fit_model.fit_img[list(self.fit_model.fit_img.keys())[0]].update(
                self.io_model.img_dict[scaler_keys[0]]
            )

        self.io_model.update_img_dict(self.fit_model.fit_img)

    # ==========================================================================
    #          The following methods are used by ROI window
    def get_roi_selected_element_list(self):
        element_list = self.roi_model.element_for_roi
        # TODO: should probably be sorted differently
        return element_list

    def set_roi_selected_element_list(self, elements_for_roi):
        self.roi_model.element_for_roi = elements_for_roi
        self.plot_model.plot_roi_bound()

    def clear_roi_element_list(self):
        for r in self.roi_model.element_list_roi:
            self.plot_model.roi_dict[r].show_plot = False
        self.plot_model.plot_roi_bound()
        self.roi_model.clear_selected_elements()

    def load_roi_element_list_from_selected(self):
        for r in self.roi_model.element_list_roi:
            self.plot_model.roi_dict[r].show_plot = False
        self.plot_model.plot_roi_bound()
        selected_element_list = self.param_model.EC.get_element_list()
        selected_element_list = [
            _ for _ in selected_element_list if self.param_model.get_eline_name_category(_) == "eline"
        ]
        self.roi_model.select_elements_from_list(selected_element_list)

    def get_roi_subtract_background(self):
        return self.roi_model.subtract_background

    def set_roi_subtract_background(self, subtract_background):
        self.roi_model.subtract_background = bool(subtract_background)

    def compute_rois(self):
        if len(self.roi_model.element_for_roi) == 0:
            raise RuntimeError("No elements are selected for ROI computation. Select at least one element.")
        roi_result = self.roi_model.get_roi_sum()

        self.io_model.update_img_dict(roi_result)

    def show_roi(self, eline, show_status):
        self.plot_model.roi_dict = self.roi_model.roi_dict
        if show_status:
            # self.plot_model.plot_exp_opt = True
            self.plot_model.roi_dict[eline].show_plot = True
            self.plot_model.plot_roi_bound()
        else:
            self.plot_model.roi_dict[eline].show_plot = False
            self.plot_model.plot_roi_bound()

    def change_roi(self, eline, low, high):
        # Convert keV to eV (current implementation of ROIModel is using eV.
        self.roi_model.roi_dict[eline].left_val = int(low * 1000)
        self.roi_model.roi_dict[eline].right_val = int(high * 1000)
        self.plot_model.plot_roi_bound()

    def get_roi_settings(self):
        roi_settings = []

        eline_list = list(self.roi_model.element_list_roi)

        for eline in eline_list:
            # We display values in keV, but current implementation of ROIModel
            #   is using eV.
            energy_center = self.roi_model.roi_dict[eline].line_val / 1000.0
            energy_left = self.roi_model.roi_dict[eline].left_val / 1000.0
            energy_right = self.roi_model.roi_dict[eline].right_val / 1000.0
            range_displayed = self.roi_model.roi_dict[eline].show_plot

            energy_left_default = self.roi_model.roi_dict[eline].default_left / 1000.0
            energy_right_default = self.roi_model.roi_dict[eline].default_right / 1000.0

            roi_settings.append(
                {
                    "eline": eline,
                    "energy_center": energy_center,
                    "energy_left": energy_left,
                    "energy_right": energy_right,
                    "range_displayed": range_displayed,
                    "energy_left_default": energy_left_default,
                    "energy_right_default": energy_right_default,
                }
            )

        roi_settings.sort(key=lambda _: _["energy_center"])
        return roi_settings

    # ==========================================================================
    #          The following methods are used by ROI window
    def is_quant_standard_selected(self):
        """
        Returns True if quantitative standard was selected from the list.
        The result can be used to determine if saving quantitative standard is a valid
        operation.
        """
        return bool(self.fit_model.param_quant_estimation.fluorescence_data_dict)

    def is_quant_standard_fitting_available(self):
        """
        Returns True if fitted map is available.
        """
        result_map, selected_det_channel, scaler_name = self.fit_model.get_selected_fitted_map_data()
        return bool(result_map) and bool(selected_det_channel)

    def get_suggested_quant_file_name(self):
        """
        Return suggested 'standard' file name used for saving quantitative calibration
        to a JSON file
        """
        return self.fit_model.param_quant_estimation.get_suggested_json_fln()

    def get_displayed_quant_calib_parameters(self):

        fluorescence_data_dict = self.fit_model.param_quant_estimation.fluorescence_data_dict
        if fluorescence_data_dict is None:
            raise RuntimeError("Attempt to obtain quantitative standard parameters while no standard is selected")

        result_map, selected_det_channel, scaler_name = self.fit_model.get_selected_fitted_map_data()

        if not result_map:
            raise RuntimeError("No XRF maps were found. Run Individual Pixel Fitting to generate XRF maps.")

        if not selected_det_channel:
            raise RuntimeError("Selected XRF map does not contain fitted XRF maps.")

        suggested_fln = self.fit_model.param_quant_estimation.get_suggested_json_fln()

        # Fill the available fluorescence data
        self.fit_model.param_quant_estimation.fill_fluorescence_data_dict(
            xrf_map_dict=result_map, scaler_name=scaler_name
        )
        self.fit_model.param_quant_estimation.set_detector_channel_in_data_dict(
            detector_channel=selected_det_channel
        )

        distance_to_sample = self.fit_model.qe_standard_distance_to_sample
        self.fit_model.param_quant_estimation.set_distance_to_sample_in_data_dict(
            distance_to_sample=distance_to_sample
        )

        # Set optional parameters
        if self.io_model.scan_metadata_available:
            _scan_id = None
            _scan_uid = None
            if "scan_id" in self.io_model.scan_metadata:
                _scan_id = self.io_model.scan_metadata["scan_id"]
            if "scan_uid" in self.io_model.scan_metadata:
                _scan_uid = self.io_model.scan_metadata["scan_uid"]
            self.fit_model.param_quant_estimation.set_optional_parameters(scan_id=_scan_id, scan_uid=_scan_uid)

        preview = self.fit_model.param_quant_estimation.get_fluorescence_data_dict_text_preview()

        return {"suggested_file_name": suggested_fln, "distance_to_sample": distance_to_sample, "preview": preview}

    def save_distance_to_sample(self, distance_to_sample):
        self.fit_model.qe_standard_distance_to_sample = distance_to_sample
        self.fit_model.param_quant_estimation.set_distance_to_sample_in_data_dict(
            distance_to_sample=distance_to_sample
        )

    def save_quantitative_standard(self, file_path, overwrite_existing):
        """Will raise exception if unsuccessful"""
        self.fit_model.param_quant_estimation.save_fluorescence_data_dict(
            file_path, overwrite_existing=overwrite_existing
        )

    # ==========================================================================
    #     The following methods are used for exporting maps as TIFF and TXT
    def get_parameters_for_exporting_maps(self):
        dataset_list, dset_sel = self.get_maps_dataset_list()
        scalers, scaler_sel = self.get_maps_scaler_list()
        interpolate_on = self.get_maps_grid_interpolate()
        quant_norm_on = self.get_maps_quant_norm_enabled()

        return {
            "dset_list": dataset_list,
            "dset_sel": dset_sel,  # We want it to start from 0
            "scaler_list": scalers,
            "scaler_sel": scaler_sel,
            "interpolate_on": interpolate_on,
            "quant_norm_on": quant_norm_on,
        }

    def export_xrf_maps(
        self, *, results_path, dataset_name, scaler_name, interpolate_on, quant_norm_on, file_formats
    ):
        # We don't want to make any changes to 'param_quant_analysis' at the original location
        #   This is a temporary copy.
        param_quant_analysis = copy.deepcopy(self.img_model_adv.param_quant_analysis)
        param_quant_analysis.experiment_detector_channel = self.img_model_adv.get_detector_channel_name(
            dataset_name
        )
        for file_format in file_formats:
            self.fit_model.output_2Dimage(
                results_path=results_path,
                dataset_name=dataset_name,
                scaler_name=scaler_name,
                interpolate_on=interpolate_on,
                quant_norm_on=quant_norm_on,
                param_quant_analysis=param_quant_analysis,
                file_format=file_format,
            )

    # ==========================================================================
    #     Selection of the region pixels for saved spectra in Maps tab
    def get_enable_save_spectra(self):
        return self.fit_model.save_point

    def set_enable_save_spectra(self, enabled):
        self.fit_model.save_point = enabled

    def get_dataset_map_size(self):
        return self.io_model.get_dataset_map_size()

    def get_selection_area_save_spectra(self):
        return {
            "row_min": self.fit_model.point1v + 1,
            "row_max": self.fit_model.point2v,
            "col_min": self.fit_model.point1h + 1,
            "col_max": self.fit_model.point2h,
        }

    def set_selection_area_save_spectra(self, area):
        map_size = self.get_dataset_map_size()
        map_size = (1, 1) if map_size is None else map_size
        n_rows, n_cols = map_size

        def _fit_to_range(val, val_min, val_max):
            if val < val_min:
                val = val_min
            if val > val_max:
                val = val_max
            return val

        # Modify the selection so that at least one row or column is selected
        #   (area is at least 1 pixel wide in every direction)
        row_min = _fit_to_range(area["row_min"] - 1, 0, n_rows - 1)
        row_max = _fit_to_range(area["row_max"], row_min + 1, n_rows)
        col_min = _fit_to_range(area["col_min"] - 1, 0, n_cols - 1)
        col_max = _fit_to_range(area["col_max"], col_min + 1, n_cols)
        self.fit_model.point1v = row_min
        self.fit_model.point2v = row_max
        self.fit_model.point1h = col_min
        self.fit_model.point2h = col_max


def autofind_emission_lines(
    data_file_name,
    param_file_name,
    *,
    working_directory=".",
    threshold=1.0,  # Cut-off threshold (percentage of the area of the strongest emission line)
    incident_energy=None,
    energy_bound_low=None,
    energy_bound_high=None,
    e_offset=None,
    e_linear=None,
    e_quadratic=None,
    fwhm_offset=None,
    fwhm_fanoprime=None,
):
    """
    Automatically find emission lines in total spectrum and create a parameter file.

    Parameters
    ----------
    data_file_name: str
        Full or relative path to the HDF5 file with data
    param_file_name: str
        Full or relative path to the created parameter file. The function will overwrite the existing file.
    working_directory: str
        The path is combined with ``data_file_name`` or `` param_file_name`` if those are relative paths,
        otherwise it is ignored.
    threshold: float
        Cut-off threshold (in percent, 0..100%). Emission lines with area smaller than the give percentage
        of the area of the strongest emission line are removed.
    incident_energy: float or None
        Incident (beam) energy in kEv. If ``None``, then incident energy from the experiment metadata
        (from HDF5 file) is used.
    energy_bound_low: float or None
        Lower bound (in kEv) of the range used for fitting. Default value is used if ``None``.
    energy_bound_low: float or None
        Upper bound (in kEv) of the range used for fitting. If ``None``, then the value is computed by
        adding 0.8 kEv to the incident energy.
    e_offset, e_linear, e_quadratic: float or None
        The parameters used for interpolation of energy axis. Default values (0, 1 and 0 respectively)
        are used if ``None``.
    fwhm_offset, fwhm_fanoprime: float or None
        The parameters of emission line model. Default values are used if ``None``.

    Returns
    -------
    None
    """

    def expand_file_name(fln, wd):
        fln = os.path.expanduser(fln)
        if not os.path.isabs(fln):
            wd = os.path.expanduser(wd)
            wd = os.path.abspath(wd)
            fln = os.path.join(wd, fln)
        return fln

    # Prepare file names. Working directory is ignored for absolute names.
    data_file_name = expand_file_name(data_file_name, working_directory)
    param_file_name = expand_file_name(param_file_name, working_directory)

    if (incident_energy is not None) and (energy_bound_high is None):
        energy_bound_high = incident_energy + 0.8

    default_parameters = param_data

    io_model = FileIOModel(working_directory=working_directory)
    param_model = ParamModel(default_parameters=default_parameters, io_model=io_model)

    print(f"Reading data from file {data_file_name!r} ...")
    io_model.file_path = data_file_name

    # Parameters from scan metadata
    runid, runuid = None, None
    if io_model.scan_metadata_available:
        if "scan_id" in io_model.scan_metadata:
            runid = int(io_model.scan_metadata["scan_id"])
        if "scan_uid" in io_model.scan_metadata:
            runuid = io_model.scan_metadata["scan_uid"]
    print(f"Processing scan: {runid if runid is not None else '-'} ({runuid if runuid is not None else '-'})")

    # Use incident energy from the run metadata (if available)
    if (incident_energy is None) and io_model.scan_metadata_available:
        if io_model.scan_metadata.is_mono_incident_energy_available():
            incident_energy = io_model.scan_metadata.get_mono_incident_energy()
            if energy_bound_high is None:
                energy_bound_high = incident_energy + 0.8

    if incident_energy is not None:
        param_model.param_new["coherent_sct_energy"]["value"] = incident_energy
    if energy_bound_low is not None:
        param_model.param_new["non_fitting_values"]["energy_bound_low"]["value"] = energy_bound_low
    if energy_bound_high is not None:
        param_model.param_new["non_fitting_values"]["energy_bound_high"]["value"] = energy_bound_high

    if e_offset is not None:
        param_model.param_new["e_offset"]["value"] = e_offset
    if e_linear is not None:
        param_model.param_new["e_linear"]["value"] = e_linear
    if e_quadratic is not None:
        param_model.param_new["e_quadratic"]["value"] = e_quadratic
    if fwhm_offset is not None:
        param_model.param_new["fwhm_offset"]["value"] = fwhm_offset
    if fwhm_fanoprime is not None:
        param_model.param_new["fwhm_fanoprime"]["value"] = fwhm_fanoprime

    print("Processing parameters:")
    print(f"  Incident energy: {param_model.param_new['coherent_sct_energy']['value']}")
    print(f"  Energy bound (low):  {param_model.param_new['non_fitting_values']['energy_bound_low']['value']}")
    print(f"  Energy bound (high): {param_model.param_new['non_fitting_values']['energy_bound_high']['value']}")
    print(f"  Energy offset:    {param_model.param_new['e_offset']['value']}")
    print(f"  Energy linear:    {param_model.param_new['e_linear']['value']}")
    print(f"  Energy quadratic: {param_model.param_new['e_quadratic']['value']}")
    print(f"  FWHM offset:    {param_model.param_new['fwhm_offset']['value']}")
    print(f"  FWHM fanoprime: {param_model.param_new['fwhm_fanoprime']['value']}\n")

    print("Searching for emission lines ...")
    param_model.find_peak()
    param_model.EC.order()
    param_model.update_name_list()
    param_model.EC.turn_on_all()
    param_model.create_full_param()

    param_model.remove_elements_below_threshold(threshv=threshold)
    param_model.update_name_list()
    param_model.create_full_param()

    print(f"Saving the parameter file {param_file_name!r} ...")
    save_as(param_file_name, param_model.param_new)

    print("Automatically found emission lines were successfully saved.")
