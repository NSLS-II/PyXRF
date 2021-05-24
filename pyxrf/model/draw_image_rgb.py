from __future__ import absolute_import, division, print_function

import numpy as np
import math
from functools import partial
from matplotlib.figure import Figure, Axes
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1.axes_rgb import make_rgb_axes
from atom.api import Atom, Str, Typed, Int, List, Dict, Bool

from ..core.utils import normalize_data_by_scaler, grid_interpolate
from ..core.xrf_utils import check_if_eline_supported

from .draw_image import DrawImageAdvanced

import logging

logger = logging.getLogger(__name__)

np.seterr(divide="ignore", invalid="ignore")  # turn off warning on invalid division


class DrawImageRGB(Atom):
    """
    This class draws RGB image.

    Attributes
    ----------
    fig : object
        matplotlib Figure
    ax : Axes
        The `Axes` object of matplotlib
    ax_r : Axes
        The `Axes` object to add the artist too
    ax_g : Axes
        The `Axes` object to add the artist too
    ax_b : Axes
        The `Axes` object to add the artist too
    file_name : str
    stat_dict : dict
        determine which image to show
    img_dict : dict
        multiple data sets to plot, such as fit data, or roi data
    img_dict_keys : list
    data_opt : int
        index to show which data is chosen to plot
    dict_to_plot : dict
        selected data dict to plot, i.e., fitting data or roi is selected
    map_keys : list
        keys of dict_to_plot
    color_opt : str
        orange or gray plot
    scaler_norm_dict : dict
        scaler normalization data, from img_dict
    scaler_items : list
        keys of scaler_norm_dict
    scaler_name_index : int
        index to select on GUI level
    scaler_data : None or numpy
        selected scaler data
    pixel_or_pos : int
        index to choose plot with pixel (== 0) or with positions (== 1)
    grid_interpolate: bool
        choose to interpolate 2D image in terms of x,y or not
    plot_all : Bool
        to control plot all of the data or not
    """

    # Reference to FileIOMOdel
    io_model = Typed(object)

    fig = Typed(Figure)
    ax = Typed(Axes)
    ax_r = Typed(Axes)
    ax_g = Typed(Axes)
    ax_b = Typed(Axes)
    data_opt = Int(0)
    img_title = Str()
    # plot_opt = Int(0)
    # plot_item = Str()
    dict_to_plot = Dict()
    map_keys = List()
    scaler_norm_dict = Dict()
    scaler_items = List()
    scaler_name_index = Int()
    scaler_data = Typed(object)
    pixel_or_pos = Int(0)
    grid_interpolate = Bool(False)
    plot_all = Bool(False)

    limit_dict = Dict()
    range_dict = Dict()

    # 'stat_dict' is legacy from 'DrawImageAdvanced' class. It is not used here,
    #   but it may be repurposed in the future if multicolor map presentation is developed
    stat_dict = Dict()
    # Contains dictionary {"red": <key>, "green": <key>, "blue": <key>}, key is the key
    #   from the dictionary 'self.dict_to_plot' or None.
    rgb_keys = List(str)  # The list of keys in 'rgb_dict'
    rgb_dict = Dict()

    # Reference used to access some fields
    img_model_adv = Typed(DrawImageAdvanced)
    # Variable that indicates whether quanitative normalization should be applied to data
    #   Associated with 'Quantitative' checkbox
    quantitative_normalization = Bool(False)

    rgb_name_list = List()  # List of names for RGB channels printed on the plot

    rgb_limit = Dict()
    name_not_scalable = List()

    def __init__(self, *, io_model, img_model_adv):
        self.io_model = io_model
        self.img_model_adv = img_model_adv

        self.fig = plt.figure(figsize=(3, 2))

        self.rgb_name_list = ["R", "G", "B"]

        # Do not apply scaler norm on following data
        self.name_not_scalable = [
            "r2_adjust",
            "r_factor",
            "alive",
            "dead",
            "elapsed_time",
            "scaler_alive",
            "i0_time",
            "time",
            "time_diff",
            "dwell_time",
        ]

        self.rgb_keys = ["red", "green", "blue"]
        self._init_rgb_dict()

    def img_dict_updated(self, change):
        """
        Observer function to be connected to the fileio model
        in the top-level gui.py startup

        Parameters
        ----------
        changed : bool
            True - 'io_model.img_dict` was updated, False - ignore
        """
        if change["value"]:
            self.select_dataset(self.io_model.img_dict_default_selected_item)
            self.init_plot_status()

    def init_plot_status(self):
        # init of pos values
        self.set_pixel_or_pos(0)

        # init of scaler for normalization
        self.scaler_name_index = 0

        scaler_groups = [v for v in list(self.io_model.img_dict.keys()) if "scaler" in v]
        if len(scaler_groups) > 0:
            # self.scaler_group_name = scaler_groups[0]
            self.scaler_norm_dict = self.io_model.img_dict[scaler_groups[0]]
            # for GUI purpose only
            self.scaler_items = []
            self.scaler_items = list(self.scaler_norm_dict.keys())
            self.scaler_items.sort()
            self.scaler_data = None

        logger.debug(
            "The following groups are included for RGB image display: {}".format(self.io_model.img_dict_keys)
        )

        self.show_image()

    def select_dataset(self, dataset_index):
        """
        Select dataset. Meaning of the index: 0 - no dataset is selected,
        1, 2, ... datasets with index 0, 1, ... is selected

        Parameters
        ----------
        dataset_index: int
            index of the selected dataset
        """
        self.data_opt = dataset_index

        try:
            if self.data_opt == 0:
                self.dict_to_plot = {}
                self.map_keys.clear()
                self.init_limits_and_stat()
                self.img_title = ""

            elif self.data_opt > 0:
                plot_item = self._get_current_plot_item()
                self.img_title = str(plot_item)
                self.dict_to_plot = self.io_model.img_dict[plot_item]
                # for GUI purpose only
                self.set_map_keys()
                self.init_limits_and_stat()

                # Select the first 3 entries for RGB display
                for n in range(min(len(self.rgb_keys), len(self.map_keys))):
                    self.rgb_dict[self.rgb_keys[n]] = self.map_keys[n]

        except IndexError:
            pass

        # Redraw image
        self.show_image()

    def set_map_keys(self):
        """
        Create sorted list of map keys. The list starts with sorted sequence of emission lines,
        followed by the sorted list of scalers and other maps.
        """
        self.map_keys.clear()
        # The key to use with 'img_dict', the name of the current dataset.
        plot_item = self._get_current_plot_item()
        keys_unsorted = list(self.io_model.img_dict[plot_item].keys())
        if len(keys_unsorted) != len(set(keys_unsorted)):
            logger.warning(
                f"DrawImageAdvanced:set_map_keys(): repeated keys in the dictionary 'img_dict': {keys_unsorted}"
            )
        keys_elines, keys_scalers = [], []
        for key in keys_unsorted:
            if check_if_eline_supported(key):  # Check if 'key' is an emission line (such as "Ca_K")
                keys_elines.append(key)
            else:
                keys_scalers.append(key)
        keys_elines.sort()
        keys_scalers.sort()
        self.map_keys = keys_elines + keys_scalers

    def get_selected_scaler_name(self):
        if self.scaler_name_index == 0:
            return None
        else:
            return self.scaler_items[self.scaler_name_index - 1]

    def set_scaler_index(self, scaler_index):

        self.scaler_name_index = scaler_index

        if self.scaler_name_index == 0:
            self.scaler_data = None
        else:
            try:
                scaler_name = self.scaler_items[self.scaler_name_index - 1]
            except IndexError:
                scaler_name = None
            if scaler_name:
                self.scaler_data = self.scaler_norm_dict[scaler_name]
                logger.info(
                    "Use scaler data to normalize, "
                    "and the shape of scaler data is {}, "
                    "with (low, high) as ({}, {})".format(
                        self.scaler_data.shape, np.min(self.scaler_data), np.max(self.scaler_data)
                    )
                )
        self.set_low_high_value()  # reset low high values based on normalization
        self.show_image()

    def _get_current_plot_item(self):
        """Get the key for the current plot item (use in dictionary 'img_dict')"""
        return self.io_model.img_dict_keys[self.data_opt - 1]

    def set_pixel_or_pos(self, pixel_or_pos):
        self.pixel_or_pos = pixel_or_pos
        self.show_image()

    def set_grid_interpolate(self, grid_interpolate):
        self.grid_interpolate = grid_interpolate
        self.show_image()

    def enable_quantitative_normalization(self, enable):
        """
        Enable/Disable quantitative normalization.

        Parameters
        ----------
        enable: bool
            Enable quantitative normalization if True, disable if False.
        """
        self.quantitative_normalization = bool(enable)
        self.set_low_high_value()  # reset low high values based on normalization
        self.show_image()

    def set_low_high_value(self):
        """Set default low and high values based on normalization for each image."""
        # do not apply scaler norm on not scalable data
        self.range_dict.clear()

        for data_name in self.dict_to_plot.keys():

            if self.quantitative_normalization:
                # Quantitative normalization
                data_arr, _ = self.img_model_adv.param_quant_analysis.apply_quantitative_normalization(
                    data_in=self.dict_to_plot[data_name],
                    scaler_dict=self.scaler_norm_dict,
                    scaler_name_default=self.get_selected_scaler_name(),
                    data_name=data_name,
                    name_not_scalable=self.name_not_scalable,
                )
            else:
                # Normalize by the selected scaler in a regular way
                data_arr = normalize_data_by_scaler(
                    data_in=self.dict_to_plot[data_name],
                    scaler=self.scaler_data,
                    data_name=data_name,
                    name_not_scalable=self.name_not_scalable,
                )

            lowv, highv = np.min(data_arr), np.max(data_arr)
            # Create some 'artificially' small range in case the array is constant
            if lowv == highv:
                lowv -= 0.005
                highv += 0.005
            self.range_dict[data_name] = {"low": lowv, "low_default": lowv, "high": highv, "high_default": highv}

    def reset_low_high(self, name):
        """Reset low and high value to default based on normalization."""
        self.range_dict[name]["low"] = self.range_dict[name]["low_default"]
        self.range_dict[name]["high"] = self.range_dict[name]["high_default"]
        self.limit_dict[name]["low"] = 0.0
        self.limit_dict[name]["high"] = 100.0
        self.show_image()

    def _init_rgb_dict(self):
        self.rgb_dict = {_: None for _ in self.rgb_keys}

    def init_limits_and_stat(self):
        """
        Set plotting status for all the 2D images.
        Note: 'self.map_keys' must be updated before calling this function!
        """
        self.stat_dict.clear()
        self.stat_dict = {k: False for k in self.map_keys}

        self._init_rgb_dict()

        self.limit_dict.clear()
        self.limit_dict = {k: {"low": 0.0, "high": 100.0} for k in self.map_keys}

        self.set_low_high_value()

    def preprocess_data(self):
        """
        Normalize data or prepare for linear/log plot.
        """

        selected_data = []
        selected_name = []
        quant_norm_applied = []

        rgb_color_to_keys = self.get_rgb_items_for_plot()
        for data_key in rgb_color_to_keys.values():
            if data_key in self.dict_to_plot:
                selected_name.append(data_key)

        if self.scaler_data is not None:
            if np.count_nonzero(self.scaler_data) == 0:
                logger.warning("scaler is zero - scaling was not applied")
            elif len(self.scaler_data[self.scaler_data == 0]) > 0:
                logger.warning("scaler data has zero values")

        for i, k in enumerate(selected_name):
            q_norm_applied = False
            if self.quantitative_normalization:
                # Quantitative normalization
                (
                    data_arr,
                    q_norm_applied,
                ) = self.img_model_adv.param_quant_analysis.apply_quantitative_normalization(
                    data_in=self.dict_to_plot[k],
                    scaler_dict=self.scaler_norm_dict,
                    scaler_name_default=self.get_selected_scaler_name(),
                    data_name=k,
                    name_not_scalable=self.name_not_scalable,
                )
            else:
                # Normalize by the selected scaler in a regular way
                data_arr = normalize_data_by_scaler(
                    data_in=self.dict_to_plot[k],
                    scaler=self.scaler_data,
                    data_name=k,
                    name_not_scalable=self.name_not_scalable,
                )

            selected_data.append(data_arr)
            quant_norm_applied.append(q_norm_applied)

        return selected_data, selected_name, rgb_color_to_keys, quant_norm_applied

    def show_image(self):
        # Don't plot the image if dictionary is empty (causes a lot of issues)
        if not self.io_model.img_dict:
            return

        self.fig.clf()

        self.ax = self.fig.add_subplot(111)
        self.ax_r, self.ax_g, self.ax_b = make_rgb_axes(self.ax, pad=0.02)

        # Check if positions data is available. Positions data may be unavailable
        # (not recorded in HDF5 file) if experiment is has not been completed.
        # While the data from the completed part of experiment may still be used,
        # plotting vs. x-y or scatter plot may not be displayed.
        positions_data_available = False
        if "positions" in self.io_model.img_dict.keys():
            positions_data_available = True

        # Create local copy of self.pixel_or_pos and self.grid_interpolate
        pixel_or_pos_local = self.pixel_or_pos
        grid_interpolate_local = self.grid_interpolate

        # Disable plotting vs x-y coordinates if 'positions' data is not available
        if not positions_data_available:
            if pixel_or_pos_local:
                pixel_or_pos_local = 0  # Switch to plotting vs. pixel number
                logger.error("'Positions' data is not available. Plotting vs. x-y coordinates is disabled")
            if grid_interpolate_local:
                grid_interpolate_local = False  # Switch to plotting vs. pixel number
                logger.error("'Positions' data is not available. Interpolation is disabled.")

        selected_data, selected_names, rgb_color_to_keys, quant_norm_applied = self.preprocess_data()
        selected_data = np.asarray(selected_data)

        # Hide unused axes
        if rgb_color_to_keys["red"] is None:
            self.ax_r.set_visible(False)
        if rgb_color_to_keys["green"] is None:
            self.ax_g.set_visible(False)
        if rgb_color_to_keys["blue"] is None:
            self.ax_b.set_visible(False)

        if selected_data.ndim != 3:
            # There is no data to display. Hide the last axis and exit
            self.ax.set_visible(False)
            return

        def _compute_equal_axes_ranges(x_min, x_max, y_min, y_max):
            """
            Compute ranges for x- and y- axes of the plot. Make sure that the ranges for x- and y-axes are
            always equal and fit the maximum of the ranges for x and y values:
                  max(abs(x_max-x_min), abs(y_max-y_min))
            The ranges are set so that the data is always centered in the middle of the ranges

            Parameters
            ----------

            x_min, x_max, y_min, y_max : float
                lower and upper boundaries of the x and y values

            Returns
            -------

            x_axis_min, x_axis_max, y_axis_min, y_axis_max : float
                lower and upper boundaries of the x- and y-axes ranges
            """

            x_axis_min, x_axis_max, y_axis_min, y_axis_max = x_min, x_max, y_min, y_max
            x_range, y_range = abs(x_max - x_min), abs(y_max - y_min)
            if x_range > y_range:
                y_center = (y_max + y_min) / 2
                y_axis_max = y_center + x_range / 2
                y_axis_min = y_center - x_range / 2
            else:
                x_center = (x_max + x_min) / 2
                x_axis_max = x_center + y_range / 2
                x_axis_min = x_center - y_range / 2

            return x_axis_min, x_axis_max, y_axis_min, y_axis_max

        def _adjust_data_range_using_min_ratio(c_min, c_max, c_axis_range, *, min_ratio=0.01):
            """
            Adjust the range for plotted data along one axis (x or y). The adjusted range is
            applied to the 'extent' attribute of imshow(). The adjusted range is always greater
            than 'axis_range * min_ratio'. Such transformation has no physical meaning
            and performed for aesthetic reasons: stretching the image presentation of
            a scan with only a few lines (1-3) greatly improves visibility of data.

            Parameters
            ----------

            c_min, c_max : float
                boundaries of the data range (along x or y axis)
            c_axis_range : float
                range presented along the same axis

            Returns
            -------

            cmin, c_max : float
                adjusted boundaries of the data range
            """
            c_range = c_max - c_min
            if c_range < c_axis_range * min_ratio:
                c_center = (c_max + c_min) / 2
                c_new_range = c_axis_range * min_ratio
                c_min = c_center - c_new_range / 2
                c_max = c_center + c_new_range / 2
            return c_min, c_max

        if pixel_or_pos_local:

            # xd_min, xd_max, yd_min, yd_max = min(self.x_pos), max(self.x_pos),
            #     min(self.y_pos), max(self.y_pos)
            x_pos_2D = self.io_model.img_dict["positions"]["x_pos"]
            y_pos_2D = self.io_model.img_dict["positions"]["y_pos"]
            xd_min, xd_max, yd_min, yd_max = x_pos_2D.min(), x_pos_2D.max(), y_pos_2D.min(), y_pos_2D.max()
            xd_axis_min, xd_axis_max, yd_axis_min, yd_axis_max = _compute_equal_axes_ranges(
                xd_min, xd_max, yd_min, yd_max
            )

            xd_min, xd_max = _adjust_data_range_using_min_ratio(xd_min, xd_max, xd_axis_max - xd_axis_min)
            yd_min, yd_max = _adjust_data_range_using_min_ratio(yd_min, yd_max, yd_axis_max - yd_axis_min)

            # Adjust the direction of each axis depending on the direction in which encoder values changed
            #   during the experiment. Data is plotted starting from the upper-right corner of the plot
            if x_pos_2D[0, 0] > x_pos_2D[0, -1]:
                xd_min, xd_max, xd_axis_min, xd_axis_max = xd_max, xd_min, xd_axis_max, xd_axis_min
            if y_pos_2D[0, 0] > y_pos_2D[-1, 0]:
                yd_min, yd_max, yd_axis_min, yd_axis_max = yd_max, yd_min, yd_axis_max, yd_axis_min

        else:
            if selected_data.ndim == 3:
                # Set equal ranges for the axes data
                yd, xd = selected_data.shape[1], selected_data.shape[2]
                xd_min, xd_max, yd_min, yd_max = 0, xd, 0, yd
                # Select minimum range for data
                if (yd <= math.floor(xd / 100)) and (xd >= 200):
                    yd_min, yd_max = -math.floor(xd / 200), math.ceil(xd / 200)
                if (xd <= math.floor(yd / 100)) and (yd >= 200):
                    xd_min, xd_max = -math.floor(yd / 200), math.ceil(yd / 200)

                xd_axis_min, xd_axis_max, yd_axis_min, yd_axis_max = _compute_equal_axes_ranges(
                    xd_min, xd_max, yd_min, yd_max
                )

        name_r, data_r, limits_r = "", None, {"low": 0, "high": 100.0}
        name_g, data_g, limits_g = "", None, {"low": 0, "high": 100.0}
        name_b, data_b, limits_b = "", None, {"low": 0, "high": 100.0}
        for color, name in rgb_color_to_keys.items():
            if name:
                try:
                    ind = selected_names.index(name)
                    name_label = name
                    if quant_norm_applied[ind]:
                        name_label += " - Q"  # Add suffix to name if quantitative normalization was applied
                    if color == "red":
                        name_r, data_r = name_label, selected_data[ind]
                        limits_r = self.limit_dict[name]
                    elif color == "green":
                        name_g, data_g = name_label, selected_data[ind]
                        limits_g = self.limit_dict[name]
                    elif color == "blue":
                        name_b, data_b = name_label, selected_data[ind]
                        limits_b = self.limit_dict[name]
                except ValueError:
                    pass

        def _norm_data(data):
            """
            Normalize data between (0, 1).
            Parameters
            ----------
            data : 2D array
            """
            if data is None:
                return data
            data_min = np.min(data)
            c_norm = np.max(data) - data_min
            return (data - data_min) / c_norm if (c_norm != 0) else (data - data_min)

        def _stretch_range(data_in, v_low, v_high):

            # 'data is already normalized, so that the values are in the range 0..1
            # v_low, v_high are in the range 0..100
            if data_in is None:
                return data_in

            if (v_low <= 0) and (v_high >= 100):
                return data_in

            if v_high - v_low < 1:  # This should not happen in practice, but check just in case
                v_high = v_low + 1

            v_low, v_high = v_low / 100.0, v_high / 100.0
            c = 1.0 / (v_high - v_low)
            data_out = (data_in - v_low) * c

            return np.clip(data_out, 0, 1.0)

        # Interpolate non-uniformly spaced data to uniform grid
        if grid_interpolate_local:
            data_r, _, _ = grid_interpolate(
                data_r, self.io_model.img_dict["positions"]["x_pos"], self.io_model.img_dict["positions"]["y_pos"]
            )
            data_g, _, _ = grid_interpolate(
                data_g, self.io_model.img_dict["positions"]["x_pos"], self.io_model.img_dict["positions"]["y_pos"]
            )
            data_b, _, _ = grid_interpolate(
                data_b, self.io_model.img_dict["positions"]["x_pos"], self.io_model.img_dict["positions"]["y_pos"]
            )

        # The dictionaries 'rgb_view_data' and 'pos_limits' are used for monitoring
        #   the map values at current cursor positions.
        rgb_view_data = {_: None for _ in self.rgb_keys}
        if data_r is not None:
            rgb_view_data["red"] = data_r
        if data_g is not None:
            rgb_view_data["green"] = data_g
        if data_b is not None:
            rgb_view_data["blue"] = data_b
        pos_limits = {"x_low": xd_min, "x_high": xd_max, "y_low": yd_min, "y_high": yd_max}

        # Normalize data
        data_r_norm = _norm_data(data_r)
        data_g_norm = _norm_data(data_g)
        data_b_norm = _norm_data(data_b)

        data_r_norm = _stretch_range(data_r_norm, limits_r["low"], limits_r["high"])
        data_g_norm = _stretch_range(data_g_norm, limits_g["low"], limits_g["high"])
        data_b_norm = _stretch_range(data_b_norm, limits_b["low"], limits_b["high"])

        R, G, B, RGB = make_cube(data_r_norm, data_g_norm, data_b_norm)

        red_patch = mpatches.Patch(color="red", label=name_r)
        green_patch = mpatches.Patch(color="green", label=name_g)
        blue_patch = mpatches.Patch(color="blue", label=name_b)

        def format_coord_func(x, y, *, pixel_or_pos, rgb_color_to_keys, rgb_view_data, pos_limits, colors=None):
            x0, y0 = pos_limits["x_low"], pos_limits["y_low"]
            if colors is None:
                colors = list(rgb_color_to_keys.keys())

            s = ""
            for n, color in enumerate(self.rgb_keys):
                if (color not in colors) or (rgb_color_to_keys[color] is None) or (rgb_view_data[color] is None):
                    continue
                map = rgb_view_data[color]

                ny, nx = map.shape
                dy = (pos_limits["y_high"] - y0) / ny if ny else 0
                dx = (pos_limits["x_high"] - x0) / nx if nx else 0
                cy = 1 / dy if dy else 1
                cx = 1 / dx if dx else 1

                x_pixel = math.floor((x - x0) * cx)
                y_pixel = math.floor((y - y0) * cy)

                if (0 <= x_pixel < nx) and (0 <= y_pixel < ny):
                    # The following line is extremely useful for debugging the feature. Keep it.
                    # s += f" <b>{rgb_color_to_keys[color]}</b>: {x_pixel} {y_pixel}"
                    s += f" <b>{rgb_color_to_keys[color]}</b>: {map[y_pixel, x_pixel]:.5g}"

            s = " - " + s if s else s  # Add dash if something is to be printed

            if pixel_or_pos:
                # Spatial coordinates (double)
                s_coord = f"({x:.5g}, {y:.5g})"
            else:
                # Pixel coordinates (int)
                s_coord = f"({int(x)}, {int(y)})"

            return s_coord + s

        format_coord = partial(
            format_coord_func,
            pixel_or_pos=pixel_or_pos_local,
            rgb_color_to_keys=rgb_color_to_keys,
            rgb_view_data=rgb_view_data,
            pos_limits=pos_limits,
        )

        def format_cursor_data(data):
            return ""  # Print nothing

        kwargs = dict(origin="upper", interpolation="nearest", extent=(xd_min, xd_max, yd_max, yd_min))
        if RGB is not None:
            img = self.ax.imshow(RGB, **kwargs)

            self.ax.format_coord = format_coord
            img.format_cursor_data = format_cursor_data

            self.ax.set_xlim(xd_axis_min, xd_axis_max)
            self.ax.set_ylim(yd_axis_max, yd_axis_min)

        if R is not None:
            img = self.ax_r.imshow(R, **kwargs)
            self.ax_r.set_xlim(xd_axis_min, xd_axis_max)
            self.ax_r.set_ylim(yd_axis_max, yd_axis_min)

            format_coord_r = partial(format_coord, colors=["red"])
            self.ax_r.format_coord = format_coord_r
            img.format_cursor_data = format_cursor_data

        if G is not None:
            img = self.ax_g.imshow(G, **kwargs)
            self.ax_g.set_xlim(xd_axis_min, xd_axis_max)
            self.ax_g.set_ylim(yd_axis_max, yd_axis_min)

            format_coord_g = partial(format_coord, colors=["green"])
            self.ax_g.format_coord = format_coord_g
            img.format_cursor_data = format_cursor_data

        if B is not None:
            img = self.ax_b.imshow(B, **kwargs)
            self.ax_b.set_xlim(xd_axis_min, xd_axis_max)
            self.ax_b.set_ylim(yd_axis_max, yd_axis_min)

            format_coord_b = partial(format_coord, colors=["blue"])
            self.ax_b.format_coord = format_coord_b
            img.format_cursor_data = format_cursor_data

        self.ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins="auto"))
        self.ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins="auto"))

        plt.setp(self.ax_r.get_xticklabels(), visible=False)
        plt.setp(self.ax_r.get_yticklabels(), visible=False)
        plt.setp(self.ax_g.get_xticklabels(), visible=False)
        plt.setp(self.ax_g.get_yticklabels(), visible=False)
        plt.setp(self.ax_b.get_xticklabels(), visible=False)
        plt.setp(self.ax_b.get_yticklabels(), visible=False)

        # self.ax_r.set_xticklabels([])
        # self.ax_r.set_yticklabels([])

        # sb_x = 38
        # sb_y = 46
        # sb_length = 10
        # sb_height = 1
        # ax.add_patch(mpatches.Rectangle(( sb_x, sb_y), sb_length, sb_height, color='white'))
        # ax.text(sb_x + sb_length /2, sb_y - 1*sb_height,  '100 nm', color='w', ha='center',
        #         va='bottom', backgroundcolor='black', fontsize=18)

        self.ax_r.legend(
            loc="upper left",
            bbox_to_anchor=(1.1, 0),
            frameon=False,
            handles=[red_patch, green_patch, blue_patch],
            mode="expand",
        )

        # self.fig.tight_layout(pad=4.0, w_pad=0.8, h_pad=0.8)
        # self.fig.tight_layout()
        # self.fig.canvas.draw_idle()
        # self.fig.suptitle(self.img_title, fontsize=20)

        self.fig.canvas.draw_idle()

    def get_selected_items_for_plot(self):
        """Collect the selected items for plotting."""
        # We want the dictionary to be sorted the same way as 'map_keys'
        sdict = self.stat_dict
        selected_keys = [_ for _ in self.map_keys if (_ in sdict) and (sdict[_] is True)]
        return selected_keys

    def get_rgb_items_for_plot(self):
        # Verify integrity of the dictionary
        if len(self.rgb_dict) != 3:
            raise ValueError(
                "DrawImageRGB.get_rgb_items_for_plot: dictionary 'rgb_dict' has "
                f"{len(self.rgb_dict)} elements. Expected number of elements: "
                f"{len(self.rgb_keys)}."
            )
        for key in self.rgb_keys:
            if key not in self.rgb_dict:
                raise ValueError(
                    "DrawImageRGB.get_rgb_items_for_plot: dictionary 'rgb_dict' is "
                    f"incomplete or contains incorrect set of keys: {list(self.rgb_dict.keys())}. "
                    f"Expected keys: {self.rgb_keys}: "
                )
        return self.rgb_dict


def make_cube(r, g, b):
    """
    Create 3D array for rgb image.
    Parameters
    ----------
    r : 2D array
    g : 2D array
    b : 2D array
    """
    if r is None and g is None and b is None:
        logger.error("'make_cube': 'r', 'g' and 'b' input arrays are all None")
        R, G, B, RGB = None

    else:
        for arr in [r, g, b]:
            if arr is not None:
                ny, nx = arr.shape
                break

        R = np.zeros([ny, nx, 3])
        R[:, :, 0] = r
        G = np.zeros_like(R)
        G[:, :, 1] = g
        B = np.zeros_like(R)
        B[:, :, 2] = b

        RGB = R + G + B

    return R, G, B, RGB
