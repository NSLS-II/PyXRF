from __future__ import (absolute_import, division,
                        print_function)

import six
import numpy as np
import math
from collections import OrderedDict
from matplotlib.figure import Figure, Axes
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1.axes_rgb import make_rgb_axes
from atom.api import Atom, Str, observe, Typed, Int, List, Dict, Bool

from ..core.utils import normalize_data_by_scaler, grid_interpolate

import logging
logger = logging.getLogger()

np.seterr(divide='ignore', invalid='ignore')  # turn off warning on invalid division

#
# class plot_limit(Atom):
#     low = Float(0)
#     high = Float(100)
#     # g_low = Float(0)
#     # g_high = Float(100)
#     # b_low = Float(0)
#     # b_high = Float(100)


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
    data_dict : dict
        multiple data sets to plot, such as fit data, or roi data
    data_dict_keys : list
    data_opt : int
        index to show which data is chosen to plot
    dict_to_plot : dict
        selected data dict to plot, i.e., fitting data or roi is selected
    items_in_selected_group : list
        keys of dict_to_plot
    color_opt : str
        orange or gray plot
    scaler_norm_dict : dict
        scaler normalization data, from data_dict
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

    fig = Typed(Figure)
    ax = Typed(Axes)
    ax_r = Typed(Axes)
    ax_g = Typed(Axes)
    ax_b = Typed(Axes)
    stat_dict = Dict()
    data_dict = Dict()
    data_dict_keys = List()
    data_opt = Int(0)
    img_title = Str()
    # plot_opt = Int(0)
    # plot_item = Str()
    dict_to_plot = Dict()
    items_in_selected_group = List()
    scaler_norm_dict = Dict()
    scaler_items = List()
    scaler_name_index = Int()
    scaler_data = Typed(object)
    pixel_or_pos = Int(0)
    grid_interpolate = Bool(False)
    plot_all = Bool(False)

    rgb_name_list = List()
    index_red = Int(0)
    index_green = Int(1)
    index_blue = Int(2)
    # ic_norm = Float()
    rgb_limit = Dict()
    r_low = Int(0)
    r_high = Int(100)
    g_low = Int(0)
    g_high = Int(100)
    b_low = Int(0)
    b_high = Int(100)
    # r_bound = List()
    # rgb_limit = plot_limit()
    name_not_scalable = List()

    def __init__(self):
        self.rgb_name_list = ['R', 'G', 'B']

    def data_dict_update(self, change):
        """
        Observer function to be connected to the fileio model
        in the top-level gui.py startup

        Parameters
        ----------
        change : dict
            This is the dictionary that gets passed to a function
            with the @observe decorator
        """
        self.data_dict = change['value']

    @observe('data_dict')
    def init_plot_status(self, change):
        # initiate the plotting status once new data is coming
        self.data_opt = 0
        self.rgb_name_list = ['R', 'G', 'B']
        self.index_red = 0
        self.index_green = 1
        self.index_blue = 2

        # init of pos values
        self.pixel_or_pos = 0

        # init of scaler for normalization
        self.scaler_name_index = 0
        self.data_dict_keys = []
        self.data_dict_keys = list(self.data_dict.keys())
        # logger.info('The following groups are included for 2D image display: {}'.format(self.data_dict_keys))

        scaler_groups = [v for v in list(self.data_dict.keys()) if 'scaler' in v]
        if len(scaler_groups) > 0:
            # self.scaler_group_name = scaler_groups[0]
            self.scaler_norm_dict = self.data_dict[scaler_groups[0]]
            # for GUI purpose only
            self.scaler_items = []
            self.scaler_items = list(self.scaler_norm_dict.keys())
            self.scaler_items.sort()
            self.scaler_data = None

        self.show_image()

    @observe('data_opt')
    def _update_file(self, change):
        try:
            if self.data_opt == 0:
                self.dict_to_plot = {}
                self.items_in_selected_group = []
                self.set_stat_for_all(bool_val=False)
                self.img_title = ''
            elif self.data_opt > 0:
                self.set_stat_for_all(bool_val=False)
                plot_item = sorted(self.data_dict_keys)[self.data_opt-1]
                self.img_title = str(plot_item)
                self.dict_to_plot = self.data_dict[plot_item]
                # for GUI purpose only
                self.items_in_selected_group = []
                self.items_in_selected_group = list(self.dict_to_plot.keys())
                self.set_stat_for_all(bool_val=False)
                # set rgb value to 0 and 100
                self.init_rgb()
        except IndexError:
            pass

    def init_rgb(self):
        self.r_low = 0
        self.r_high = 100
        self.g_low = 0
        self.g_high = 100
        self.b_low = 0
        self.b_high = 100

    @observe('scaler_name_index')
    def _get_scaler_data(self, change):

        if self.scaler_name_index == 0:
            self.scaler_data = None
        else:
            scaler_name = self.scaler_items[self.scaler_name_index-1]
            # self.scaler_data = self.data_dict[self.scaler_group_name][scaler_name]
            self.scaler_data = self.scaler_norm_dict[scaler_name]
            logger.info('Use scaler data to normalize,'
                        'and the shape of scaler data is {}'.format(self.scaler_data.shape))
        self.show_image()

    @observe('pixel_or_pos')
    def _update_pp(self, change):
        self.show_image()

    @observe('grid_interpolate')
    def _update_gi(self, change):
        self.show_image()

    def set_stat_for_all(self, bool_val=False):
        """
        Set plotting status for all the 2D images.
        """
        self.stat_dict.clear()
        self.stat_dict = {k: bool_val for k in self.items_in_selected_group}

    def preprocess_data(self):
        """
        Normalize data or prepare for linear/log plot.
        """

        selected_data = []
        selected_name = []

        stat_temp = self.get_activated_num()
        stat_temp = OrderedDict(sorted(six.iteritems(stat_temp), key=lambda x: x[0]))

        # plot_interp = 'Nearest'

        if self.scaler_data is not None:
            if np.count_nonzero(self.scaler_data) == 0:
                logger.warning('scaler is zero - scaling was not applied')
            elif len(self.scaler_data[self.scaler_data == 0]) > 0:
                logger.warning('scaler data has zero values')

        for i, (k, v) in enumerate(six.iteritems(stat_temp)):

            data_dict = normalize_data_by_scaler(self.dict_to_plot[k], self.scaler_data,
                                                 data_name=k, name_not_scalable=self.name_not_scalable)

            selected_data.append(data_dict)
            selected_name.append(k)  # self.file_name+'_'+str(k)

        return selected_data, selected_name

    # @observe('r_low', 'r_high', 'g_low', 'g_high', 'b_low', 'b_high')
    # def _update_scale(self, change):
    #     if change['type'] != 'create':
    #         self.show_image()

    def show_image(self):

        self.fig = plt.figure(figsize=(3, 2))
        self.ax = self.fig.add_subplot(111)
        self.ax_r, self.ax_g, self.ax_b = make_rgb_axes(self.ax, pad=0.02)
        self.name_not_scalable = ['r2_adjust']  # do not apply scaler norm on those data

        # Check if positions data is available. Positions data may be unavailable
        # (not recorded in HDF5 file) if experiment is has not been completed.
        # While the data from the completed part of experiment may still be used,
        # plotting vs. x-y or scatter plot may not be displayed.
        positions_data_available = False
        if 'positions' in self.data_dict.keys():
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

        selected_data, selected_name = self.preprocess_data()
        selected_data = np.asarray(selected_data)

        if len(selected_name) != 3:
            logger.error('Please select three elements for RGB plot.')
            return
        self.rgb_name_list = selected_name[:3]

        try:
            data_r = selected_data[0, :, :]
        except IndexError:
            selected_data = np.ones([3, 10, 10])

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
            x_pos_2D = self.data_dict['positions']['x_pos']
            y_pos_2D = self.data_dict['positions']['y_pos']
            xd_min, xd_max, yd_min, yd_max = x_pos_2D.min(), x_pos_2D.max(), y_pos_2D.min(), y_pos_2D.max()
            xd_axis_min, xd_axis_max, yd_axis_min, yd_axis_max = \
                _compute_equal_axes_ranges(xd_min, xd_max, yd_min, yd_max)

            xd_min, xd_max = _adjust_data_range_using_min_ratio(xd_min, xd_max, xd_axis_max - xd_axis_min)
            yd_min, yd_max = _adjust_data_range_using_min_ratio(yd_min, yd_max, yd_axis_max - yd_axis_min)

            # Adjust the direction of each axis depending on the direction in which encoder values changed
            #   during the experiment. Data is plotted starting from the upper-right corner of the plot
            if x_pos_2D[0, 0] > x_pos_2D[0, -1]:
                xd_min, xd_max, xd_axis_min, xd_axis_max = xd_max, xd_min, xd_axis_max, xd_axis_min
            if y_pos_2D[0, 0] > y_pos_2D[-1, 0]:
                yd_min, yd_max, yd_axis_min, yd_axis_max = yd_max, yd_min, yd_axis_max, yd_axis_min

        else:

            # Set equal ranges for the axes data
            yd, xd = selected_data.shape[1], selected_data.shape[2]
            xd_min, xd_max, yd_min, yd_max = 0, xd, 0, yd
            # Select minimum range for data
            if (yd <= math.floor(xd / 100)) and (xd >= 200):
                yd_min, yd_max = -math.floor(xd / 200), math.ceil(xd / 200)
            if (xd <= math.floor(yd / 100)) and (yd >= 200):
                xd_min, xd_max = -math.floor(yd / 200), math.ceil(yd / 200)

            xd_axis_min, xd_axis_max, yd_axis_min, yd_axis_max = \
                _compute_equal_axes_ranges(xd_min, xd_max, yd_min, yd_max)

        name_r = self.rgb_name_list[self.index_red]
        data_r = selected_data[self.index_red, :, :]
        name_g = self.rgb_name_list[self.index_green]
        data_g = selected_data[self.index_green, :, :]
        name_b = self.rgb_name_list[self.index_blue]
        data_b = selected_data[self.index_blue, :, :]

        rgb_l_h = ({'low': self.r_low, 'high': self.r_high},
                   {'low': self.g_low, 'high': self.g_high},
                   {'low': self.b_low, 'high': self.b_high})

        def _norm_data(data):
            """
            Normalize data between (0, 1).
            Parameters
            ----------
            data : 2D array
            """
            data_min = np.min(data)
            c_norm = np.max(data) - data_min
            return (data - data_min) / c_norm if (c_norm != 0) else (data - data_min)

        def _stretch_range(data_in, v_low, v_high):

            # 'data is already normalized, so that the values are in the range 0..1
            # v_low, v_high are in the range 0..100

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
            data_r, _, _ = grid_interpolate(data_r,
                                            self.data_dict['positions']['x_pos'],
                                            self.data_dict['positions']['y_pos'])
            data_g, _, _ = grid_interpolate(data_g,
                                            self.data_dict['positions']['x_pos'],
                                            self.data_dict['positions']['y_pos'])
            data_b, _, _ = grid_interpolate(data_b,
                                            self.data_dict['positions']['x_pos'],
                                            self.data_dict['positions']['y_pos'])

        # Normalize data
        data_r = _norm_data(data_r)
        data_g = _norm_data(data_g)
        data_b = _norm_data(data_b)

        data_r = _stretch_range(data_r, rgb_l_h[self.index_red]['low'], rgb_l_h[self.index_red]['high'])
        data_g = _stretch_range(data_g, rgb_l_h[self.index_green]['low'], rgb_l_h[self.index_green]['high'])
        data_b = _stretch_range(data_b, rgb_l_h[self.index_blue]['low'], rgb_l_h[self.index_blue]['high'])

        R, G, B, RGB = make_cube(data_r,
                                 data_g,
                                 data_b)

        red_patch = mpatches.Patch(color='red', label=name_r)
        green_patch = mpatches.Patch(color='green', label=name_g)
        blue_patch = mpatches.Patch(color='blue', label=name_b)

        kwargs = dict(origin="upper", interpolation="nearest", extent=(xd_min, xd_max, yd_max, yd_min))
        self.ax.imshow(RGB, **kwargs)
        self.ax_r.imshow(R, **kwargs)
        self.ax_g.imshow(G, **kwargs)
        self.ax_b.imshow(B, **kwargs)

        self.ax.set_xlim(xd_axis_min, xd_axis_max)
        self.ax.set_ylim(yd_axis_max, yd_axis_min)
        self.ax_r.set_xlim(xd_axis_min, xd_axis_max)
        self.ax_r.set_ylim(yd_axis_max, yd_axis_min)
        self.ax_g.set_xlim(xd_axis_min, xd_axis_max)
        self.ax_g.set_ylim(yd_axis_max, yd_axis_min)
        self.ax_b.set_xlim(xd_axis_min, xd_axis_max)
        self.ax_b.set_ylim(yd_axis_max, yd_axis_min)

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

        self.ax.legend(bbox_to_anchor=(0., 1.0, 1., .10), ncol=3,
                       handles=[red_patch, green_patch, blue_patch], mode="expand", loc=3)

        # self.fig.tight_layout(pad=4.0, w_pad=0.8, h_pad=0.8)
        # self.fig.tight_layout()
        # self.fig.canvas.draw_idle()
        self.fig.suptitle(self.img_title, fontsize=20)
        self.fig.canvas.draw_idle()

    def get_activated_num(self):
        return {k: v for (k, v) in six.iteritems(self.stat_dict) if v is True}


def make_cube(r, g, b):
    """
    Create 3D array for rgb image.
    Parameters
    ----------
    r : 2D array
    g : 2D array
    b : 2D array
    """
    ny, nx = r.shape
    R = np.zeros([ny, nx, 3])
    R[:, :, 0] = r
    G = np.zeros_like(R)
    G[:, :, 1] = g
    B = np.zeros_like(R)
    B[:, :, 2] = b

    RGB = R + G + B

    return R, G, B, RGB
