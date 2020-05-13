from __future__ import absolute_import

import six
import numpy as np
from collections import OrderedDict
# from scipy.interpolate import interp1d, interp2d
import copy
import re

import math
import matplotlib
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1 import ImageGrid
from atom.api import Atom, Str, observe, Typed, Int, List, Dict, Bool, Float

from ..core.utils import normalize_data_by_scaler, grid_interpolate

from ..core.quant_analysis import ParamQuantitativeAnalysis

import logging
logger = logging.getLogger()


class DrawImageAdvanced(Atom):
    """
    This class performs 2D image rendering, such as showing multiple
    2D fitting or roi images based on user's selection.

    Attributes
    ----------
    img_data : dict
        dict of 2D array
    fig : object
        matplotlib Figure
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
    scale_opt : str
        linear or log plot
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
    x_pos : list
        define data range in horizontal direction
    y_pos : list
        define data range in vertical direction
    pixel_or_pos : int
        index to choose plot with pixel (== 0) or with positions (== 1)
    grid_interpolate: bool
        choose to interpolate 2D image in terms of x,y or not
    limit_dict : Dict
        save low and high limit for image scaling
    """

    fig = Typed(Figure)
    stat_dict = Dict()
    data_dict = Dict()
    data_dict_keys = List()
    data_opt = Int(0)
    dict_to_plot = Dict()
    items_in_selected_group = List()
    items_previous_selected = List()

    scale_opt = Str('Linear')
    color_opt = Str('viridis')
    img_title = Str()

    scaler_norm_dict = Dict()
    scaler_items = List()
    scaler_name_index = Int()
    scaler_data = Typed(object)

    x_pos = List()
    y_pos = List()

    pixel_or_pos = Int(0)
    grid_interpolate = Bool(False)
    data_dict_default = Dict()
    limit_dict = Dict()
    range_dict = Dict()
    scatter_show = Bool(False)
    name_not_scalable = List()

    # The value is synchronized with currently selected value of incident energy
    #   Should not be changed directly
    incident_energy = Float(12.0)

    # Variable that indicates whether quanitative normalization should be applied to data
    #   Associated with 'Quantitative' checkbox
    quantitative_normalization = Bool(False)

    # The following fields are used for storing parameters used for quantitative analysis
    param_quant_analysis = Typed(object)
    # Distance to sample for the scan that is currently processed, synchronized with the contents
    #   of the field in 'Quantitative Analysis' tab, may be used for computations
    #   Enaml 'FloatField' requires this variable to be a string, but the string will always
    #   represent valid 'float' number
    quant_distance_to_sample = Float(0.0)
    # The following fields are used to facilitate GUI operation, not for long-term data storage
    #   Don't use those fields for any computations outside the GUI controls
    quant_calibration_data = List()
    quant_calibration_settings = List()
    quant_active_emission_lines = List()
    quant_calibration_data_preview = Str()

    def __init__(self):
        self.fig = plt.figure(figsize=(3, 2))
        matplotlib.rcParams['axes.formatter.useoffset'] = True

        # Do not apply scaler norm on following data
        self.name_not_scalable = ['r2_adjust', 'r_factor', 'alive', 'dead', 'elapsed_time',
                                  'scaler_alive', 'i0_time', 'time', 'time_diff', 'dwell_time']

        self.param_quant_analysis = ParamQuantitativeAnalysis()
        self.param_quant_analysis.set_experiment_distance_to_sample(distance_to_sample=0.0)
        self.param_quant_analysis.set_experiment_incident_energy(incident_energy=self.incident_energy)

    def data_dict_update(self, change):
        """
        Observer function to be connected to the fileio model
        in the top-level gui.py startup

        Parameters
        ----------
        changed : dict
            This is the dictionary that gets passed to a function
            with the @observe decorator
        """
        self.data_dict = change['value']

    def set_default_dict(self, data_dict):
        self.data_dict_default = copy.deepcopy(data_dict)

    @observe('data_dict')
    def init_plot_status(self, change):
        scaler_groups = [v for v in list(self.data_dict.keys()) if 'scaler' in v]
        if len(scaler_groups) > 0:
            # self.scaler_group_name = scaler_groups[0]
            self.scaler_norm_dict = self.data_dict[scaler_groups[0]]
            # for GUI purpose only
            self.scaler_items = []
            self.scaler_items = list(self.scaler_norm_dict.keys())
            self.scaler_items.sort()
            self.scaler_data = None

        # init of pos values
        self.pixel_or_pos = 0

        if 'positions' in self.data_dict:
            try:
                logger.debug(f"Position keys: {list(self.data_dict['positions'].keys())}")
                self.x_pos = list(self.data_dict['positions']['x_pos'][0, :])
                self.y_pos = list(self.data_dict['positions']['y_pos'][:, -1])
                # when we use imshow, the x and y start at lower left,
                # so flip y, we want y starts from top left
                self.y_pos.reverse()

            except KeyError:
                pass
        else:
            self.x_pos = []
            self.y_pos = []

        self.get_default_items()   # use previous defined elements as default
        logger.info('Use previously selected items as default: {}'.format(self.items_previous_selected))

        # initiate the plotting status once new data is coming
        self.reset_to_default()
        self.data_dict_keys = []
        self.data_dict_keys = list(self.data_dict.keys())
        logger.debug('The following groups are included for 2D image display: {}'.format(self.data_dict_keys))

        self.show_image()

    def reset_to_default(self):
        """Set variables to default values as initiated.
        """
        self.data_opt = 0
        # init of scaler for normalization
        self.scaler_name_index = 0
        self.plot_deselect_all()

    def get_default_items(self):
        """Add previous selected items as default.
        """
        if len(self.items_previous_selected) != 0:
            default_items = {}
            for item in self.items_previous_selected:
                for v, k in self.data_dict.items():
                    if item in k:
                        default_items[item] = k[item]
            self.data_dict['use_default_selection'] = default_items

    def get_detector_channel_name(self):
        # Return the name of the selected detector channel ('sum', 'det1', 'det2' etc)
        #   The channel name is extracted from 'self.img_title' (selected in 'ElementMap' tab)
        if self.img_title:
            if(re.search(r"_det\d+_fit$", self.img_title)):
                s = re.search(r"_det\d+_", self.img_title)[0]
                return s.strip('_')
            if(re.search(r"_fit", self.img_title)):
                return "sum"
        else:
            return None

    @observe('data_opt')
    def _update_file(self, change):
        try:
            if self.data_opt == 0:
                self.dict_to_plot = {}
                self.items_in_selected_group = []
                self.set_stat_for_all(bool_val=False)
                self.img_title = ''
            elif self.data_opt > 0:
                # self.set_stat_for_all(bool_val=False)
                plot_item = sorted(self.data_dict_keys)[self.data_opt-1]
                self.img_title = str(plot_item)
                self.dict_to_plot = self.data_dict[plot_item]
                self.set_stat_for_all(bool_val=False)

                self.update_img_wizard_items()
                self.get_default_items()   # get default elements every time when fitting is done

            # The detector channel name should be updated in any case
            self.param_quant_analysis.experiment_detector_channel = self.get_detector_channel_name()

        except IndexError:
            pass

    def get_selected_scaler_name(self):
        if self.scaler_name_index == 0:
            return None
        else:
            return self.scaler_items[self.scaler_name_index - 1]

    @observe('scaler_name_index')
    def _get_scaler_data(self, change):
        if change['type'] == 'create':
            return

        if self.scaler_name_index == 0:
            self.scaler_data = None
        else:
            try:
                scaler_name = self.scaler_items[self.scaler_name_index-1]
            except IndexError:
                scaler_name = None
            if scaler_name:
                self.scaler_data = self.scaler_norm_dict[scaler_name]
                logger.info('Use scaler data to normalize, '
                            'and the shape of scaler data is {}, '
                            'with (low, high) as ({}, {})'.format(self.scaler_data.shape,
                                                                  np.min(self.scaler_data),
                                                                  np.max(self.scaler_data)))
        self.set_low_high_value()  # reset low high values based on normalization
        self.show_image()
        self.update_img_wizard_items()

    # TODO: document the following functions
    def update_quant_calibration_gui(self):
        self.quant_calibration_data = []
        self.quant_calibration_data = self.param_quant_analysis.calibration_data
        self.quant_calibration_settings = []
        self.quant_calibration_settings = self.param_quant_analysis.calibration_settings
        self.quant_active_emission_lines = []
        self.quant_active_emission_lines = self.param_quant_analysis.active_emission_lines

    def load_quantitative_calibration_data(self, file_path):
        self.param_quant_analysis.load_entry(file_path)
        self.update_quant_calibration_gui()

    def remove_quantitative_calibration_data(self, file_path):
        try:
            self.param_quant_analysis.remove_entry(file_path)
            self.update_quant_calibration_gui()
        except Exception as ex:
            logger.error(f"Calibration data was not removed: {ex}")

    def update_img_wizard_items(self):
        """This is for GUI purpose only.
        Table items will not be updated if list items keep the same.
        """
        self.items_in_selected_group = []
        self.items_in_selected_group = list(self.dict_to_plot.keys())

    def format_img_wizard_limit(self, value):
        """
        This function is used for formatting of range values in 'Image Wizard'.
        The presentation of the number was tweaked so that it is nicely formatted
           in the enaml field with adequate precision.

        ..note::

        The function is called externally from 'enaml' code.

        Parameters:
        ===========
        value : float
            The value to be formatted

        Returns:
        ========
        str - the string representation of the floating point variable
        """
        if value != 0:
            value_log10 = math.log10(abs(value))
        else:
            value_log10 = 0
        if (value_log10 > 3) or (value_log10 < -3):
            return f"{value:.6e}"
        return f"{value:.6f}"

    @observe('scale_opt', 'color_opt')
    def _update_scale(self, change):
        if change['type'] != 'create':
            self.show_image()

    @observe('pixel_or_pos')
    def _update_pp(self, change):
        self.show_image()

    @observe('grid_interpolate')
    def _update_gi(self, change):
        self.show_image()

    @observe('quantitative_normalization')
    def _update_qn(self, change):

        # Propagate current value of 'self.param_quant_analysis' (activate 'observer' functions)
        tmp = self.param_quant_analysis
        self.param_quant_analysis = ParamQuantitativeAnalysis()
        self.param_quant_analysis = tmp

        self.set_low_high_value()  # reset low high values based on normalization
        self.show_image()
        self.update_img_wizard_items()

    def plot_select_all(self):
        self.set_stat_for_all(bool_val=True)

    def plot_deselect_all(self):
        self.set_stat_for_all(bool_val=False)

    @observe('scatter_show')
    def _change_image_plot_method(self, change):
        if change['type'] != 'create':
            self.show_image()

    @observe('quant_distance_to_sample')
    def _on_change_distance_to_sample(self, change):
        # Set the value of the quantitative analysis parameter
        self.param_quant_analysis.set_experiment_distance_to_sample(self.quant_distance_to_sample)
        # Recompute range of plotted values (for each emission line in the dataset)
        self.set_low_high_value()
        # Update limits shown in 'Image Wizard'
        self.update_img_wizard_items()

    def set_incident_energy(self, change):
        """
        The observer function that changes the value of incident energy
        and upper bound for fitted energy range. Should not be called directly.

        Parameters
        ----------

        change : dict
            ``change["value"]`` is the new value of incident energy
        """
        self.incident_energy = change["value"]

    @observe('incident_energy')
    def _on_change_incident_energy(self, change):
        self.param_quant_analysis.set_experiment_incident_energy(incident_energy=self.incident_energy)

    def set_stat_for_all(self, bool_val=False):
        """
        Set plotting status for all the 2D images, including low and high values.
        """
        self.stat_dict.clear()
        self.stat_dict = {k: bool_val for k in self.dict_to_plot.keys()}

        self.limit_dict.clear()
        self.limit_dict = {k: {'low': 0.0, 'high': 100.0} for k in self.dict_to_plot.keys()}

        self.set_low_high_value()

    def set_low_high_value(self):
        """Set default low and high values based on normalization for each image.
        """
        # do not apply scaler norm on not scalable data
        self.range_dict.clear()

        for data_name in self.dict_to_plot.keys():

            if self.quantitative_normalization:
                # Quantitative normalization
                data_arr, _ = self.param_quant_analysis.apply_quantitative_normalization(
                    data_in=self.dict_to_plot[data_name],
                    scaler_dict=self.scaler_norm_dict,
                    scaler_name_default=self.get_selected_scaler_name(),
                    data_name=data_name,
                    name_not_scalable=self.name_not_scalable)
            else:
                # Normalize by the selected scaler in a regular way
                data_arr = normalize_data_by_scaler(data_in=self.dict_to_plot[data_name],
                                                    scaler=self.scaler_data,
                                                    data_name=data_name,
                                                    name_not_scalable=self.name_not_scalable)

            lowv = np.min(data_arr)
            highv = np.max(data_arr)
            self.range_dict[data_name] = {'low': lowv, 'low_default': lowv,
                                          'high': highv, 'high_default': highv}

    def reset_low_high(self, name):
        """Reset low and high value to default based on normalization.
        """
        self.range_dict[name]['low'] = self.range_dict[name]['low_default']
        self.range_dict[name]['high'] = self.range_dict[name]['high_default']
        self.limit_dict[name]['low'] = 0.0
        self.limit_dict[name]['high'] = 100.0
        self.update_img_wizard_items()
        self.show_image()

    def show_image(self):
        self.fig.clf()
        stat_temp = self.get_activated_num()
        stat_temp = OrderedDict(sorted(six.iteritems(stat_temp), key=lambda x: x[0]))

        # Check if positions data is available. Positions data may be unavailable
        # (not recorded in HDF5 file) if experiment is has not been completed.
        # While the data from the completed part of experiment may still be used,
        # plotting vs. x-y or scatter plot may not be displayed.
        positions_data_available = False
        if 'positions' in self.data_dict.keys():
            positions_data_available = True

        # Create local copies of self.pixel_or_pos, self.scatter_show and self.grid_interpolate
        pixel_or_pos_local = self.pixel_or_pos
        scatter_show_local = self.scatter_show
        grid_interpolate_local = self.grid_interpolate

        # Disable plotting vs x-y coordinates if 'positions' data is not available
        if not positions_data_available:
            if pixel_or_pos_local:
                pixel_or_pos_local = 0  # Switch to plotting vs. pixel number
                logger.error("'Positions' data is not available. Plotting vs. x-y coordinates is disabled")
            if scatter_show_local:
                scatter_show_local = False  # Switch to plotting vs. pixel number
                logger.error("'Positions' data is not available. Scatter plot is disabled.")
            if grid_interpolate_local:
                grid_interpolate_local = False  # Switch to plotting vs. pixel number
                logger.error("'Positions' data is not available. Interpolation is disabled.")

        low_lim = 1e-4  # define the low limit for log image
        plot_interp = 'Nearest'

        if self.scaler_data is not None:
            if np.count_nonzero(self.scaler_data) == 0:
                logger.warning('scaler is zero - scaling was not applied')
            elif len(self.scaler_data[self.scaler_data == 0]) > 0:
                logger.warning('scaler data has zero values')

        grey_use = self.color_opt

        ncol = int(np.ceil(np.sqrt(len(stat_temp))))
        try:
            nrow = int(np.ceil(len(stat_temp)/float(ncol)))
        except ZeroDivisionError:
            ncol = 1
            nrow = 1

        a_pad_v = 0.8
        a_pad_h = 0.5

        grid = ImageGrid(self.fig, 111,
                         nrows_ncols=(nrow, ncol),
                         axes_pad=(a_pad_v, a_pad_h),
                         cbar_location='right',
                         cbar_mode='each',
                         cbar_size='7%',
                         cbar_pad='2%',
                         share_all=True)

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
            applied to the 'extend' attribute of imshow(). The adjusted range is always greater
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

        for i, (k, v) in enumerate(six.iteritems(stat_temp)):

            quant_norm_applied = False
            if self.quantitative_normalization:
                # Quantitative normalization
                data_dict, quant_norm_applied = self.param_quant_analysis.apply_quantitative_normalization(
                    data_in=self.dict_to_plot[k],
                    scaler_dict=self.scaler_norm_dict,
                    scaler_name_default=self.get_selected_scaler_name(),
                    data_name=k,
                    name_not_scalable=self.name_not_scalable)
            else:
                # Normalize by the selected scaler in a regular way
                data_dict = normalize_data_by_scaler(data_in=self.dict_to_plot[k],
                                                     scaler=self.scaler_data,
                                                     data_name=k,
                                                     name_not_scalable=self.name_not_scalable)

            if pixel_or_pos_local or scatter_show_local:

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

                yd, xd = data_dict.shape

                xd_min, xd_max, yd_min, yd_max = 0, xd, 0, yd
                if (yd <= math.floor(xd / 100)) and (xd >= 200):
                    yd_min, yd_max = -math.floor(xd / 200), math.ceil(xd / 200)
                if (xd <= math.floor(yd / 100)) and (yd >= 200):
                    xd_min, xd_max = -math.floor(yd / 200), math.ceil(yd / 200)

                xd_axis_min, xd_axis_max, yd_axis_min, yd_axis_max = \
                    _compute_equal_axes_ranges(xd_min, xd_max, yd_min, yd_max)

            if self.scale_opt == 'Linear':

                low_ratio = self.limit_dict[k]['low']/100.0
                high_ratio = self.limit_dict[k]['high']/100.0
                if (self.scaler_data is None) and (not quant_norm_applied):
                    minv = self.range_dict[k]['low']
                    maxv = self.range_dict[k]['high']
                else:
                    # Unfortunately, the new normalization procedure requires to recalculate min and max values
                    minv = np.min(data_dict)
                    maxv = np.max(data_dict)
                low_limit = (maxv-minv)*low_ratio + minv
                high_limit = (maxv-minv)*high_ratio + minv

                # Set some minimum range for the colorbar (otherwise it will have white fill)
                if math.isclose(low_limit, high_limit, abs_tol=2e-20):
                    if abs(low_limit) < 1e-20:  # The value is zero
                        dv = 1e-20
                    else:
                        dv = math.fabs(low_limit * 0.01)
                    high_limit += dv
                    low_limit -= dv

                if not scatter_show_local:
                    if grid_interpolate_local:
                        data_dict, _, _ = grid_interpolate(data_dict,
                                                           self.data_dict['positions']['x_pos'],
                                                           self.data_dict['positions']['y_pos'])
                    im = grid[i].imshow(data_dict,
                                        cmap=grey_use,
                                        interpolation=plot_interp,
                                        extent=(xd_min, xd_max, yd_max, yd_min),
                                        origin='upper',
                                        clim=(low_limit, high_limit))
                    grid[i].set_ylim(yd_axis_max, yd_axis_min)
                else:
                    xx = self.data_dict['positions']['x_pos']
                    yy = self.data_dict['positions']['y_pos']

                    # The following condition prevents crash if different file is loaded while
                    #    the scatter plot is open (PyXRF specific issue)
                    if data_dict.shape == xx.shape and data_dict.shape == yy.shape:
                        im = grid[i].scatter(xx, yy, c=data_dict,
                                             marker='s', s=500,
                                             alpha=1.0,  # Originally: alpha=0.8
                                             cmap=grey_use,
                                             vmin=low_limit, vmax=high_limit,
                                             linewidths=1, linewidth=0)
                        grid[i].set_ylim(yd_axis_max, yd_axis_min)

                grid[i].set_xlim(xd_axis_min, xd_axis_max)

                grid_title = k
                if quant_norm_applied:
                    grid_title += " - Q"  # Mark the plots that represent quantitative information
                grid[i].text(0, 1.01, grid_title, ha='left', va='bottom', transform=grid[i].axes.transAxes)

                grid.cbar_axes[i].colorbar(im)
                im.colorbar.formatter = im.colorbar.cbar_axis.get_major_formatter()
                # im.colorbar.ax.get_xaxis().set_ticks([])
                # im.colorbar.ax.get_xaxis().set_ticks([], minor=True)
                grid.cbar_axes[i].ticklabel_format(style='sci', scilimits=(-3, 4), axis='both')

                #  Do not remove this code, may be useful in the future (Dmitri G.) !!!
                #  Print label for colorbar
                # cax = grid.cbar_axes[i]
                # axis = cax.axis[cax.orientation]
                # axis.label.set_text("$[a.u.]$")

            else:

                maxz = np.max(data_dict)
                # Set some reasonable minimum range for the colorbar
                #   Zeros or negative numbers will be shown in white
                if maxz <= 1e-30:
                    maxz = 1

                if not scatter_show_local:
                    if grid_interpolate_local:
                        data_dict, _, _ = grid_interpolate(data_dict,
                                                           self.data_dict['positions']['x_pos'],
                                                           self.data_dict['positions']['y_pos'])
                    im = grid[i].imshow(data_dict,
                                        norm=LogNorm(vmin=low_lim*maxz,
                                                     vmax=maxz, clip=True),
                                        cmap=grey_use,
                                        interpolation=plot_interp,
                                        extent=(xd_min, xd_max, yd_max, yd_min),
                                        origin='upper',
                                        clim=(low_lim*maxz, maxz))
                    grid[i].set_ylim(yd_axis_max, yd_axis_min)
                else:
                    im = grid[i].scatter(self.data_dict['positions']['x_pos'],
                                         self.data_dict['positions']['y_pos'],
                                         norm=LogNorm(vmin=low_lim*maxz,
                                                      vmax=maxz, clip=True),
                                         c=data_dict, marker='s', s=500, alpha=1.0,  # Originally: alpha=0.8
                                         cmap=grey_use,
                                         linewidths=1, linewidth=0)
                    grid[i].set_ylim(yd_axis_min, yd_axis_max)

                grid[i].set_xlim(xd_axis_min, xd_axis_max)

                grid_title = k
                if quant_norm_applied:
                    grid_title += " - Q"  # Mark the plots that represent quantitative information
                grid[i].text(0, 1.01, grid_title, ha='left', va='bottom', transform=grid[i].axes.transAxes)

                grid.cbar_axes[i].colorbar(im)
                im.colorbar.formatter = im.colorbar.cbar_axis.get_major_formatter()
                im.colorbar.ax.get_xaxis().set_ticks([])
                im.colorbar.ax.get_xaxis().set_ticks([], minor=True)
                im.colorbar.cbar_axis.set_minor_formatter(mticker.LogFormatter())

            grid[i].get_xaxis().set_major_locator(mticker.MaxNLocator(nbins="auto"))
            grid[i].get_yaxis().set_major_locator(mticker.MaxNLocator(nbins="auto"))

            grid[i].get_xaxis().get_major_formatter().set_useOffset(False)
            grid[i].get_yaxis().get_major_formatter().set_useOffset(False)

        self.fig.suptitle(self.img_title, fontsize=20)
        self.fig.canvas.draw_idle()

    def get_activated_num(self):
        """Collect the selected items for plotting.
        """
        current_items = {k: v for (k, v) in six.iteritems(self.stat_dict) if v is True}
        return current_items

    def record_selected(self):
        """Save the list of items in cache for later use.
        """
        self.items_previous_selected = [k for (k, v) in self.stat_dict.items() if v is True]
        logger.info('Items are set as default: {}'.format(self.items_previous_selected))
        self.data_dict['use_default_selection'] = {k: self.dict_to_plot[k] for k in self.items_previous_selected}
        self.data_dict_keys = list(self.data_dict.keys())
