from __future__ import absolute_import, division, print_function
import math
import numpy as np
import matplotlib
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.collections import BrokenBarHCollection
import matplotlib.ticker as mticker
from matplotlib.colors import LogNorm
from enum import Enum
from mpl_toolkits.axes_grid1 import ImageGrid

from atom.api import Atom, Str, observe, Typed, Int, List, Dict, Float, Bool

from skbeam.core.fitting.xrf_model import K_TRANSITIONS, L_TRANSITIONS, M_TRANSITIONS
from skbeam.fluorescence import XrfElement as Element

from ..core.xrf_utils import get_eline_parameters

import logging

logger = logging.getLogger(__name__)


def get_color_name():

    # usually line plot will not go beyond 10
    first_ten = [
        "indigo",
        "maroon",
        "green",
        "darkblue",
        "darkgoldenrod",
        "blue",
        "darkcyan",
        "sandybrown",
        "black",
        "darkolivegreen",
    ]

    # Avoid red color, as those color conflict with emission lines' color.
    nonred_list = [
        v
        for v in matplotlib.colors.cnames.keys()
        if "pink" not in v and "fire" not in v and "sage" not in v and "tomato" not in v and "red" not in v
    ]
    return first_ten + nonred_list + list(matplotlib.colors.cnames.keys())


class PlotTypes(Enum):
    LINLOG = 0
    LINEAR = 1


class EnergyRangePresets(Enum):
    SELECTED_RANGE = 0
    FULL_SPECTRUM = 1


class MapTypes(Enum):
    LINEAR = 0
    LOG = 1


class MapAxesUnits(Enum):
    PIXELS = 0
    POSITIONS = 1


class LinePlotModel(Atom):
    """
    This class performs all the required line plots.

    Attributes
    ----------
    data : array
        Experimental data
    _fit : class object
        Figure object from matplotlib
    _ax : class object
        Axis object from matplotlib
    _canvas : class object
        Canvas object from matplotlib
    element_id : int
        Index of element
    parameters : `atom.List`
        A list of `Parameter` objects, subclassed from the `Atom` base class.
        These `Parameter` objects hold all relevant xrf information
    elist : list
        Emission energy and intensity for given element
    plot_opt : int
        Linear or log plot
    total_y : dict
        Results for k lines
    total_l : dict
        Results for l and m lines
    prefit_x : array
        X axis with limited range
    plot_title : str
        Title for plotting
    fit_x : array
        x value for fitting
    fit_y : array
        fitted data
    plot_type_names : list
        linear or log plot
    max_v : float
        max value of data array
    incident_energy : float
        in KeV
    param_model : Typed(object)
        Reference to ParamModel object
    """

    # data = Typed(object)  # Typed(np.ndarray)
    exp_data_label = Str("experiment")

    number_pts_to_show = Int(3000)  # The number of spectrum point to show

    # -------------------------------------------------------------
    # Preview plot (raw experimental spectra)
    _fig_preview = Typed(Figure)
    _ax_preview = Typed(Axes)
    _lines_preview = List()
    _bahr_preview = Typed(BrokenBarHCollection)

    plot_type_preview = Typed(PlotTypes)
    energy_range_preview = Typed(EnergyRangePresets)

    min_v_preview = Float()
    max_v_preview = Float()
    min_e_preview = Float()
    max_e_preview = Float()

    # -----------------------------------------------------------
    # Preview of Total Count Maps
    _fig_maps = Typed(Figure)
    map_type_preview = Typed(MapTypes)
    map_axes_units_preview = Typed(MapAxesUnits)
    map_scatter_plot = Bool(False)

    map_preview_color_scheme = Str("viridis")
    map_preview_range_low = Float(-1)
    map_preview_range_high = Float(-1)
    # ------------------------------------------------------------

    _fig = Typed(Figure)
    _ax = Typed(Axes)
    _canvas = Typed(object)
    plot_fit_x_min = Float(0)  # The variables are used to store x_min and x_max for the current plot
    plot_fit_x_max = Float(0)
    element_id = Int(0)
    elist = List()
    scale_opt = Int(0)
    # total_y = Dict()
    # total_l = Dict()
    # total_m = Dict()
    # total_pileup = Dict()

    prefit_x = Typed(object)
    plot_title = Str()
    # fit_x = Typed(np.ndarray)
    # fit_y = Typed(np.ndarray)
    # residual = Typed(np.ndarray)

    plot_type_names = List()
    max_v = Float()
    incident_energy = Float(12.0)

    energy_range_names = List()
    energy_range_fitting = Str()

    eline_obj = List()

    plot_exp_opt = Bool(False)
    plot_exp_obj = Typed(Line2D)
    show_exp_opt = Bool(False)  # Flag: show spectrum preview

    # Reference to artist responsible for displaying the selected range of energies on the plot
    plot_energy_barh = Typed(BrokenBarHCollection)
    t_bar = Typed(object)

    plot_exp_list = List()

    auto_fit_obj = List()
    show_autofit_opt = Bool()

    plot_fit_obj = List()  # Typed(Line2D)
    show_fit_opt = Bool(False)
    # fit_all = Typed(object)

    plot_style = Dict()

    roi_plot_dict = Dict()
    roi_dict = Typed(object)  # OrderedDict()

    log_range = List()
    linear_range = List()
    plot_escape_line = Int(0)
    emission_line_window = Bool(True)
    det_materials = Int(0)
    escape_e = Float(1.73998)
    limit_cut = Int()
    # prefix_name_roi = Str()
    # element_for_roi = Str()
    # element_list_roi = List()
    # roi_dict = Typed(object) #OrderedDict()

    # img_dict = Dict()
    # roi_result = Dict()

    # Reference to ParamModel object
    param_model = Typed(object)
    # Reference to FileIOModel object
    io_model = Typed(object)

    # Location of the vertical (mouse-selected) marker on the plot.
    # Value is in kev. Negative value - no marker is placed.
    vertical_marker_kev = Float(-1)
    # Reference to the respective Matplotlib artist
    line_vertical_marker = Typed(object)
    vertical_marker_is_visible = Bool(False)

    report_marker_state = Typed(object)

    def __init__(self, *, param_model, io_model):

        # Reference to ParamModel object
        self.param_model = param_model
        self.io_model = io_model

        # self.data = None

        self._fig = plt.figure()

        self._ax = self._fig.add_subplot(111)
        try:
            self._ax.set_axis_bgcolor("lightgrey")
        except AttributeError:
            self._ax.set_facecolor("lightgrey")

        self._ax.set_xlabel("Energy (keV)")
        self._ax.set_ylabel("Spectrum (Counts)")
        self._ax.grid(which="both")
        self._ax.set_yscale("log")
        self.plot_type_names = ["LinLog", "Linear"]

        self.energy_range_names = ["selected", "full"]
        self.energy_range_fitting = "selected"

        self._ax.autoscale_view(tight=True)
        self._ax.legend(loc=2)

        self._color_config()
        self._fig.tight_layout(pad=0.5)
        self.max_v = 1.0
        # when we calculate max value, data smaller than 500, 0.5 Kev, can be ignored.
        # And the last point of data is also huge, and should be cut off.
        self.limit_cut = 100
        # self._ax.margins(x=0.0, y=0.10)

        # --------------------------------------------------------------
        # Spectrum preview figure
        self._fig_preview = Figure()
        self.plot_type_preview = PlotTypes.LINLOG
        self.energy_range_preview = EnergyRangePresets.SELECTED_RANGE

        # --------------------------------------------------------------
        # Preview of Total Count Maps
        self._fig_maps = Figure()
        self.map_type_preview = MapTypes.LINEAR
        self.map_axes_units_preview = MapAxesUnits.PIXELS

    def _color_config(self):
        self.plot_style = {
            "experiment": {"color": "blue", "linestyle": "", "marker": ".", "label": self.exp_data_label},
            "background": {"color": "indigo", "marker": "+", "markersize": 1, "label": "background"},
            "emission_line": {"color": "black", "linewidth": 2},
            "roi_line": {"color": "red", "linewidth": 2},
            "k_line": {"color": "green", "label": "k lines"},
            "l_line": {"color": "magenta", "label": "l lines"},
            "m_line": {"color": "brown", "label": "m lines"},
            "compton": {"color": "darkcyan", "linewidth": 1.5, "label": "compton"},
            "elastic": {"color": "purple", "label": "elastic"},
            "escape": {"color": "darkblue", "label": "escape"},
            "pileup": {"color": "darkgoldenrod", "label": "pileup"},
            "userpeak": {"color": "orange", "label": "userpeak"},
            # 'auto_fit': {'color': 'black', 'label': 'auto fitted', 'linewidth': 2.5},
            "fit": {"color": "red", "label": "fit", "linewidth": 2.5},
            "residual": {"color": "black", "label": "residual", "linewidth": 2.0},
        }

    def plot_exp_data_update(self, change):
        """
        Observer function to be connected to the fileio model
        in the top-level gui.py startup

        Parameters
        ----------
        changed : dict
            This is the dictionary that gets passed to a function
            with the @observe decorator
        """
        self.plot_exp_opt = False  # exp data for fitting
        self.show_exp_opt = False  # all exp data from different channels
        self.show_fit_opt = False

        # Reset currently selected element_id (mostly to reset GUI elements)
        self.element_id = 0

    def init_mouse_event(self):
        """Set up callback for mouse button-press event"""
        # Reference to the toolbar
        self.t_bar = self._fig.canvas.toolbar
        # Set callback for Button Press event
        self._fig.canvas.mpl_connect("button_press_event", self.canvas_onpress)

    def _update_canvas(self):
        # It may be sufficient to initialize the event only once, but at this point
        #   it seems to be the most reliable option. May be changed in the future.
        self.init_mouse_event()
        self.plot_vertical_marker()

        self._ax.legend(loc=2)
        try:
            self._ax.legend(framealpha=0.2).set_draggable(True)
        except AttributeError:
            self._ax.legend(framealpha=0.2)
        self._fig.tight_layout(pad=0.5)
        # self._ax.margins(x=0.0, y=0.10)

        # when we click the home button on matplotlib gui,
        # relim will remember the previously defined x range
        self._ax.relim(visible_only=True)
        self._fig.canvas.draw()

    def _update_ylimit(self):
        # manually define y limit, from experience
        self.log_range = [self.max_v * 1e-5, self.max_v * 2]
        # self.linear_range = [-0.3*self.max_v, self.max_v*1.2]
        self.linear_range = [0, self.max_v * 1.2]

    def exp_label_update(self, change):
        """
        Observer function to be connected to the fileio model
        in the top-level gui.py startup

        Parameters
        ----------
        changed : dict
            This is the dictionary that gets passed to a function
            with the @observe decorator
        """
        self.exp_data_label = change["value"]
        self.plot_style["experiment"]["label"] = change["value"]

    # @observe('exp_data_label')
    # def _change_exp_label(self, change):
    #     if change['type'] == 'create':
    #         return
    #     self.plot_style['experiment']['label'] = change['value']

    @observe("parameters")
    def _update_energy(self, change):
        if "coherent_sct_energy" not in self.param_model.param_new:
            return
        self.incident_energy = self.param_model.param_new["coherent_sct_energy"]["value"]

    def set_energy_range_fitting(self, energy_range_name):
        if energy_range_name not in self.energy_range_names:
            raise ValueError(
                f"Unknown energy range name {energy_range_name}. Allowed names: {self.energy_range_names}"
            )
        self.energy_range_fitting = energy_range_name
        self.plot_experiment()

    def set_incident_energy(self, change):
        """
        The observer function that changes the value of incident energy
        and upper bound for fitted energy range. Should not be called directly.

        Parameters
        ----------

        change : dict
            ``change["value"]`` is the new value of incident energy
        """
        self.change_incident_energy(change["value"])

    def change_incident_energy(self, energy_new):
        """
        The function that perfroms the changes the value of incident energy
        and upper bound for fitted energy range.

        Parameters
        ----------

        incident_energy : float
            New value of incident energy
        """

        margin = 0.8  # Value by which the upper bound of the range used for fitting
        #               exceeds the incident energy. Selected for convenience, but
        #               is subject to change. This is the place to change it to take effect
        #               throughout the program.

        # Limit the number of decimal points for better visual presentation
        energy_new = round(energy_new, ndigits=6)
        # Change the value twice to ensure that all observer functions are called
        self.incident_energy = energy_new + 1.0  # Arbitrary number different from 'energy_new'
        self.incident_energy = energy_new
        if "coherent_sct_energy" in self.param_model.param_new:
            self.param_model.param_new["coherent_sct_energy"]["value"] = energy_new

        # Change the value twice to ensure that all observer functions are called
        self.param_model.energy_bound_high_buf = energy_new + 1.8  # Arbitrary number
        upper_bound = energy_new + margin
        # Limit the number of decimal points for better visual presentation
        upper_bound = round(upper_bound, ndigits=5)
        self.param_model.energy_bound_high_buf = upper_bound

    @observe("scale_opt")
    def _new_opt(self, change):
        self.log_linear_plot()
        self._update_canvas()

    def energy_bound_high_update(self, change):
        """Observer function for 'param_model.energy_bound_high_buf'"""
        if self.io_model.data is None:
            return
        self.exp_data_update({"value": self.io_model.data})
        self.plot_selected_energy_range_original(e_high=change["value"])
        self.plot_vertical_marker(e_high=change["value"])
        self._update_canvas()

    def energy_bound_low_update(self, change):
        """Observer function for 'param_model.energy_bound_low_buf'"""
        if self.io_model.data is None:
            return
        self.exp_data_update({"value": self.io_model.data})
        self.plot_selected_energy_range_original(e_low=change["value"])
        self.plot_vertical_marker(e_low=change["value"])
        self._update_canvas()

    def log_linear_plot(self):
        if self.plot_type_names[self.scale_opt] == "LinLog":
            self._ax.set_yscale("log")
            # self._ax.margins(x=0.0, y=0.5)
            # self._ax.autoscale_view(tight=True)
            # self._ax.relim(visible_only=True)
            self._ax.set_ylim(self.log_range)

        else:
            self._ax.set_yscale("linear")
            # self._ax.margins(x=0.0, y=0.10)
            # self._ax.autoscale_view(tight=True)
            # self._ax.relim(visible_only=True)
            self._ax.set_ylim(self.linear_range)

    def exp_data_update(self, change):
        """
        Observer function to be connected to the fileio model
        in the top-level gui.py startup

        Parameters
        ----------
        changed : dict
            This is the dictionary that gets passed to a function
            with the @observe decorator
        """
        # TODO: This function does not change the data. Instead it is expected to
        #   perform a number of operation when data is changed.

        # self.data = change['value']
        if self.io_model.data is None:
            return

        e_range = self.energy_range_fitting
        e_range_full, e_range_selected = "full", "selected"
        if set([e_range_full, e_range_selected]) < set(self.energy_range_names):
            raise ValueError(
                f"Some names for energy range {(e_range_full, e_range_selected)} are not supported. "
                "Please report the error to the development team."
            )
        if e_range not in (e_range_full, e_range_selected):
            logger.error(
                f"Spectrum preview: Unknown option for the energy range: {e_range}\n"
                "Please report the error to the development team."
            )
            # This is not a critical error, so we still can proceed
            e_range = e_range_full

        if not self.param_model.param_new:
            return

        # The number of points in the displayed dataset
        n_dset_points = len(self.io_model.data)

        if e_range == e_range_selected:
            n_range_low, n_range_high = self.selected_range_indices(n_indexes=n_dset_points)
        else:
            n_range_low, n_range_high = 0, n_dset_points

        n_low = int(np.clip(n_range_low, a_min=0, a_max=n_dset_points - 1))
        n_high = int(np.clip(n_range_high, a_min=1, a_max=n_dset_points))

        # Find the maximum value (skip the first and last 'limit_cut' points of the dataset
        n1, n2 = max(self.limit_cut, n_low), min(n_dset_points - self.limit_cut, n_high)
        if n2 <= n1:  # This is just a precaution: it is expected that n_dset_points >> 2 * limit_cut
            n1, n2 = n_low, n_high
        self.max_v = float(np.max(self.io_model.data[n1:n2]))

        try:
            self.plot_exp_obj.remove()
            logger.debug("Previous experimental data is removed.")
        except AttributeError:
            logger.debug("No need to remove experimental data.")

        data_arr = self.io_model.data
        x_v = (
            self.param_model.param_new["e_offset"]["value"]
            + np.arange(n_low, n_high) * self.param_model.param_new["e_linear"]["value"]
            + np.arange(n_low, n_high) ** 2 * self.param_model.param_new["e_quadratic"]["value"]
        )

        data_arr = data_arr[n_low:n_high]

        (self.plot_exp_obj,) = self._ax.plot(
            x_v,
            data_arr,
            linestyle=self.plot_style["experiment"]["linestyle"],
            color=self.plot_style["experiment"]["color"],
            marker=self.plot_style["experiment"]["marker"],
            label=self.plot_style["experiment"]["label"],
        )

        # Rescale the plot along x-axis if needed
        x_min, x_max = x_v[0], x_v[-1]
        if (x_min != self.plot_fit_x_min) or (x_max != self.plot_fit_x_max):
            self.plot_fit_x_min = x_min
            self.plot_fit_x_max = x_max
            self._ax.set_xlim(x_min, x_max)

        self._update_ylimit()
        self.log_linear_plot()

        self._set_eline_select_controls()

        self.plot_selected_energy_range_original()

        # _show_hide_exp_plot is called to show or hide current plot based
        #           on the state of _show_exp_opt flag
        self._show_hide_exp_plot(self.show_exp_opt or self.plot_exp_opt)

    def _show_hide_exp_plot(self, plot_show):

        if self.io_model.data is None:
            return

        try:
            if plot_show:
                self.plot_exp_obj.set_visible(True)
                lab = self.plot_exp_obj.get_label()
                self.plot_exp_obj.set_label(lab.strip("_"))
            else:
                self.plot_exp_obj.set_visible(False)
                lab = self.plot_exp_obj.get_label()
                self.plot_exp_obj.set_label("_" + lab)

            self._update_canvas()
        except Exception:
            pass

    @observe("plot_exp_opt")
    def _new_exp_plot_opt(self, change):

        if self.io_model.data is None:
            return

        if change["type"] != "create":
            if change["value"]:
                self.plot_experiment()
            # _show_hide_exp_plot is already called inside 'plot_experiment()',
            #    but visibility flag was not used correctly. So we need to
            #    call it again.
            self._show_hide_exp_plot(change["value"])
            self._set_eline_select_controls()

    # @observe('show_exp_opt')
    # def _update_exp(self, change):
    #     if change['type'] != 'create':
    #         if change['value']:
    #             if len(self.plot_exp_list):
    #                 for v in self.plot_exp_list:
    #                     v.set_visible(True)
    #                     lab = v.get_label()
    #                     if lab != '_nolegend_':
    #                         v.set_label(lab.strip('_'))
    #         else:
    #             if len(self.plot_exp_list):
    #                 for v in self.plot_exp_list:
    #                     v.set_visible(False)
    #                     lab = v.get_label()
    #                     if lab != '_nolegend_':
    #                         v.set_label('_' + lab)
    #         self._update_canvas()

    @observe("show_fit_opt")
    def _update_fit(self, change):
        if change["type"] != "create":
            self.update_fit_plots(visible=bool(change["value"]))

    def update_fit_plots(self, visible=True):
        for v in self.plot_fit_obj:
            v.set_visible(visible)
            lab = v.get_label()
            if lab != "_nolegend_":
                v.set_label(lab.strip("_"))
        self._update_canvas()

    def plot_experiment(self):
        """
        PLot raw experiment data for fitting.
        """

        # Do nothing if no data is loaded
        if self.io_model.data is None:
            return

        data_arr = np.asarray(self.io_model.data)
        self.exp_data_update({"value": data_arr})

    def plot_vertical_marker(self, *, e_low=None, e_high=None):

        # It doesn't seem necessary to force the marker inside the selected range.
        #   It may be used for purposes that require to set it outside the range
        # self._vertical_marker_set_inside_range(e_low=e_low, e_high=e_high)

        x_v = (self.vertical_marker_kev, self.vertical_marker_kev)
        y_v = (-1e30, 1e30)  # This will cover the range of possible values of accumulated counts

        if self.line_vertical_marker:
            self._ax.lines.remove(self.line_vertical_marker)
            self.line_vertical_marker = None
        if self.vertical_marker_is_visible:
            (self.line_vertical_marker,) = self._ax.plot(x_v, y_v, color="blue")

    def set_plot_vertical_marker(self, marker_position=None, mouse_clicked=False):
        """
        The function is called when setting the position of the marker interactively

        If the parameter `marker_position` is `None`, then don't set or change the value.
        Just make the marker visible.
        """

        # Ignore the new value if it is outside the range of selected energies.
        # If 'marker_position' is None, then show the marker at its current location.
        # Totally ignore clicks if 'marker_position' is outside the range (but still
        # display the marker if 'mouse_clicked' is False.
        marker_in_range = True
        if marker_position is not None:
            e_low = self.param_model.param_new["non_fitting_values"]["energy_bound_low"]["value"]
            e_high = self.param_model.param_new["non_fitting_values"]["energy_bound_high"]["value"]
            if e_low <= marker_position <= e_high or not mouse_clicked:
                # If the function was called to display marker (e.g. for existing peak) outside
                #   the selected range, then show it. If button was clicked, then ignore it.
                self.vertical_marker_kev = marker_position
            else:
                marker_in_range = False

        if marker_in_range:
            # Make the marker visible
            self.vertical_marker_is_visible = True

            # Compute peak intensity. The displayed value will change only for user defined peak,
            #   since it is moved to the position of the marker.
            self.compute_manual_peak_intensity()

            # Update the location of the marker and the canvas
            self.plot_vertical_marker()
            self._update_canvas()

            if mouse_clicked:
                try:
                    self.report_marker_state(True)  # This is an externally set callback function
                except Exception:
                    pass

    def hide_plot_vertical_marker(self, mouse_clicked=False):
        """Hide vertical marker"""
        self.vertical_marker_is_visible = False
        self.plot_vertical_marker()
        self._update_canvas()

        if mouse_clicked:
            try:
                self.report_marker_state(False)  # This is an externally set callback function
            except Exception:
                pass

    def plot_selected_energy_range_original(self, *, e_low=None, e_high=None):
        """
        Plot the range of energies selected for processing. The range may be optionally
        provided as arguments. The range values that are not provided, are read from
        globally accessible dictionary of parameters. The values passed as arguments
        are mainly used if the function is called during interactive update of the
        range, when the order of update is undetermined and the parameter dictionary
        may be updated after the function is called.
        """
        # The range of energy selected for analysis
        if e_low is None:
            e_low = self.param_model.param_new["non_fitting_values"]["energy_bound_low"]["value"]
        if e_high is None:
            e_high = self.param_model.param_new["non_fitting_values"]["energy_bound_high"]["value"]

        n_x = 4096  # Set to the maximum possible number of points

        # Generate the values for 'energy' axis
        x_v = (
            self.param_model.param_new["e_offset"]["value"]
            + np.arange(n_x) * self.param_model.param_new["e_linear"]["value"]
            + np.arange(n_x) ** 2 * self.param_model.param_new["e_quadratic"]["value"]
        )

        ss = (x_v < e_high) & (x_v > e_low)
        y_min, y_max = -1e30, 1e30  # Select the max and min values for plotted rectangles

        # Remove the plot if it exists
        if self.plot_energy_barh in self._ax.collections:
            self._ax.collections.remove(self.plot_energy_barh)

        # Create the new plot (based on new parameters if necessary
        self.plot_energy_barh = BrokenBarHCollection.span_where(
            x_v, ymin=y_min, ymax=y_max, where=ss, facecolor="white", edgecolor="yellow", alpha=1
        )
        self._ax.add_collection(self.plot_energy_barh)

    def plot_multi_exp_data(self):
        while len(self.plot_exp_list):
            self.plot_exp_list.pop().remove()

        color_n = get_color_name()

        self.max_v = 1.0
        m = 0
        for (k, v) in self.io_model.data_sets.items():
            if v.selected_for_preview:

                data_arr = np.asarray(v.data)
                # Truncate the array (1D spectrum)
                data_arr = data_arr[0 : self.number_pts_to_show]
                self.max_v = np.max([self.max_v, np.max(data_arr[self.limit_cut : -self.limit_cut])])

                x_v = (
                    self.param_model.param_new["e_offset"]["value"]
                    + np.arange(len(data_arr)) * self.param_model.param_new["e_linear"]["value"]
                    + np.arange(len(data_arr)) ** 2 * self.param_model.param_new["e_quadratic"]["value"]
                )

                (plot_exp_obj,) = self._ax.plot(
                    x_v,
                    data_arr,
                    color=color_n[m],
                    label=v.filename.split(".")[0],
                    linestyle=self.plot_style["experiment"]["linestyle"],
                    marker=self.plot_style["experiment"]["marker"],
                )
                self.plot_exp_list.append(plot_exp_obj)
                m += 1

        self.plot_selected_energy_range_original()

        self._update_ylimit()
        self.log_linear_plot()
        self._update_canvas()

    def plot_emission_line(self):
        """
        Plot emission line and escape peaks associated with given lines.
        The value of self.max_v is needed in this function in order to plot
        the relative height of each emission line.
        """
        while len(self.eline_obj):
            self.eline_obj.pop().remove()

        escape_e = self.escape_e

        if len(self.elist):
            for i in range(len(self.elist)):
                (eline,) = self._ax.plot(
                    [self.elist[i][0], self.elist[i][0]],
                    [0, self.elist[i][1] * self.max_v],
                    color=self.plot_style["emission_line"]["color"],
                    linewidth=self.plot_style["emission_line"]["linewidth"],
                )
                self.eline_obj.append(eline)
                if self.plot_escape_line and self.elist[i][0] > escape_e:
                    (eline,) = self._ax.plot(
                        [self.elist[i][0] - escape_e, self.elist[i][0] - escape_e],
                        [0, self.elist[i][1] * self.max_v],
                        color=self.plot_style["escape"]["color"],
                        linewidth=self.plot_style["emission_line"]["linewidth"],
                    )
                    self.eline_obj.append(eline)

    def _set_eline_select_controls(self, *, element_id=None, data="use_self_data"):
        if element_id is None:
            element_id = self.element_id
        if data == "use_self_data":
            data = self.io_model.data

    def is_line_in_selected_list(self, n_id):
        """
        Checks if the line with ID ``n_id`` is in the list of
        selected element lines.
        Used to enable/disable 'Add Line' and 'Remove Line' buttons.

        Parameters
        ----------

        n_id : Int
            index of the element emission line in the list
            (often equal to ``self.element_id``)

        Returns True if the element line
        is in the list of selected lines. False otherwise.
        """

        ename = self.get_element_line_name_by_id(n_id)

        if ename is None:
            return False

        if self.param_model.EC.is_element_in_list(ename):
            return True
        else:
            return False

    def is_element_line_id_valid(self, n_id):
        """
        Checks if ID (``n_id``) of the element emission line is valid,
        i.e. the name of the line may be obtained by using the ID.

        Parameters
        ----------

        n_id : Int
            index of the element emission line in the list
            (often equal to 'self.element_id')

        Returns True if the element line is valid
        """

        # There may be a more efficient way to check 'n_id',
        #   but we want to use the same function as we use
        #   to retrive the line name
        ename = self.get_element_line_name_by_id(n_id)

        if ename is None:
            return False
        else:
            return True

    def get_element_line_name_by_id(self, n_id):
        """
        Retrieves the name of the element emission line from its ID
        (the number in the list). The lines are numbered starting with 1.
        If the ID is invalid, the function returns None.

        Parameters
        ----------

        n_id : int
            index of the element emission line in the list
            (often equal to 'self.element_id')

        Returns the line name (str). If the name can not be retrieved, then
        the function returns None.
        """
        if n_id < 1:
            # Elements are numbered starting with 1. Element #0 does not exist.
            #   (Element #0 means that no element is selected)
            return None

        # This is the fixed list of element emission line names.
        #   The element with ID==1 is found in total_list[0]
        total_list = self.param_model.get_user_peak_list()
        try:
            ename = total_list[n_id - 1]
        except Exception:
            ename = None
        return ename

    def _vertical_marker_set_inside_range(self, *, e_low=None, e_high=None):
        """
        Don't move the marker if it is inside range. If it is outside range,
        then set the marker to the center of the range
        """
        # The range of energy selected for analysis
        if e_low is None:
            e_low = self.param_model.param_new["non_fitting_values"]["energy_bound_low"]["value"]
        if e_high is None:
            e_high = self.param_model.param_new["non_fitting_values"]["energy_bound_high"]["value"]

        # By default, place the marker in the middle of the range if its original position
        #   is outside the range
        if (self.vertical_marker_kev > e_high) or (self.vertical_marker_kev < e_low):
            self.vertical_marker_kev = (e_low + e_high) / 2.0

    def _fill_elist(self):

        _elist = []

        incident_energy = self.incident_energy
        k_len = len(K_TRANSITIONS)
        l_len = len(L_TRANSITIONS)
        m_len = len(M_TRANSITIONS)

        ename = self.get_element_line_name_by_id(self.element_id)

        if ename is not None:

            _elist = []
            if ename.lower().startswith("userpeak"):
                # Make sure that the marker is in the selected range of energies
                self._vertical_marker_set_inside_range()
                # The tuple structure: (center_energy, ratio)
                _elist.append((self.vertical_marker_kev, 1.0))

            elif "_K" in ename:
                e = Element(ename[:-2])
                if e.cs(incident_energy)["ka1"] != 0:
                    for i in range(k_len):
                        _elist.append(
                            (
                                e.emission_line.all[i][1],
                                e.cs(incident_energy).all[i][1] / e.cs(incident_energy).all[0][1],
                            )
                        )

            elif "_L" in ename:
                e = Element(ename[:-2])
                if e.cs(incident_energy)["la1"] != 0:
                    for i in range(k_len, k_len + l_len):
                        _elist.append(
                            (
                                e.emission_line.all[i][1],
                                e.cs(incident_energy).all[i][1] / e.cs(incident_energy).all[k_len][1],
                            )
                        )

            else:
                e = Element(ename[:-2])
                if e.cs(incident_energy)["ma1"] != 0:
                    for i in range(k_len + l_len, k_len + l_len + m_len):
                        _elist.append(
                            (
                                e.emission_line.all[i][1],
                                e.cs(incident_energy).all[i][1] / e.cs(incident_energy).all[k_len + l_len][1],
                            )
                        )

        return _elist

    def _get_pileup_lines(self, eline):
        """
        Returns the energy (center) of pileup peak. And the energies of two components.

        Parameters
        ----------
        eline: str
            Name of the pileup peak, e.g. V_Ka1-Co_Ka1

        Returns
        -------
        list(float)
            Energy in keV of pileup peak and two components
        """
        try:
            element_line1, element_line2 = eline.split("-")
            e1_cen = get_eline_parameters(element_line1, self.incident_energy)["energy"]
            e2_cen = get_eline_parameters(element_line2, self.incident_energy)["energy"]
            en = [e1_cen + e2_cen, e1_cen, e2_cen]
        except Exception:
            en = []
        return en

    def _fill_elist_pileup(self, eline=None):
        if eline is None:
            eline = self.param_model.e_name

        elist = []
        energies = self._get_pileup_lines(eline)
        if energies:
            elist = list(zip(energies, [1, 0.2, 0.2]))

        return elist

    def _fill_elist_userpeak(self):
        """
        Fill the list of 'emission lines' for user defined peak. There is only ONE
        'emission line', with position determined by the location of the marker.
        If the marker is not currently visible, then don't put any emission lines in the list.
        The list is used during adding user-defined peaks.
        """
        elist = []
        energy, marker_visible = self.get_suggested_new_manual_peak_energy()
        if marker_visible:
            elist.append((energy, 1))
        return elist

    def _reset_eline_plot(self):
        while len(self.eline_obj):
            self.eline_obj.pop().remove()
        self.elist = []
        self._fig.canvas.draw()

    @observe("element_id")
    def set_element(self, change):

        self._set_eline_select_controls(element_id=change["value"])
        self.compute_manual_peak_intensity(n_id=change["value"])

        if change["value"] == 0:
            self._reset_eline_plot()
            return

        self.plot_current_eline()

    def plot_current_eline(self, eline=None):
        """
        Plots emission lines for the selected peak based on 'self.element_id` and provided `eline`.
        """
        if eline is None:
            eline = self.param_model.e_name

        incident_energy = self.incident_energy

        # Name of the emission line (if emission line is selected)
        ename = self.get_element_line_name_by_id(self.element_id)
        # Check if pileup peak is selected
        is_pileup = self.param_model.get_eline_name_category(eline) == "pileup"

        if (ename is not None) or is_pileup:

            logger.debug(
                "Plot emission line for element: "
                "{} with incident energy {}".format(self.element_id, incident_energy)
            )

            if ename is not None:
                self.elist = self._fill_elist()
            elif is_pileup:
                self.elist = self._fill_elist_pileup(eline)
            else:
                self.elist = []  # Just in case
            self.plot_emission_line()
            self._update_canvas()

            # Do it the second time, since the 'self.elist' has changed
            self.compute_manual_peak_intensity(n_id=self.element_id)

        else:
            self._reset_eline_plot()
            logger.debug(f"Selected emission line with ID #{self.element_id} is not in the list.")

    @observe("det_materials")
    def _update_det_materials(self, change):
        if change["value"] == 0:
            self.escape_e = 1.73998
        else:
            self.escape_e = 9.88640

    def change_escape_peak_settings(self, plot_escape_line, det_material):
        self.plot_escape_line = plot_escape_line
        self.det_materials = det_material
        # Now update the displayed emission line
        self.plot_emission_line()
        self._update_canvas()

    def plot_roi_bound(self):
        """
        Plot roi with low, high and ceter value.
        """
        for k, v in self.roi_plot_dict.items():
            for data in v:
                data.remove()
        self.roi_plot_dict.clear()

        if len(self.roi_dict):
            # self._ax.hold(True)
            for k, v in self.roi_dict.items():
                temp_list = []
                for linev in np.array([v.left_val, v.line_val, v.right_val]) / 1000.0:
                    (lineplot,) = self._ax.plot(
                        [linev, linev],
                        [0, 1 * self.max_v],
                        color=self.plot_style["roi_line"]["color"],
                        linewidth=self.plot_style["roi_line"]["linewidth"],
                    )
                    if v.show_plot:
                        lineplot.set_visible(True)
                    else:
                        lineplot.set_visible(False)
                    temp_list.append(lineplot)
                self.roi_plot_dict.update({k: temp_list})

        self._update_canvas()

    @observe("roi_dict")
    def show_roi_bound(self, change):
        logger.debug("roi dict changed {}".format(change["value"]))
        self.plot_roi_bound()

        if len(self.roi_dict):
            for k, v in self.roi_dict.items():
                if v.show_plot:
                    for ln in self.roi_plot_dict[k]:
                        ln.set_visible(True)
                else:
                    for ln in self.roi_plot_dict[k]:
                        ln.set_visible(False)
        self._update_canvas()

    def get_suggested_new_manual_peak_energy(self):
        """
        Returns energy pointed by the vertical marker in keV and the status of the marker.

        Returns
        -------
        float
            Energy of the manual peak center in keV. The energy is determined
            by vertical marker on the screen.
        bool
            True if the vertical marker is visible, otherwise False.
        """
        energy = self.vertical_marker_kev
        marker_visible = self.vertical_marker_is_visible
        return energy, marker_visible

    def _compute_intensity(self, elist):
        # Some default value
        intensity = 1000.0

        if (
            self.io_model.data is not None
            and self.param_model.param_new is not None
            and self.param_model.prefit_x is not None
            and self.param_model.total_y is not None
            and len(self.io_model.data) > 1
            and len(self.param_model.prefit_x) > 1
        ):

            # Range of energies in fitting results
            e_fit_min = self.param_model.prefit_x[0]
            e_fit_max = self.param_model.prefit_x[-1]
            de_fit = (e_fit_max - e_fit_min) / (len(self.param_model.prefit_x) - 1)

            e_raw_min = self.param_model.param_new["e_offset"]["value"]
            e_raw_max = (
                self.param_model.param_new["e_offset"]["value"]
                + (len(self.io_model.data) - 1) * self.param_model.param_new["e_linear"]["value"]
                + (len(self.io_model.data) - 1) ** 2 * self.param_model.param_new["e_quadratic"]["value"]
            )

            de_raw = (e_raw_max - e_raw_min) / (len(self.io_model.data) - 1)

            # Note: the above algorithm for finding 'de_raw' is far from perfect but will
            #    work for now. As a result 'de_fit' and
            #    'de_raw' == sself.param_model.param_new['e_linear']['value'].
            #    So the quadratic coefficent is ignored. This is OK, since currently
            #    quadratic coefficient is always ZERO. When the program is rewritten,
            #    the complete algorithm should be revised.

            # Find the line with maximum energy. It must come first in the list,
            #    but let's check just to make sure
            max_line_energy, max_line_intensity = 0, 0
            if elist:
                for e, i in elist:
                    # e - line peak energy
                    # i - peak intensity relative to maximum peak
                    if e >= e_fit_min and e <= e_fit_max and e > e_raw_min and e < e_raw_max:
                        if max_line_intensity < i:
                            max_line_energy, max_line_intensity = e, i

            # Find the index of peak maximum in the 'fitted' data array
            n = (max_line_energy - e_fit_min) / de_fit
            n = np.clip(n, 0, len(self.param_model.total_y) - 1)
            n_fit = int(round(n))
            # Find the index of peak maximum in the 'raw' data array
            n = (max_line_energy - e_raw_min) / de_raw
            n = np.clip(n, 0, len(self.io_model.data) - 1)
            n_raw = int(round(n))
            # Intensity of the fitted data at the peak
            in_fit = self.param_model.total_y[n_fit]
            # Intensity of the raw data at the peak
            in_raw = self.io_model.data[n_raw]
            # The estimated peak intensity is the difference:
            intensity = in_raw - in_fit

            # The following step is questionable. We assign some reasonably small number.
            #   The desired value can always be manually entered
            if intensity < 0.0:
                intensity = abs(in_raw / 100)

        return intensity

    def compute_manual_peak_intensity(self, n_id=None):

        if n_id is None:
            n_id = self.element_id

        # Check if the emission line is in the list of supported emission lines (e.g. Ca_K)
        if not self.is_element_line_id_valid(n_id):
            # This is not a supported emission line (n_id==0)
            # This means we are probably dealing with user defined peak.
            if self.is_line_in_selected_list(n_id):
                # Display intensity if the peak is in the list.
                name = self.get_element_line_name_by_id(n_id)
                intensity = self.param_model.EC.element_dict[name].maxv
            else:
                elist = self._fill_elist_userpeak()
                intensity = self._compute_intensity(elist)
        else:
            if self.is_line_in_selected_list(n_id):
                # Display intensity if the peak is in the list.
                name = self.get_element_line_name_by_id(n_id)
                intensity = self.param_model.EC.element_dict[name].maxv
            else:
                # This is a new peak
                elist = self._fill_elist()
                intensity = self._compute_intensity(elist)

        # Round the intensity for nicer printing
        self.param_model.add_element_intensity = round(intensity, 2)

    def plot_fit(self, fit_x, fit_y, fit_all, residual=None):
        """
        Parameters
        ----------
        fit_x : array
            energy axis
        fit_y : array
            fitted spectrum
        fit_all : dict
            dict of individual line
        residual : array
            residual between fit and exp
        """
        if fit_x is None or fit_y is None:
            return

        while len(self.plot_fit_obj):
            self.plot_fit_obj.pop().remove()

        (ln,) = self._ax.plot(
            fit_x,
            fit_y,
            color=self.plot_style["fit"]["color"],
            label=self.plot_style["fit"]["label"],
            linewidth=self.plot_style["fit"]["linewidth"],
        )
        self.plot_fit_obj.append(ln)

        if residual is not None:
            # shiftv = 1.5  # move residual down by some amount
            (ln,) = self._ax.plot(
                fit_x,
                residual - 0.15 * self.max_v,  # shiftv*(np.max(np.abs(self.residual))),
                label=self.plot_style["residual"]["label"],
                color=self.plot_style["residual"]["color"],
            )
            self.plot_fit_obj.append(ln)

        k_num = 0
        l_num = 0
        m_num = 0
        p_num = 0
        for k, v in fit_all.items():
            if k == "background":
                (ln,) = self._ax.plot(
                    fit_x,
                    v,
                    color=self.plot_style["background"]["color"],
                    # marker=self.plot_style['background']['marker'],
                    # markersize=self.plot_style['background']['markersize'],
                    label=self.plot_style["background"]["label"],
                )
                self.plot_fit_obj.append(ln)
            elif k == "compton":
                (ln,) = self._ax.plot(
                    fit_x,
                    v,
                    color=self.plot_style["compton"]["color"],
                    linewidth=self.plot_style["compton"]["linewidth"],
                    label=self.plot_style["compton"]["label"],
                )
                self.plot_fit_obj.append(ln)
            elif k == "elastic":
                (ln,) = self._ax.plot(
                    fit_x, v, color=self.plot_style["elastic"]["color"], label=self.plot_style["elastic"]["label"]
                )
                self.plot_fit_obj.append(ln)
            elif k == "escape":
                (ln,) = self._ax.plot(
                    fit_x, v, color=self.plot_style["escape"]["color"], label=self.plot_style["escape"]["label"]
                )
                self.plot_fit_obj.append(ln)

            elif "user" in k.lower():
                (ln,) = self._ax.plot(
                    fit_x,
                    v,
                    color=self.plot_style["userpeak"]["color"],
                    label=self.plot_style["userpeak"]["label"],
                )
                self.plot_fit_obj.append(ln)

            elif "-" in k:  # Si_K-Si_K
                if p_num == 0:
                    (ln,) = self._ax.plot(
                        fit_x,
                        v,
                        color=self.plot_style["pileup"]["color"],
                        label=self.plot_style["pileup"]["label"],
                    )
                else:
                    (ln,) = self._ax.plot(fit_x, v, color=self.plot_style["pileup"]["color"], label="_nolegend_")
                self.plot_fit_obj.append(ln)
                p_num += 1

            elif ("_K" in k.upper()) and (len(k) <= 4):
                if k_num == 0:
                    (ln,) = self._ax.plot(
                        fit_x,
                        v,
                        color=self.plot_style["k_line"]["color"],
                        label=self.plot_style["k_line"]["label"],
                    )
                else:
                    (ln,) = self._ax.plot(fit_x, v, color=self.plot_style["k_line"]["color"], label="_nolegend_")
                self.plot_fit_obj.append(ln)
                k_num += 1

            elif ("_L" in k.upper()) and (len(k) <= 4):
                if l_num == 0:
                    (ln,) = self._ax.plot(
                        fit_x,
                        v,
                        color=self.plot_style["l_line"]["color"],
                        label=self.plot_style["l_line"]["label"],
                    )
                else:
                    (ln,) = self._ax.plot(fit_x, v, color=self.plot_style["l_line"]["color"], label="_nolegend_")
                self.plot_fit_obj.append(ln)
                l_num += 1

            elif ("_M" in k.upper()) and (len(k) <= 4):
                if m_num == 0:
                    (ln,) = self._ax.plot(
                        fit_x,
                        v,
                        color=self.plot_style["m_line"]["color"],
                        label=self.plot_style["m_line"]["label"],
                    )
                else:
                    (ln,) = self._ax.plot(fit_x, v, color=self.plot_style["m_line"]["color"], label="_nolegend_")
                self.plot_fit_obj.append(ln)
                m_num += 1

            else:
                pass

        # self._update_canvas()

    def canvas_onpress(self, event):
        """Callback, mouse button pressed"""
        if self.t_bar.mode == "":
            if event.inaxes == self._ax:
                if event.button == 1:
                    xd = event.xdata
                    self.set_plot_vertical_marker(marker_position=xd, mouse_clicked=True)
                else:
                    self.hide_plot_vertical_marker(mouse_clicked=True)

    # ===========================================================
    #         Functions for plotting spectrum preview

    def selected_range_indices(self, *, e_low=None, e_high=None, n_indexes=None, margin=2.0):
        """
        The function computes the range of indices based on the selected energy range
        and parameters for the energy axis.

        Parameters
        ----------
        e_low, e_high: float or None
            Energy values (in keV) that set the selected range
        n_indexes: int
            Total number of indexes in the energy array (typically 4096)
        margin: float
            The displayed energy range is extended by the value of `margin` in both directions.

        Returns
        -------
        n_low, n_high: int
            The range of indices of the energy array (n_low..n_high-1) that cover the selected energy range
        """
        # The range of energy selected for analysis
        if e_low is None:
            e_low = self.param_model.param_new["non_fitting_values"]["energy_bound_low"]["value"]
        if e_high is None:
            e_high = self.param_model.param_new["non_fitting_values"]["energy_bound_high"]["value"]
        # Protection for the case if e_high < e_low
        e_high = e_high if e_high > e_low else e_low
        # Extend the range (by the value of 'margin')
        e_low, e_high = e_low - margin, e_high + margin

        # The following calculations ignore quadratic term, which is expected to be small
        c0 = self.param_model.param_new["e_offset"]["value"]
        c1 = self.param_model.param_new["e_linear"]["value"]
        # If more precision if needed, then implement more complicated algorithm using
        #   the quadratic term: c2 = self.param_model.param_new['e_quadratic']['value']

        n_low = int(np.clip(int((e_low - c0) / c1), a_min=0, a_max=n_indexes - 1))
        n_high = int(np.clip(int((e_high - c0) / c1) + 1, a_min=1, a_max=n_indexes))

        return n_low, n_high

    def _datasets_max_size(self, *, only_displayed=True):
        """
        Return maximum size of the longest available dataset. The datasets that contain
        no data are ignored.

        Parameters
        ----------
        only_displayed: bool
            Limit search to the datasets that are going to be displayed
        """
        max_size = 0
        for dset in self.io_model.data_sets.values():
            if not only_displayed or dset.selected_for_preview:
                # Raw data shape: (n_rows, n_columns, n_energy_bins)
                max_size = max(max_size, dset.get_raw_data_shape()[2])
        return max_size

    def plot_selected_energy_range(self, *, axes, barh_existing, e_low=None, e_high=None, n_points=4096):
        """
        Plot the range of energies selected for processing. The range may be optionally
        provided as arguments. The range values that are not provided, are read from
        globally accessible dictionary of parameters. The values passed as arguments
        are mainly used if the function is called during interactive update of the
        range, when the order of update is undetermined and the parameter dictionary
        may be updated after the function is called.
        """
        # The range of energy selected for analysis
        if e_low is None:
            e_low = self.param_model.param_new["non_fitting_values"]["energy_bound_low"]["value"]
        if e_high is None:
            e_high = self.param_model.param_new["non_fitting_values"]["energy_bound_high"]["value"]

        # Model coefficients for the energy axis
        c0 = self.param_model.param_new["e_offset"]["value"]
        c1 = self.param_model.param_new["e_linear"]["value"]
        c2 = self.param_model.param_new["e_quadratic"]["value"]

        # Generate the values for 'energy' axis
        x_v = c0 + np.arange(n_points) * c1 + np.arange(n_points) ** 2 * c2
        ss = (x_v < e_high + c1) & (x_v > e_low - c1)

        # Trim both arrays to minimize the number of points
        x_v = x_v[ss]
        ss = ss[ss]
        ss[0] = False
        ss[-1] = False

        # Negative values will work for semilog plot as well
        y_min, y_max = -1e30, 1e30  # Select the max and min values for plotted rectangles

        # Remove the plot if it exists
        if barh_existing in axes.collections:
            axes.collections.remove(barh_existing)

        # Create the new plot (based on new parameters if necessary
        barh_new = BrokenBarHCollection.span_where(
            x_v, ymin=y_min, ymax=y_max, where=ss, facecolor="white", edgecolor="yellow", alpha=1
        )
        axes.add_collection(barh_new)

        return barh_new

    def prepare_preview_spectrum_plot(self):

        if self._ax_preview:
            self._ax_preview.clear()
        else:
            self._ax_preview = self._fig_preview.add_subplot(111)
        self._ax_preview.set_facecolor("lightgrey")
        self._ax_preview.grid(which="both")

        self._fig_preview.set_visible(False)

    def _show_preview_spectrum_plot(self):

        # Completely redraw the plot each time the function is called
        self.prepare_preview_spectrum_plot()

        # Remove all lines from the plot
        while len(self._lines_preview):
            self._lines_preview.pop().remove()

        # The list of color names
        color_names = get_color_name()

        e_range = self.energy_range_preview
        e_range_supported = (EnergyRangePresets.SELECTED_RANGE, EnergyRangePresets.FULL_SPECTRUM)
        if e_range not in e_range_supported:
            logger.error(
                f"Spectrum preview: Unknown option for the energy range: {e_range}\n"
                "Please report the error to the development team."
            )
            # This is not a critical error, so we still can proceed
            e_range = EnergyRangePresets.FULL_SPECTRUM

        p_type = self.plot_type_preview
        p_type_supported = (PlotTypes.LINLOG, PlotTypes.LINEAR)
        if p_type not in p_type_supported:
            logger.error(
                f"Spectrum preview: Unknown option for the plot type: {p_type}\n"
                "Please report the error to the development team."
            )
            p_type = PlotTypes.LINEAR

        # Maximum number of points in the displayed dataset
        n_dset_points = self._datasets_max_size()

        if e_range == EnergyRangePresets.SELECTED_RANGE:
            n_range_low, n_range_high = self.selected_range_indices(n_indexes=n_dset_points)
        else:
            n_range_low, n_range_high = 0, n_dset_points

        # All available datasets, we will print only the selected datasets
        dset_names = list(self.io_model.data_sets.keys())

        if p_type == PlotTypes.LINLOG:
            top_margin_coef = 2.0
            # Minimum for semilog plots may need to be computed, but 1.0 is good
            self.min_v_preview = 1.0
            self._ax_preview.set_yscale("log")
        else:
            top_margin_coef = 1.05
            self.min_v_preview = 0.0  # Minimum will always be 0 for linear plots

        self.max_v_preview = 1.0
        self.min_e_preview = 1000.0  # Start with some large number
        self.max_e_preview = 0.1  # Start with some small number
        for n_line, dset_name in enumerate(dset_names):
            dset = self.io_model.data_sets[dset_name]

            # Select color (even if the dataset is not displayed). This is done in order
            #   to ensure that each dataset is assigned the unique color.
            color = color_names[n_line % len(color_names)]

            if dset.selected_for_preview:

                data_arr = np.asarray(dset.get_total_spectrum())
                if data_arr is None:  # Just a precaution, it shouldn't happen
                    logger.error("Spectrum review: attempting to print empty dataset.")
                    continue

                # The assumption is that some datasets may have different length (which is
                #   currently not the case). So we have to take it into account when using
                #   maximum dataset length. This is essentially a safety precaution.
                n_low = int(np.clip(n_range_low, a_min=0, a_max=data_arr.size - 1))
                n_high = int(np.clip(n_range_high, a_min=1, a_max=data_arr.size))

                # From now on we work with the trimmed data array
                x_v = (
                    self.param_model.param_new["e_offset"]["value"]
                    + np.arange(n_low, n_high) * self.param_model.param_new["e_linear"]["value"]
                    + np.arange(n_low, n_high) ** 2 * self.param_model.param_new["e_quadratic"]["value"]
                )

                data_arr = data_arr[n_low:n_high]

                self.max_v_preview = np.max(
                    [self.max_v_preview, np.max(data_arr[self.limit_cut : -self.limit_cut])]
                )
                self.max_e_preview = np.max([self.max_e_preview, x_v[-1]])
                self.min_e_preview = np.min([self.min_e_preview, x_v[0]])

                (line,) = self._ax_preview.plot(
                    x_v,
                    data_arr,
                    color=color,
                    label=dset.filename.split(".")[0],
                    linestyle=self.plot_style["experiment"]["linestyle"],
                    marker=self.plot_style["experiment"]["marker"],
                )

                self._lines_preview.append(line)

        self._ax_preview.set_xlim(self.min_e_preview, self.max_e_preview)
        self._ax_preview.set_ylim(self.min_v_preview, self.max_v_preview * top_margin_coef)
        self._ax_preview.legend()
        self._ax_preview.set_xlabel("Energy (keV)")
        self._ax_preview.set_ylabel("Total Spectrum (Counts)")
        self._fig_preview.set_visible(True)

        # Reset navigation toolbar (specifically clear ZOOM history, since it becomes invalid
        #   when the new data is loaded, i.e. zooming out may not show the whole plot)
        tb = self._fig_preview.canvas.toolbar
        tb.update()

        self._bahr_preview = self.plot_selected_energy_range(
            axes=self._ax_preview, barh_existing=self._bahr_preview
        )

    def _hide_preview_spectrum_plot(
        self,
    ):
        self._fig_preview.set_visible(False)

    def update_preview_spectrum_plot(self, *, hide=False):
        """
        Update spectrum preview plot based on available/selected dataset and `hide` flag.

        Parameters
        ----------
        hide: bool
            `False` - plot data if datasets are available and at least one dataset is selected,
            otherwise hide the plot, `True` - hide the plot in any case
        """
        # Find out if any data is selected
        show_plot = False
        if self.io_model.data_sets:
            show_plot = any([_.selected_for_preview for _ in self.io_model.data_sets.values()])
        logger.debug(f"LinePlotModel.update_preview_spectrum_plot(): show_plot={show_plot} hide={hide}")
        if show_plot and not hide:
            logger.debug("LinePlotModel.update_preview_spectrum_plot(): plotting existing datasets")
            self._show_preview_spectrum_plot()
        else:
            logger.debug("LinePlotModel.update_preview_spectrum_plot(): hiding plots")
            self._hide_preview_spectrum_plot()
        self._fig_preview.canvas.draw()

    # ===========================================================================================
    #   Plotting the preview of Total Count Maps

    def clear_map_preview_range(self):
        self.set_map_preview_range(low=-1, high=-1)

    def set_map_preview_range(self, *, low, high):
        self.map_preview_range_low = low
        self.map_preview_range_high = high

    def get_selected_datasets(self):
        """Returns the datasets selected for preview"""
        return {k: v for (k, v) in self.io_model.data_sets.items() if v.selected_for_preview}

    def _compute_map_preview_range(self, img_dict, key_list):
        range_min, range_max = None, None
        for key in key_list:
            data = img_dict[key]
            v_min, v_max = np.min(data), np.max(data)
            if range_min is None or range_max is None:
                range_min, range_max = v_min, v_max
            else:
                range_min, range_max = min(range_min, v_min), max(range_max, v_max)
        return range_min, range_max

    def _show_total_count_map_preview(self):
        self._fig_maps.set_visible(True)
        self._fig_maps.clf()
        selected_dsets = self.get_selected_datasets()
        data_for_plotting = {k: v.get_total_count() for (k, v) in selected_dsets.items()}

        # Check if positions data is available. Positions data may be unavailable
        # (not recorded in HDF5 file) if experiment is has not been completed.
        # While the data from the completed part of experiment may still be used,
        # plotting vs. x-y or scatter plot may not be displayed.
        positions_data_available = False
        if "positions" in self.io_model.img_dict.keys():
            data_for_plotting["positions"] = self.io_model.img_dict["positions"]
            positions_data_available = True

        # Create local copies of self.pixel_or_pos, self.scatter_show and self.grid_interpolate
        pixel_or_pos_local = self.map_axes_units_preview
        scatter_show_local = self.map_scatter_plot

        # Disable plotting vs x-y coordinates if 'positions' data is not available
        if not positions_data_available:
            if pixel_or_pos_local:
                pixel_or_pos_local = MapAxesUnits.PIXELS  # Switch to plotting vs. pixel number
                logger.error("'Positions' data is not available. Plotting vs. x-y coordinates is disabled")
            if scatter_show_local:
                scatter_show_local = False  # Switch to plotting vs. pixel number
                logger.error("'Positions' data is not available. Scatter plot is disabled.")

        # low_lim = 1e-4  # define the low limit for log image
        plot_interp = "Nearest"

        grey_use = self.map_preview_color_scheme

        ncol = int(np.ceil(np.sqrt(len(selected_dsets))))
        try:
            nrow = int(np.ceil(len(selected_dsets) / float(ncol)))
        except ZeroDivisionError:
            ncol = 1
            nrow = 1

        a_pad_v = 0.8
        a_pad_h = 0.5

        n_displayed_axes = ncol * nrow  # Total number of axes in the grid

        grid = ImageGrid(
            self._fig_maps,
            111,
            nrows_ncols=(nrow, ncol),
            axes_pad=(a_pad_v, a_pad_h),
            cbar_location="right",
            cbar_mode="each",
            cbar_size="7%",
            cbar_pad="2%",
            share_all=True,
        )

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

        # Hide the axes that are unused (they are unsightly)
        for i in range(len(selected_dsets), n_displayed_axes):
            grid[i].set_visible(False)
            grid.cbar_axes[i].set_visible(False)

        for i, (k, v) in enumerate(selected_dsets.items()):

            data_arr = data_for_plotting[k]

            if pixel_or_pos_local == MapAxesUnits.POSITIONS or scatter_show_local:

                # xd_min, xd_max, yd_min, yd_max = min(self.x_pos), max(self.x_pos),
                #     min(self.y_pos), max(self.y_pos)
                x_pos_2D = data_for_plotting["positions"]["x_pos"]
                y_pos_2D = data_for_plotting["positions"]["y_pos"]
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

                yd, xd = data_arr.shape

                xd_min, xd_max, yd_min, yd_max = 0, xd, 0, yd
                if (yd <= math.floor(xd / 100)) and (xd >= 200):
                    yd_min, yd_max = -math.floor(xd / 200), math.ceil(xd / 200)
                if (xd <= math.floor(yd / 100)) and (yd >= 200):
                    xd_min, xd_max = -math.floor(yd / 200), math.ceil(yd / 200)

                xd_axis_min, xd_axis_max, yd_axis_min, yd_axis_max = _compute_equal_axes_ranges(
                    xd_min, xd_max, yd_min, yd_max
                )

            # Compute range for data values
            low_limit = self.map_preview_range_low
            high_limit = self.map_preview_range_high
            # If limit is not set, then compute the limit based on the selected datasets.
            # It is assumed that at least one dataset is selected.
            if low_limit == -1 and high_limit == -1:
                low_limit, high_limit = self._compute_map_preview_range(data_for_plotting, selected_dsets.keys())
                if low_limit is None or high_limit is None:
                    low_limit, high_limit = 0
            # Set some minimum range for the colorbar (otherwise it will have white fill)
            if math.isclose(low_limit, high_limit, abs_tol=2e-20):
                if abs(low_limit) < 1e-20:  # The value is zero
                    dv = 1e-20
                else:
                    dv = math.fabs(low_limit * 0.01)
                high_limit += dv
                low_limit -= dv

            if self.map_type_preview == MapTypes.LINEAR:
                if not scatter_show_local:
                    im = grid[i].imshow(
                        data_arr,
                        cmap=grey_use,
                        interpolation=plot_interp,
                        extent=(xd_min, xd_max, yd_max, yd_min),
                        origin="upper",
                        clim=(low_limit, high_limit),
                    )
                    grid[i].set_ylim(yd_axis_max, yd_axis_min)
                else:
                    xx = self.io_model.img_dict["positions"]["x_pos"]
                    yy = self.io_model.img_dict["positions"]["y_pos"]

                    # The following condition prevents crash if different file is loaded while
                    #    the scatter plot is open (PyXRF specific issue)
                    if data_arr.shape == xx.shape and data_arr.shape == yy.shape:
                        im = grid[i].scatter(
                            xx,
                            yy,
                            c=data_arr,
                            marker="s",
                            s=500,
                            alpha=1.0,  # Originally: alpha=0.8
                            cmap=grey_use,
                            vmin=low_limit,
                            vmax=high_limit,
                            linewidths=1,
                            linewidth=0,
                        )
                        grid[i].set_ylim(yd_axis_max, yd_axis_min)

                grid[i].set_xlim(xd_axis_min, xd_axis_max)

                grid_title = k
                # Display only the channel name (e.g. 'sum', 'det1' etc.)
                grid_title = grid_title.split("_")[-1]
                grid[i].text(0, 1.01, grid_title, ha="left", va="bottom", transform=grid[i].axes.transAxes)

                grid.cbar_axes[i].colorbar(im)

                im.colorbar.formatter = im.colorbar.ax.yaxis.get_major_formatter()
                # im.colorbar.ax.get_xaxis().set_ticks([])
                # im.colorbar.ax.get_xaxis().set_ticks([], minor=True)
                grid.cbar_axes[i].ticklabel_format(style="sci", scilimits=(-3, 4), axis="both")

            else:

                # maxz = np.max(data_arr)
                # # Set some reasonable minimum range for the colorbar
                # #   Zeros or negative numbers will be shown in white
                # if maxz <= 1e-30:
                #     maxz = 1

                if not scatter_show_local:
                    im = grid[i].imshow(
                        data_arr,
                        # norm=LogNorm(vmin=low_lim*maxz,
                        #              vmax=maxz, clip=True),
                        norm=LogNorm(vmin=low_limit, vmax=high_limit, clip=True),
                        cmap=grey_use,
                        interpolation=plot_interp,
                        extent=(xd_min, xd_max, yd_max, yd_min),
                        origin="upper",
                        # clim=(low_lim*maxz, maxz))
                        clim=(low_limit, high_limit),
                    )

                    grid[i].set_ylim(yd_axis_max, yd_axis_min)
                else:
                    im = grid[i].scatter(
                        self.io_model.img_dict["positions"]["x_pos"],
                        self.io_model.img_dict["positions"]["y_pos"],
                        # norm=LogNorm(vmin=low_lim*maxz,
                        #              vmax=maxz, clip=True),
                        norm=LogNorm(vmin=low_limit, vmax=high_limit, clip=True),
                        c=data_arr,
                        marker="s",
                        s=500,
                        alpha=1.0,  # Originally: alpha=0.8
                        cmap=grey_use,
                        linewidths=1,
                        linewidth=0,
                    )
                    grid[i].set_ylim(yd_axis_min, yd_axis_max)

                grid[i].set_xlim(xd_axis_min, xd_axis_max)

                grid_title = k
                # Display only the channel name (e.g. 'sum', 'det1' etc.)
                grid_title = grid_title.split("_")[-1]
                grid[i].text(0, 1.01, grid_title, ha="left", va="bottom", transform=grid[i].axes.transAxes)

                grid.cbar_axes[i].colorbar(im)
                im.colorbar.formatter = im.colorbar.ax.yaxis.get_major_formatter()
                im.colorbar.ax.get_xaxis().set_ticks([])
                im.colorbar.ax.get_xaxis().set_ticks([], minor=True)
                im.colorbar.ax.yaxis.set_minor_formatter(mticker.LogFormatter())

            grid[i].get_xaxis().set_major_locator(mticker.MaxNLocator(nbins="auto"))
            grid[i].get_yaxis().set_major_locator(mticker.MaxNLocator(nbins="auto"))

            grid[i].get_xaxis().get_major_formatter().set_useOffset(False)
            grid[i].get_yaxis().get_major_formatter().set_useOffset(False)

        self._fig_maps.canvas.draw_idle()

    def _hide_total_count_map_preview(self):
        self._fig_maps.set_visible(False)

    def update_total_count_map_preview(self, *, hide=False, new_plot=False):
        """
        Update total count map preview based on available/selected dataset and `hide` flag.

        Parameters
        ----------
        hide: bool
            `True` - plot data if datasets are available and at least one dataset is selected,
            otherwise hide the plot, `False` - hide the plot in any case
        new_plot: bool
            `True` - plotting new data that was just loaded, reset the plot settings
        """

        if new_plot and not hide:
            # Clear the displayed data range. The range will be computed based on the available data.
            self.clear_map_preview_range()

        # Find out if any data is selected
        show_plot = False
        if self.io_model.data_sets:
            show_plot = any([_.selected_for_preview for _ in self.io_model.data_sets.values()])
        logger.debug(f"LinePlotModel.update_total_count_map_preview(): show_plot={show_plot} hide={hide}")
        if show_plot and not hide:
            logger.debug("LinePlotModel.update_total_count_map_preview(): plotting existing datasets")
            self._show_total_count_map_preview()
        else:
            logger.debug("LinePlotModel.update_total_count_map_preview(): hiding plots")
            self._hide_total_count_map_preview()
        self._fig_maps.canvas.draw()
