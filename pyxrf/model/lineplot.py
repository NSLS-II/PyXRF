from __future__ import (absolute_import, division,
                        print_function)

import numpy as np
import six
import matplotlib
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.collections import BrokenBarHCollection
from collections import OrderedDict

from atom.api import Atom, Str, observe, Typed, Int, List, Dict, Float, Bool

from skbeam.core.fitting.xrf_model import (K_TRANSITIONS, L_TRANSITIONS, M_TRANSITIONS)

from skbeam.fluorescence import XrfElement as Element

import logging
logger = logging.getLogger()


def get_color_name():

    # usually line plot will not go beyond 10
    first_ten = ['indigo', 'maroon', 'green', 'darkblue', 'darkgoldenrod', 'blue',
                 'darkcyan', 'sandybrown', 'black', 'darkolivegreen']

    # Avoid red color, as those color conflict with emission lines' color.
    nonred_list = [v for v in matplotlib.colors.cnames.keys()
                   if 'pink' not in v and 'fire' not in v and
                   'sage' not in v and 'tomato' not in v and 'red' not in v]
    return first_ten + nonred_list + list(matplotlib.colors.cnames.keys())


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
    plot_type : list
        linear or log plot
    max_v : float
        max value of data array
    incident_energy : float
        in KeV
    param_model : Typed(object)
        Reference to GuessParamModel object
    """
    data = Typed(object)  # Typed(np.ndarray)
    exp_data_label = Str('experiment')

    _fig = Typed(Figure)
    _ax = Typed(Axes)
    _canvas = Typed(object)
    element_id = Int(0)
    parameters = Dict()
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

    plot_type = List()
    max_v = Float()
    incident_energy = Float(12.0)

    eline_obj = List()

    plot_exp_opt = Bool(False)
    plot_exp_obj = Typed(Line2D)
    show_exp_opt = Bool(False)

    # Reference to artist responsible for displaying the selected range of energies on the plot
    plot_energy_barh = Typed(BrokenBarHCollection)
    t_bar = Typed(object)

    plot_exp_list = List()
    data_sets = Typed(OrderedDict)

    auto_fit_obj = List()
    show_autofit_opt = Bool()

    plot_fit_obj = List()  # Typed(Line2D)
    show_fit_opt = Bool(False)
    # fit_all = Typed(object)

    allow_add_eline = Bool(False)
    allow_remove_eline = Bool(False)
    allow_select_elines = Bool(False)
    # The same flag for 'Fit' spectrum window
    #   Those flag also must allow to select Userpeak1 .. Userpeak10
    allow_add_eline_fit = Bool(False)
    allow_remove_eline_fit = Bool(False)

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

    # data_dict = Dict()
    # roi_result = Dict()

    # Reference to GuessParamModel object
    param_model = Typed(object)

    # Location of the vertical (mouse-selected) marker on the plot.
    # Value is in kev. Negative value - no marker is placed.
    vertical_marker_kev = Float(-1)
    # Reference to the respective Matplotlib artist
    line_vertical_marker = Typed(object)
    vertical_marker_is_visible = Bool(False)

    def __init__(self, param_model):

        # Reference to GuessParamModel object
        self.param_model = param_model

        self.data = None

        self._fig = plt.figure(figsize=(3, 2))

        self._ax = self._fig.add_subplot(111)
        try:
            self._ax.set_axis_bgcolor('lightgrey')
        except AttributeError:
            self._ax.set_facecolor('lightgrey')

        self._ax.set_xlabel('Energy [keV]')
        self._ax.set_ylabel('Counts')
        self._ax.set_yscale('log')
        self.plot_type = ['LinLog', 'Linear']

        self._ax.autoscale_view(tight=True)
        self._ax.legend(loc=2)

        self._color_config()
        self._fig.tight_layout(pad=0.5)
        self.max_v = 1.0
        # when we calculate max value, data smaller than 500, 0.5 Kev, can be ignored.
        # And the last point of data is also huge, and should be cut off.
        self.limit_cut = 100
        # self._ax.margins(x=0.0, y=0.10)

    def _color_config(self):
        self.plot_style = {
            'experiment': {'color': 'blue', 'linestyle': '',
                           'marker': '.', 'label': self.exp_data_label},
            'background': {'color': 'indigo', 'marker': '+',
                           'markersize': 1, 'label': 'background'},
            'emission_line': {'color': 'black', 'linewidth': 2},
            'roi_line': {'color': 'red', 'linewidth': 2},
            'k_line': {'color': 'green', 'label': 'k lines'},
            'l_line': {'color': 'magenta', 'label': 'l lines'},
            'm_line': {'color': 'brown', 'label': 'm lines'},
            'compton': {'color': 'darkcyan', 'linewidth': 1.5, 'label': 'compton'},
            'elastic': {'color': 'purple', 'label': 'elastic'},
            'escape': {'color': 'darkblue', 'label': 'escape'},
            'pileup': {'color': 'darkgoldenrod', 'label': 'pileup'},
            'userpeak': {'color': 'orange', 'label': 'userpeak'},
            # 'auto_fit': {'color': 'black', 'label': 'auto fitted', 'linewidth': 2.5},
            'fit': {'color': 'red', 'label': 'fit', 'linewidth': 2.5},
            'residual': {'color': 'black', 'label': 'residual', 'linewidth': 2.0}
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
        self.plot_exp_opt = False   # exp data for fitting
        self.show_exp_opt = False   # all exp data from different channels
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
        self.log_range = [self.max_v*1e-5, self.max_v*2]
        # self.linear_range = [-0.3*self.max_v, self.max_v*1.2]
        self.linear_range = [0, self.max_v*1.2]

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
        self.exp_data_label = change['value']
        self.plot_style['experiment']['label'] = change['value']

    # @observe('exp_data_label')
    # def _change_exp_label(self, change):
    #     if change['type'] == 'create':
    #         return
    #     self.plot_style['experiment']['label'] = change['value']

    @observe('parameters')
    def _update_energy(self, change):
        if 'coherent_sct_energy' not in self.parameters:
            return
        self.incident_energy = self.parameters['coherent_sct_energy']['value']

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
        if 'coherent_sct_energy' in self.param_model.param_new:
            self.param_model.param_new['coherent_sct_energy']['value'] = energy_new

        # Change the value twice to ensure that all observer functions are called
        self.param_model.energy_bound_high_buf = energy_new + 1.8  # Arbitrary number
        upper_bound = energy_new + margin
        # Limit the number of decimal points for better visual presentation
        upper_bound = round(upper_bound, ndigits=5)
        self.param_model.energy_bound_high_buf = upper_bound

    @observe('scale_opt')
    def _new_opt(self, change):
        self.log_linear_plot()
        self._update_canvas()

    def energy_bound_high_update(self, change):
        """Observer function for 'param_model.energy_bound_high_buf'"""
        if self.data is None:
            return
        self.plot_selected_energy_range(e_high=change["value"])
        self.plot_vertical_marker(e_high=change["value"])
        self._update_canvas()

    def energy_bound_low_update(self, change):
        """Observer function for 'param_model.energy_bound_low_buf'"""
        if self.data is None:
            return
        self.plot_selected_energy_range(e_low=change["value"])
        self.plot_vertical_marker(e_low=change["value"])
        self._update_canvas()

    def log_linear_plot(self):
        if self.plot_type[self.scale_opt] == 'LinLog':
            self._ax.set_yscale('log')
            # self._ax.margins(x=0.0, y=0.5)
            # self._ax.autoscale_view(tight=True)
            # self._ax.relim(visible_only=True)
            self._ax.set_ylim(self.log_range)

        else:
            self._ax.set_yscale('linear')
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
        self.data = change['value']
        if self.data is None:
            return

        # conflicts between float and np.float32 ???
        self.max_v = float(np.max(self.data[self.limit_cut:-self.limit_cut]))

        if not self.parameters:
            return

        try:
            self.plot_exp_obj.remove()
            logger.debug('Previous experimental data is removed.')
        except AttributeError:
            logger.debug('No need to remove experimental data.')

        x_v = (self.parameters['e_offset']['value'] +
               np.arange(len(self.data)) *
               self.parameters['e_linear']['value'] +
               np.arange(len(self.data))**2 *
               self.parameters['e_quadratic']['value'])

        self.plot_exp_obj, = self._ax.plot(x_v, self.data,
                                           linestyle=self.plot_style['experiment']['linestyle'],
                                           color=self.plot_style['experiment']['color'],
                                           marker=self.plot_style['experiment']['marker'],
                                           label=self.plot_style['experiment']['label'])

        self._update_ylimit()
        self.log_linear_plot()

        self._set_eline_select_controls()

        self.plot_selected_energy_range()

        # _show_hide_exp_plot is called to show or hide current plot based
        #           on the state of _show_exp_opt flag
        self._show_hide_exp_plot(self.show_exp_opt or self.plot_exp_opt)

    def _show_hide_exp_plot(self, plot_show):

        if self.data is None:
            return

        try:
            if plot_show:
                self.plot_exp_obj.set_visible(True)
                lab = self.plot_exp_obj.get_label()
                self.plot_exp_obj.set_label(lab.strip('_'))
            else:
                self.plot_exp_obj.set_visible(False)
                lab = self.plot_exp_obj.get_label()
                self.plot_exp_obj.set_label('_' + lab)

            self._update_canvas()
        except Exception:
            pass

    @observe('plot_exp_opt')
    def _new_exp_plot_opt(self, change):

        if self.data is None:
            return

        if change['type'] != 'create':
            if change['value']:
                self.plot_experiment()
            # _show_hide_exp_plot is already called inside 'plot_experiment()',
            #    but visibility flag was not used correctly. So we need to
            #    call it again.
            self._show_hide_exp_plot(change['value'])
            self._set_eline_select_controls()

    @observe('show_exp_opt')
    def _update_exp(self, change):
        if change['type'] != 'create':
            if change['value']:
                if len(self.plot_exp_list):
                    for v in self.plot_exp_list:
                        v.set_visible(True)
                        lab = v.get_label()
                        if lab != '_nolegend_':
                            v.set_label(lab.strip('_'))
            else:
                if len(self.plot_exp_list):
                    for v in self.plot_exp_list:
                        v.set_visible(False)
                        lab = v.get_label()
                        if lab != '_nolegend_':
                            v.set_label('_' + lab)
            self._update_canvas()

    @observe('show_fit_opt')
    def _update_fit(self, change):
        if change['type'] != 'create':
            if change['value']:
                for v in self.plot_fit_obj:
                    v.set_visible(True)
                    lab = v.get_label()
                    if lab != '_nolegend_':
                        v.set_label(lab.strip('_'))
            else:
                for v in self.plot_fit_obj:
                    v.set_visible(False)
                    lab = v.get_label()
                    if lab != '_nolegend_':
                        v.set_label('_' + lab)
            self._update_canvas()

    def plot_experiment(self):
        """
        PLot raw experiment data for fitting.
        """

        # Do nothing if no data is loaded
        if self.data is None:
            return

        data_arr = np.asarray(self.data)
        self.exp_data_update({'value': data_arr})

    def plot_vertical_marker(self, *, e_low=None, e_high=None):

        self._vertical_marker_set_inside_range(e_low=e_low, e_high=e_high)

        x_v = (self.vertical_marker_kev, self.vertical_marker_kev)
        y_v = (-1e30, 1e30)  # This will cover the range of possible values of accumulated counts

        if self.line_vertical_marker:
            self._ax.lines.remove(self.line_vertical_marker)
            self.line_vertical_marker = None
        if self.vertical_marker_is_visible:
            self.line_vertical_marker, = self._ax.plot(x_v, y_v, color="blue")

    def set_plot_vertical_marker(self, marker_position=None):
        """
        The function is called when setting the position of the marker interactively

        If the parameter `marker_position` is `None`, then don't set or change the value.
        Just make the marker visible.
        """

        # Ignore the new value if it is outside the range of selected energies
        if marker_position is not None:
            e_low = self.param_model.param_new['non_fitting_values']['energy_bound_low']['value']
            e_high = self.param_model.param_new['non_fitting_values']['energy_bound_high']['value']
            if (marker_position >= e_low) and (marker_position <= e_high):
                self.vertical_marker_kev = marker_position

        # Compute peak intensity. The displayed value will change only for user defined peak,
        #   since it is moved to the position of the marker.
        self.compute_manual_peak_intensity()

        # Make the marker visible
        self.vertical_marker_is_visible = True

        # Update the location of the marker and the canvas
        self.plot_vertical_marker()
        self._update_canvas()

    def hide_plot_vertical_marker(self):
        """Hide vertical marker"""
        self.vertical_marker_is_visible = False
        self.plot_vertical_marker()
        self._update_canvas()

    def plot_selected_energy_range(self, *, e_low=None, e_high=None):
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
            e_low = self.param_model.param_new['non_fitting_values']['energy_bound_low']['value']
        if e_high is None:
            e_high = self.param_model.param_new['non_fitting_values']['energy_bound_high']['value']

        n_x = 4096  # Set to the maximum possible number of points

        # Generate the values for 'energy' axis
        x_v = (self.parameters['e_offset']['value'] +
               np.arange(n_x) *
               self.parameters['e_linear']['value'] +
               np.arange(n_x) ** 2 *
               self.parameters['e_quadratic']['value'])

        ss = (x_v < e_high) & (x_v > e_low)
        y_min, y_max = 1e-30, 1e30  # Select the max and min values for plotted rectangles

        # Remove the plot if it exists
        if self.plot_energy_barh in self._ax.collections:
            self._ax.collections.remove(self.plot_energy_barh)

        # Create the new plot (based on new parameters if necessary
        self.plot_energy_barh = BrokenBarHCollection.span_where(
            x_v, ymin=y_min, ymax=y_max, where=ss, facecolor='white', edgecolor='yellow', alpha=1)
        self._ax.add_collection(self.plot_energy_barh)

    def plot_multi_exp_data(self):
        while(len(self.plot_exp_list)):
            self.plot_exp_list.pop().remove()

        color_n = get_color_name()

        self.max_v = 1.0
        m = 0
        for (k, v) in self.data_sets.items():
            if v.plot_index:

                data_arr = np.asarray(v.data)
                self.max_v = np.max([self.max_v,
                                     np.max(data_arr[self.limit_cut:-self.limit_cut])])

                x_v = (self.parameters['e_offset']['value'] +
                       np.arange(len(data_arr)) *
                       self.parameters['e_linear']['value'] +
                       np.arange(len(data_arr))**2 *
                       self.parameters['e_quadratic']['value'])

                plot_exp_obj, = self._ax.plot(x_v, data_arr,
                                              color=color_n[m],
                                              label=v.filename.split('.')[0],
                                              linestyle=self.plot_style['experiment']['linestyle'],
                                              marker=self.plot_style['experiment']['marker'])
                self.plot_exp_list.append(plot_exp_obj)
                m += 1

        self.plot_selected_energy_range()

        self._update_ylimit()
        self.log_linear_plot()
        self._update_canvas()

    def plot_emission_line(self):
        """
        Plot emission line and escape peaks associated with given lines.
        The value of self.max_v is needed in this function in order to plot
        the relative height of each emission line.
        """
        while(len(self.eline_obj)):
            self.eline_obj.pop().remove()

        escape_e = self.escape_e

        if len(self.elist):
            for i in range(len(self.elist)):
                eline, = self._ax.plot([self.elist[i][0], self.elist[i][0]],
                                       [0, self.elist[i][1]*self.max_v],
                                       color=self.plot_style['emission_line']['color'],
                                       linewidth=self.plot_style['emission_line']['linewidth'])
                self.eline_obj.append(eline)
                if self.plot_escape_line and self.elist[i][0] > escape_e:
                    eline, = self._ax.plot([self.elist[i][0]-escape_e,
                                            self.elist[i][0]-escape_e],
                                           [0, self.elist[i][1]*self.max_v],
                                           color=self.plot_style['escape']['color'],
                                           linewidth=self.plot_style['emission_line']['linewidth'])
                    self.eline_obj.append(eline)

    def _set_eline_select_controls(self, *, element_id=None, data="use_self_data"):
        if element_id is None:
            element_id = self.element_id
        if data == "use_self_data":
            data = self.data
        self._set_allow_add_eline(element_id=element_id, data=data)
        self._set_allow_remove_eline(element_id=element_id, data=data)
        self._set_allow_select_elines(data=data)

    def _set_allow_add_eline(self, *, element_id, data):
        """
        Sets the flag, which enables/disables the controls
        for manually adding an element emission line.

        Parameters
        ----------

        element_id : int
            The number of the element emission line in the list

        data : ndarray or None
            Reference to the data array.
        """
        flag, flag_fit = True, True
        if data is None:
            flag, flag_fit = False, False
        if not self.is_element_line_id_valid(element_id):
            flag = False
        elif self.is_line_in_selected_list(element_id):
            flag = False
        if not self.is_element_line_id_valid(element_id, include_user_peaks=True):
            flag_fit = False
        elif self.is_line_in_selected_list(element_id, include_user_peaks=True):
            flag_fit = False
        self.allow_add_eline = flag
        self.allow_add_eline_fit = flag_fit

    def _set_allow_remove_eline(self, *, element_id, data):
        """
        Sets the flag, which enables/disables the controls
        for manually removing an element emission line.

        Parameters
        ----------

        element_id : Int
            The number of the element emission line in the list

        data : ndarray or None
            Reference to the data array.
        """
        flag, flag_fit = False, False
        if data is not None:
            flag, flag_fit = True, True
        if not self.is_line_in_selected_list(element_id):
            flag = False
        if not self.is_line_in_selected_list(element_id, include_user_peaks=True):
            flag_fit = False
        self.allow_remove_eline = flag
        self.allow_remove_eline_fit = flag_fit

    def _set_allow_select_elines(self, data):
        """
        Sets the flag, which enables/disables the controls
        for manual selection of an element emission line.
        The selected line may be added or removed.

        Parameters
        ----------

        element_id : int
            The number of the element emission line in the list

        data : ndarray or None
            Reference to the data array.
        """
        flag = True
        if data is None:
            flag = False
        self.allow_select_elines = flag

    def is_line_in_selected_list(self, n_id, *, include_user_peaks=False):
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

        ename = self.get_element_line_name_by_id(n_id, include_user_peaks=include_user_peaks)

        if ename is None:
            return False

        if self.param_model.EC.is_element_in_list(ename):
            return True
        else:
            return False

    def is_element_line_id_valid(self, n_id, *, include_user_peaks=False):
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
        ename = self.get_element_line_name_by_id(n_id, include_user_peaks=include_user_peaks)

        if ename is None:
            return False
        else:
            return True

    def get_element_line_name_by_id(self, n_id, *, include_user_peaks=False):
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
        total_list = self.param_model.get_user_peak_list(include_user_peaks=include_user_peaks)
        try:
            ename = total_list[n_id-1]
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
            e_low = self.param_model.param_new['non_fitting_values']['energy_bound_low']['value']
        if e_high is None:
            e_high = self.param_model.param_new['non_fitting_values']['energy_bound_high']['value']

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

        ename = self.get_element_line_name_by_id(self.element_id, include_user_peaks=True)

        if ename is not None:

            _elist = []
            if ename.lower().startswith("userpeak"):
                # Make sure that the marker is in the selected range of energies
                self._vertical_marker_set_inside_range()
                # The tuple structure: (center_energy, ratio)
                _elist.append((self.vertical_marker_kev, 1.0))

            elif '_K' in ename:
                e = Element(ename[:-2])
                if e.cs(incident_energy)['ka1'] != 0:
                    for i in range(k_len):
                        _elist.append((e.emission_line.all[i][1],
                                       e.cs(incident_energy).all[i][1]
                                       / e.cs(incident_energy).all[0][1]))

            elif '_L' in ename:
                e = Element(ename[:-2])
                if e.cs(incident_energy)['la1'] != 0:
                    for i in range(k_len, k_len+l_len):
                        _elist.append((e.emission_line.all[i][1],
                                       e.cs(incident_energy).all[i][1]
                                       / e.cs(incident_energy).all[k_len][1]))

            else:
                e = Element(ename[:-2])
                if e.cs(incident_energy)['ma1'] != 0:
                    for i in range(k_len+l_len, k_len+l_len+m_len):
                        _elist.append((e.emission_line.all[i][1],
                                       e.cs(incident_energy).all[i][1]
                                       / e.cs(incident_energy).all[k_len+l_len][1]))

            return _elist

    @observe('element_id')
    def set_element(self, change):

        self._set_eline_select_controls(element_id=change['value'])
        self.compute_manual_peak_intensity(n_id=change['value'])

        def _reset_eline_plot():
            while(len(self.eline_obj)):
                self.eline_obj.pop().remove()
            self.elist = []
            self._fig.canvas.draw()

        if change['value'] == 0:
            _reset_eline_plot()
            return

        incident_energy = self.incident_energy
        ename = self.get_element_line_name_by_id(self.element_id, include_user_peaks=True)

        if ename is not None:

            logger.debug('Plot emission line for element: '
                         '{} with incident energy {}'.format(self.element_id,
                                                             incident_energy))

            _elist = self._fill_elist()
            if not ename.lower().startswith("userpeak"):
                self.elist = _elist
            else:
                self.elist = []
            self.plot_emission_line()
            self._update_canvas()

            # Do it the second time, since the 'self.elist' has changed
            self.compute_manual_peak_intensity(n_id=change['value'])

        else:

            _reset_eline_plot()
            logger.warning(f"Selected emission line with ID #{self.element_id} is not in the list.")

    @observe('det_materials')
    def _update_det_materials(self, change):
        if change['value'] == 0:
            self.escape_e = 1.73998
        else:
            self.escape_e = 9.88640

    def plot_roi_bound(self):
        """
        Plot roi with low, high and ceter value.
        """
        for k, v in six.iteritems(self.roi_plot_dict):
            for data in v:
                data.remove()
        self.roi_plot_dict.clear()

        if len(self.roi_dict):
            # self._ax.hold(True)
            for k, v in six.iteritems(self.roi_dict):
                temp_list = []
                for linev in np.array([v.left_val, v.line_val, v.right_val])/1000.:
                    lineplot, = self._ax.plot([linev, linev],
                                              [0, 1*self.max_v],
                                              color=self.plot_style['roi_line']['color'],
                                              linewidth=self.plot_style['roi_line']['linewidth'])
                    if v.show_plot:
                        lineplot.set_visible(True)
                    else:
                        lineplot.set_visible(False)
                    temp_list.append(lineplot)
                self.roi_plot_dict.update({k: temp_list})

        self._update_canvas()

    @observe('roi_dict')
    def show_roi_bound(self, change):
        logger.debug('roi dict changed {}'.format(change['value']))
        self.plot_roi_bound()

        if len(self.roi_dict):
            for k, v in six.iteritems(self.roi_dict):
                if v.show_plot:
                    for l in self.roi_plot_dict[k]:
                        l.set_visible(True)
                else:
                    for l in self.roi_plot_dict[k]:
                        l.set_visible(False)
        self._update_canvas()

    def add_peak_manual(self):

        self.param_model.manual_input(userpeak_center=self.vertical_marker_kev)
        self.param_model.EC.update_peak_ratio()
        self.param_model.update_name_list()
        self.param_model.data_for_plot()

        self.hide_plot_vertical_marker()

        self.plot_fit(self.param_model.prefit_x,
                      self.param_model.total_y,
                      self.param_model.auto_fit_all)

        # For plotting purposes, otherwise plot will not update
        if self.plot_exp_opt:
            self.plot_exp_opt = False
            self.plot_exp_opt = True
        else:
            self.plot_exp_opt = True
            self.plot_exp_opt = False
        self.show_fit_opt = False
        self.show_fit_opt = True

    def remove_peak_manual(self, peak_name):

        self.param_model.EC.delete_item(peak_name)
        self.param_model.EC.update_peak_ratio()
        self.param_model.data_for_plot()

        self.plot_fit(self.param_model.prefit_x,
                      self.param_model.total_y,
                      self.param_model.auto_fit_all)

        # For plotting purposes, otherwise plot will not update
        if self.plot_exp_opt:
            self.plot_exp_opt = False
            self.plot_exp_opt = True
        else:
            self.plot_exp_opt = True
            self.plot_exp_opt = False
        self.show_fit_opt = False
        self.show_fit_opt = True
        self.param_model.update_name_list()

    def _compute_intensity(self, elist):
        # Some default value
        intensity = 1000.0

        if self.data is not None and self.parameters is not None \
                and self.param_model.prefit_x is not None \
                and len(self.data) > 1 and len(self.param_model.prefit_x) > 1:

            # Range of energies in fitting results
            e_fit_min = self.param_model.prefit_x[0]
            e_fit_max = self.param_model.prefit_x[-1]
            de_fit = (e_fit_max - e_fit_min) / (len(self.param_model.prefit_x) - 1)

            e_raw_min = self.parameters['e_offset']['value']
            e_raw_max = self.parameters['e_offset']['value'] + \
                (len(self.data) - 1) * self.parameters['e_linear']['value'] + \
                (len(self.data) - 1) ** 2 * self.parameters['e_quadratic']['value']
            de_raw = (e_raw_max - e_raw_min) / (len(self.data) - 1)

            # Note: the above algorithm for finding 'de_raw' is far from perfect but will
            #    work for now. As a result 'de_fit' and 'de_raw' == self.parameters['e_linear']['value'].
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
            n = np.clip(n, 0, len(self.data) - 1)
            n_raw = int(round(n))
            # Intensity of the fitted data at the peak
            in_fit = self.param_model.total_y[n_fit]
            # Intensity of the raw data at the peak
            in_raw = self.data[n_raw]
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

        if not self.is_element_line_id_valid(n_id, include_user_peaks=True):
            # This is typicall the case when n_id==0
            intensity = 0.0
            if self.is_element_line_id_valid(n_id, include_user_peaks=True):
                # This means we are dealing with user defined peak. Display intensity if the peak is in the list.
                if self.is_line_in_selected_list(n_id, include_user_peaks=True):
                    name = self.get_element_line_name_by_id(n_id, include_user_peaks=True)
                    intensity = self.param_model.EC.element_dict[name].maxv

        else:
            if self.is_line_in_selected_list(n_id, include_user_peaks=True):
                name = self.get_element_line_name_by_id(n_id, include_user_peaks=True)
                intensity = self.param_model.EC.element_dict[name].maxv
            else:
                _elist = self._fill_elist()
                intensity = self._compute_intensity(_elist)

        # Round the intensity for nicer printing
        self.param_model.add_element_intensity = round(intensity, 2)

    # def plot_autofit(self):
    #     sum_y = 0
    #     while(len(self.auto_fit_obj)):
    #         self.auto_fit_obj.pop().remove()
    #
    #     k_auto = 0
    #
    #     # K lines
    #     if len(self.total_y):
    #         self._ax.hold(True)
    #         for k, v in six.iteritems(self.total_y):
    #             if k == 'background':
    #                 ln, = self._ax.plot(self.prefit_x, v,
    #                                     color=self.plot_style['background']['color'],
    #                                     #marker=self.plot_style['background']['marker'],
    #                                     #markersize=self.plot_style['background']['markersize'],
    #                                     label=self.plot_style['background']['label'])
    #             elif k == 'compton':
    #                 ln, = self._ax.plot(self.prefit_x, v,
    #                                     color=self.plot_style['compton']['color'],
    #                                     linewidth=self.plot_style['compton']['linewidth'],
    #                                     label=self.plot_style['compton']['label'])
    #             elif k == 'elastic':
    #                 ln, = self._ax.plot(self.prefit_x, v,
    #                                     color=self.plot_style['elastic']['color'],
    #                                     label=self.plot_style['elastic']['label'])
    #             elif k == 'escape':
    #                 ln, = self._ax.plot(self.prefit_x, v,
    #                                     color=self.plot_style['escape']['color'],
    #                                     label=self.plot_style['escape']['label'])
    #             else:
    #                 # only the first one has label
    #                 if k_auto == 0:
    #                     ln, = self._ax.plot(self.prefit_x, v,
    #                                         color=self.plot_style['k_line']['color'],
    #                                         label=self.plot_style['k_line']['label'])
    #                 else:
    #                     ln, = self._ax.plot(self.prefit_x, v,
    #                                         color=self.plot_style['k_line']['color'],
    #                                         label='_nolegend_')
    #                 k_auto += 1
    #             self.auto_fit_obj.append(ln)
    #             sum_y += v
    #
    #     # L lines
    #     if len(self.total_l):
    #         self._ax.hold(True)
    #         for i, (k, v) in enumerate(six.iteritems(self.total_l)):
    #             # only the first one has label
    #             if i == 0:
    #                 ln, = self._ax.plot(self.prefit_x, v,
    #                                     color=self.plot_style['l_line']['color'],
    #                                     label=self.plot_style['l_line']['label'])
    #             else:
    #                 ln, = self._ax.plot(self.prefit_x, v,
    #                                     color=self.plot_style['l_line']['color'],
    #                                     label='_nolegend_')
    #             self.auto_fit_obj.append(ln)
    #             sum_y += v
    #
    #     # M lines
    #     if len(self.total_m):
    #         self._ax.hold(True)
    #         for i, (k, v) in enumerate(six.iteritems(self.total_m)):
    #             # only the first one has label
    #             if i == 0:
    #                 ln, = self._ax.plot(self.prefit_x, v,
    #                                     color=self.plot_style['m_line']['color'],
    #                                     label=self.plot_style['m_line']['label'])
    #             else:
    #                 ln, = self._ax.plot(self.prefit_x, v,
    #                                     color=self.plot_style['m_line']['color'],
    #                                     label='_nolegend_')
    #             self.auto_fit_obj.append(ln)
    #             sum_y += v
    #
    #     # pileup
    #     if len(self.total_pileup):
    #         self._ax.hold(True)
    #         for i, (k, v) in enumerate(six.iteritems(self.total_pileup)):
    #             # only the first one has label
    #             if i == 0:
    #                 ln, = self._ax.plot(self.prefit_x, v,
    #                                     color=self.plot_style['pileup']['color'],
    #                                     label=self.plot_style['pileup']['label'])
    #             else:
    #                 ln, = self._ax.plot(self.prefit_x, v,
    #                                     color=self.plot_style['pileup']['color'],
    #                                     label='_nolegend_')
    #             self.auto_fit_obj.append(ln)
    #             sum_y += v
    #
    #     if len(self.total_y) or len(self.total_l) or len(self.total_m):
    #         self._ax.hold(True)
    #         ln, = self._ax.plot(self.prefit_x, sum_y,
    #                             color=self.plot_style['auto_fit']['color'],
    #                             label=self.plot_style['auto_fit']['label'],
    #                             linewidth=self.plot_style['auto_fit']['linewidth'])
    #         self.auto_fit_obj.append(ln)

    # @observe('show_autofit_opt')
    # def update_auto_fit(self, change):
    #     if change['value']:
    #         if len(self.auto_fit_obj):
    #             for v in self.auto_fit_obj:
    #                 v.set_visible(True)
    #                 lab = v.get_label()
    #                 if lab != '_nolegend_':
    #                     v.set_label(lab.strip('_'))
    #     else:
    #         if len(self.auto_fit_obj):
    #             for v in self.auto_fit_obj:
    #                 v.set_visible(False)
    #                 lab = v.get_label()
    #                 if lab != '_nolegend_':
    #                     v.set_label('_' + lab)
    #     self._update_canvas()

    # def set_prefit_data_and_plot(self, prefit_x,
    #                              total_y, total_l,
    #                              total_m, total_pileup):
    #     """
    #     Parameters
    #     ----------
    #     prefit_x : array
    #         X axis with limited range
    #     total_y : dict
    #         Results for k lines, bg, and others
    #     total_l : dict
    #         Results for l lines
    #     total_m : dict
    #         Results for m lines
    #     total_pileup : dict
    #         Results for pileups
    #     """
    #     self.prefit_x = prefit_x
    #     # k lines
    #     self.total_y = total_y
    #     # l lines
    #     self.total_l = total_l
    #     # m lines
    #     self.total_m = total_m
    #     # pileup
    #     self.total_pileup = total_pileup
    #
    #     #self._ax.set_xlim([self.prefit_x[0], self.prefit_x[-1]])
    #     self.plot_autofit()
    #     #self.log_linear_plot()
    #     self._update_canvas()

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

        while(len(self.plot_fit_obj)):
            self.plot_fit_obj.pop().remove()

        ln, = self._ax.plot(fit_x, fit_y,
                            color=self.plot_style['fit']['color'],
                            label=self.plot_style['fit']['label'],
                            linewidth=self.plot_style['fit']['linewidth'])
        self.plot_fit_obj.append(ln)

        if residual is not None:
            # shiftv = 1.5  # move residual down by some amount
            ln, = self._ax.plot(fit_x,
                                residual - 0.15*self.max_v,  # shiftv*(np.max(np.abs(self.residual))),
                                label=self.plot_style['residual']['label'],
                                color=self.plot_style['residual']['color'])
            self.plot_fit_obj.append(ln)

        k_num = 0
        l_num = 0
        m_num = 0
        p_num = 0
        for k, v in six.iteritems(fit_all):
            if k == 'background':
                ln, = self._ax.plot(fit_x, v,
                                    color=self.plot_style['background']['color'],
                                    # marker=self.plot_style['background']['marker'],
                                    # markersize=self.plot_style['background']['markersize'],
                                    label=self.plot_style['background']['label'])
                self.plot_fit_obj.append(ln)
            elif k == 'compton':
                ln, = self._ax.plot(fit_x, v,
                                    color=self.plot_style['compton']['color'],
                                    linewidth=self.plot_style['compton']['linewidth'],
                                    label=self.plot_style['compton']['label'])
                self.plot_fit_obj.append(ln)
            elif k == 'elastic':
                ln, = self._ax.plot(fit_x, v,
                                    color=self.plot_style['elastic']['color'],
                                    label=self.plot_style['elastic']['label'])
                self.plot_fit_obj.append(ln)
            elif k == 'escape':
                ln, = self._ax.plot(fit_x, v,
                                    color=self.plot_style['escape']['color'],
                                    label=self.plot_style['escape']['label'])
                self.plot_fit_obj.append(ln)

            elif 'user' in k.lower():
                ln, = self._ax.plot(fit_x, v,
                                    color=self.plot_style['userpeak']['color'],
                                    label=self.plot_style['userpeak']['label'])
                self.plot_fit_obj.append(ln)

            elif '-' in k:  # Si_K-Si_K
                if p_num == 0:
                    ln, = self._ax.plot(fit_x, v,
                                        color=self.plot_style['pileup']['color'],
                                        label=self.plot_style['pileup']['label'])
                else:
                    ln, = self._ax.plot(fit_x, v,
                                        color=self.plot_style['pileup']['color'],
                                        label='_nolegend_')
                self.plot_fit_obj.append(ln)
                p_num += 1

            elif ('_K' in k.upper()) and (len(k) <= 4):
                if k_num == 0:
                    ln, = self._ax.plot(fit_x, v,
                                        color=self.plot_style['k_line']['color'],
                                        label=self.plot_style['k_line']['label'])
                else:
                    ln, = self._ax.plot(fit_x, v,
                                        color=self.plot_style['k_line']['color'],
                                        label='_nolegend_')
                self.plot_fit_obj.append(ln)
                k_num += 1

            elif ('_L' in k.upper()) and (len(k) <= 4):
                if l_num == 0:
                    ln, = self._ax.plot(fit_x, v,
                                        color=self.plot_style['l_line']['color'],
                                        label=self.plot_style['l_line']['label'])
                else:
                    ln, = self._ax.plot(fit_x, v,
                                        color=self.plot_style['l_line']['color'],
                                        label='_nolegend_')
                self.plot_fit_obj.append(ln)
                l_num += 1

            elif ('_M' in k.upper()) and (len(k) <= 4):
                if m_num == 0:
                    ln, = self._ax.plot(fit_x, v,
                                        color=self.plot_style['m_line']['color'],
                                        label=self.plot_style['m_line']['label'])
                else:
                    ln, = self._ax.plot(fit_x, v,
                                        color=self.plot_style['m_line']['color'],
                                        label='_nolegend_')
                self.plot_fit_obj.append(ln)
                m_num += 1

            else:
                pass

        # self._update_canvas()

    def canvas_onpress(self, event):
        """Callback, mouse button pressed"""
        if (self.t_bar.mode == ""):
            if event.inaxes == self._ax:
                if event.button == 1:
                    xd = event.xdata
                    self.set_plot_vertical_marker(marker_position=xd)
                else:
                    self.hide_plot_vertical_marker()
