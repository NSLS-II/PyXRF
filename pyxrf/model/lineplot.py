from __future__ import (absolute_import, division,
                        print_function)

import numpy as np
import six
import matplotlib
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from collections import OrderedDict

from atom.api import Atom, Str, observe, Typed, Int, List, Dict, Float, Bool

from skbeam.core.fitting.xrf_model import (K_LINE, L_LINE, M_LINE)
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

    plot_exp_list = List()
    data_sets = Typed(OrderedDict)

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

    # data_dict = Dict()
    # roi_result = Dict()

    def __init__(self):
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

    def _update_canvas(self):
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

    @observe('scale_opt')
    def _new_opt(self, change):
        self.log_linear_plot()
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
        self._update_canvas()

    @observe('plot_exp_opt')
    def _new_exp_plot_opt(self, change):
        if change['type'] != 'create':
            if change['value']:
                self.plot_exp_obj.set_visible(True)
                lab = self.plot_exp_obj.get_label()
                self.plot_exp_obj.set_label(lab.strip('_'))
            else:
                self.plot_exp_obj.set_visible(False)
                lab = self.plot_exp_obj.get_label()
                self.plot_exp_obj.set_label('_' + lab)

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

    def plot_multi_exp_data(self):
        while(len(self.plot_exp_list)):
            self.plot_exp_list.pop().remove()

        color_n = get_color_name()

        self.max_v = 1.0
        m = 0
        for (k, v) in six.iteritems(self.data_sets):
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

        self._update_ylimit()
        self.log_linear_plot()
        self._update_canvas()

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

    @observe('element_id')
    def set_element(self, change):
        if change['value'] == 0:
            while(len(self.eline_obj)):
                self.eline_obj.pop().remove()
            self.elist = []
            self._fig.canvas.draw()
            return

        incident_energy = self.incident_energy
        k_len = len(K_TRANSITIONS)
        l_len = len(L_TRANSITIONS)
        m_len = len(M_TRANSITIONS)

        self.elist = []
        total_list = K_LINE + L_LINE + M_LINE
        logger.debug('Plot emission line for element: '
                     '{} with incident energy {}'.format(self.element_id,
                                                         incident_energy))
        ename = total_list[self.element_id-1]

        if '_K' in ename:
            e = Element(ename[:-2])
            if e.cs(incident_energy)['ka1'] != 0:
                for i in range(k_len):
                    self.elist.append((e.emission_line.all[i][1],
                                       e.cs(incident_energy).all[i][1]
                                       / e.cs(incident_energy).all[0][1]))

        elif '_L' in ename:
            e = Element(ename[:-2])
            if e.cs(incident_energy)['la1'] != 0:
                for i in range(k_len, k_len+l_len):
                    self.elist.append((e.emission_line.all[i][1],
                                       e.cs(incident_energy).all[i][1]
                                       / e.cs(incident_energy).all[k_len][1]))

        else:
            e = Element(ename[:-2])
            if e.cs(incident_energy)['ma1'] != 0:
                for i in range(k_len+l_len, k_len+l_len+m_len):
                    self.elist.append((e.emission_line.all[i][1],
                                       e.cs(incident_energy).all[i][1]
                                       / e.cs(incident_energy).all[k_len+l_len][1]))
        self.plot_emission_line()
        self._update_canvas()

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
