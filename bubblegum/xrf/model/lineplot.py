# ######################################################################
# Copyright (c) 2014, Brookhaven Science Associates, Brookhaven        #
# National Laboratory. All rights reserved.                            #
#                                                                      #
# Redistribution and use in source and binary forms, with or without   #
# modification, are permitted provided that the following conditions   #
# are met:                                                             #
#                                                                      #
# * Redistributions of source code must retain the above copyright     #
#   notice, this list of conditions and the following disclaimer.      #
#                                                                      #
# * Redistributions in binary form must reproduce the above copyright  #
#   notice this list of conditions and the following disclaimer in     #
#   the documentation and/or other materials provided with the         #
#   distribution.                                                      #
#                                                                      #
# * Neither the name of the Brookhaven Science Associates, Brookhaven  #
#   National Laboratory nor the names of its contributors may be used  #
#   to endorse or promote products derived from this software without  #
#   specific prior written permission.                                 #
#                                                                      #
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS  #
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT    #
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS    #
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE       #
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,           #
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES   #
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR   #
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)   #
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,  #
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OTHERWISE) ARISING   #
# IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE   #
# POSSIBILITY OF SUCH DAMAGE.                                          #
########################################################################

from __future__ import (absolute_import, division,
                        print_function)

__author__ = 'Li Li'

import numpy as np
import six
import matplotlib
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from collections import OrderedDict

from atom.api import Atom, Str, observe, Typed, Int, List, Dict, Float, Bool

from skxray.fitting.xrf_model import (k_line, l_line, m_line)
from skxray.constants.api import XrfElement as Element

import logging
logger = logging.getLogger(__name__)


def get_color_name():
    return matplotlib.colors.cnames.keys()


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
    total_y_l : dict
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
    data = Typed(object) #Typed(np.ndarray)
    exp_data_label = Str('experiment')

    _fig = Typed(Figure)
    _ax = Typed(Axes)
    _canvas = Typed(object)
    element_id = Int(0)
    parameters = Dict()
    elist = List()
    scale_opt = Int(0)
    total_y = Dict()
    total_y_l = Dict()
    total_y_m = Dict()

    prefit_x = Typed(object)
    plot_title = Str()
    fit_x = Typed(np.ndarray)
    fit_y = Typed(np.ndarray)
    residual = Typed(np.ndarray)

    plot_type = List()
    max_v = Typed(object)
    incident_energy = Float(30.0)

    eline_obj = List()

    plot_exp_opt = Bool()
    plot_exp_obj = Typed(Line2D)
    show_exp_opt = Bool()

    plot_exp_list = List()
    data_sets = Typed(OrderedDict)

    auto_fit_obj = List()
    show_autofit_opt = Bool()

    plot_fit_obj = List() #Typed(Line2D)
    show_fit_opt = Bool()
    fit_all = Typed(object)

    plot_style = Dict()

    roi_plot_dict = Dict()
    roi_dict = Typed(object) #OrderedDict()

    #prefix_name_roi = Str()
    #element_for_roi = Str()
    #element_list_roi = List()
    #roi_dict = Typed(object) #OrderedDict()

    #data_dict = Dict()
    #roi_result = Dict()

    def __init__(self):
        self._fig = plt.figure()

        self._ax = self._fig.add_subplot(111)
        #self._ax.set_axis_bgcolor('black')
        self._ax.legend(loc=2)

        self._ax.set_xlabel('Energy [keV]')
        self._ax.set_ylabel('Counts')
        self._ax.set_yscale('log')
        self.plot_type = ['LinLog', 'Linear']

        self._ax.autoscale_view(tight=True)
        self.max_v = 1.0
        self._color_config()
        self._fig.tight_layout(pad=0.5)

    def _color_config(self):
        self.plot_style = {
            'experiment': {'color': 'blue', 'linestyle': '', 'marker': '.', 'label': self.exp_data_label},
            'background': {'color': 'grey', 'marker': '+', 'markersize': 1, 'label': 'background'},
            'emission_line': {'color': 'red', 'linewidth': 2},
            'k_line': {'color': 'green', 'label': 'k lines'},
            'l_line': {'color': 'purple', 'label': 'l lines'},
            'm_line': {'color': 'orange', 'label': 'm lines'},
            'compton': {'color': 'cyan', 'label': 'compton'},
            'elastic': {'color': '#b66718', 'label': 'elastic'},
            'auto_fit': {'color': 'black', 'label': 'auto fitted'},
            'fit': {'color': 'black', 'label': 'fitted'}
        }

    def _update_canvas(self):
        self._ax.legend(loc=2)
        #lg = self._ax.get_legend()
        #lg.set_alpha(0.005)
        self._ax.legend(framealpha=0.2)
        self._fig.tight_layout(pad=0.5)
        self._fig.canvas.draw()

    @observe('exp_data_label')
    def _update_exp_label(self, change):
        if change['type'] == 'create':
            return
        self.plot_style['experiment']['label'] = change['value']

    @observe('parameters')
    def _update_energy(self, change):
        self.incident_energy = self.parameters['coherent_sct_energy'].value

    @observe('scale_opt')
    def _new_opt(self, change):
        if self.plot_type[change['value']] == 'LinLog':
            self._ax.set_yscale('log')
            self._ax.set_ylim([self.max_v*1e-6, self.max_v*10.0])
        else:
            self._ax.set_yscale('linear')
            self._ax.set_ylim([-0.15*self.max_v, self.max_v*1.2])
        self._update_canvas()

    @observe('data')
    def data_update(self, change):
        self.max_v = np.max(self.data)
        self._ax.set_ylim([self.max_v*1e-6, self.max_v*10.0])

    @observe('plot_exp_opt')
    def _new_exp_plot_opt(self, change):
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
        try:
            self.plot_exp_obj.remove()
            logger.debug('Previous experimental data is removed.')
        except AttributeError:
            logger.debug('No need to remove experimental data.')

        data_arr = np.asarray(self.data)
        x_v = (self.parameters['e_offset'].value +
               np.arange(len(data_arr)) *
               self.parameters['e_linear'].value +
               np.arange(len(data_arr))**2 *
               self.parameters['e_quadratic'].value)

        self.plot_exp_obj, = self._ax.plot(x_v, data_arr,
                                           linestyle=self.plot_style['experiment']['linestyle'],
                                           color=self.plot_style['experiment']['color'],
                                           marker=self.plot_style['experiment']['marker'],
                                           label=self.plot_style['experiment']['label'])

    def plot_multi_exp_data(self):
        while(len(self.plot_exp_list)):
            self.plot_exp_list.pop().remove()

        color_n = get_color_name()

        for i, (k, v) in enumerate(six.iteritems(self.data_sets)):
            if v.plot_index:
                data_arr = np.asarray(v.data)
                self.max_v = np.max(data_arr)
                x_v = (self.parameters['e_offset'].value +
                       np.arange(len(data_arr)) *
                       self.parameters['e_linear'].value +
                       np.arange(len(data_arr))**2 *
                       self.parameters['e_quadratic'].value)

                plot_exp_obj, = self._ax.plot(x_v, data_arr,
                                              color=color_n[i],
                                              label=v.filename.split('.')[0])
                self.plot_exp_list.append(plot_exp_obj)

    @observe('show_exp_opt')
    def _update_exp(self, change):
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
        while(len(self.eline_obj)):
            self.eline_obj.pop().remove()

        if len(self.elist):
            self._ax.hold(True)
            for i in range(len(self.elist)):
                eline, = self._ax.plot([self.elist[i][0], self.elist[i][0]],
                                       [0, self.elist[i][1]*self.max_v],
                                       color=self.plot_style['emission_line']['color'],
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

        self.elist = []
        total_list = k_line + l_line + m_line
        logger.debug('Plot emission line for element: {}'.format(self.element_id))
        ename = total_list[self.element_id-1]

        incident_energy = self.incident_energy
        logger.info('Use incident energy {} for emission line calculation.'.format(incident_energy))

        if len(ename) <= 2:
            e = Element(ename)
            if e.cs(incident_energy)['ka1'] != 0:
                for i in range(4):
                    self.elist.append((e.emission_line.all[i][1],
                                       e.cs(incident_energy).all[i][1]/e.cs(incident_energy).all[0][1]))

        elif '_L' in ename:
            e = Element(ename[:-2])
            if e.cs(incident_energy)['la1'] != 0:
                for i in range(4, 17):
                    self.elist.append((e.emission_line.all[i][1],
                                       e.cs(incident_energy).all[i][1]/e.cs(incident_energy).all[4][1]))

        else:
            e = Element(ename[:-2])
            if e.cs(incident_energy)['ma1'] != 0:
                for i in range(17, 21):
                    self.elist.append((e.emission_line.all[i][1],
                                       e.cs(incident_energy).all[i][1]/e.cs(incident_energy).all[17][1]))
        self.plot_emission_line()
        self._update_canvas()

    def plot_roi_bound(self):
        """
        Plot roi with low, high and ceter value.
        """
        for k, v in six.iteritems(self.roi_plot_dict):
            for data in v:
                data.remove()
        self.roi_plot_dict.clear()

        if len(self.roi_dict):
            #self._ax.hold(True)
            for k, v in six.iteritems(self.roi_dict):
                temp_list = []
                for linev in np.array([v.left_val, v.line_val, v.right_val])/1000.:
                    lineplot, = self._ax.plot([linev, linev],
                                              [0, 1*self.max_v],
                                              color=self.plot_style['emission_line']['color'],
                                              linewidth=self.plot_style['emission_line']['linewidth'])
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

    def plot_autofit(self):
        sum = 0
        while(len(self.auto_fit_obj)):
            self.auto_fit_obj.pop().remove()

        k_auto = 0
        l_auto = 0
        m_auto = 0

        # K lines
        if len(self.total_y):
            self._ax.hold(True)
            for i, (k, v) in enumerate(six.iteritems(self.total_y)):
                if k == 'background':
                    ln, = self._ax.plot(self.prefit_x, v,
                                        color=self.plot_style['background']['color'],
                                        marker=self.plot_style['background']['marker'],
                                        markersize=self.plot_style['background']['markersize'],
                                        label=self.plot_style['background']['label'])
                elif k == 'compton':
                    ln, = self._ax.plot(self.prefit_x, v,
                                        color=self.plot_style['compton']['color'],
                                        label=self.plot_style['compton']['label'])
                elif k == 'elastic':
                    ln, = self._ax.plot(self.prefit_x, v,
                                        color=self.plot_style['elastic']['color'],
                                        label=self.plot_style['elastic']['label'])
                else:
                    # only the first one has label
                    if k_auto == 0:
                        ln, = self._ax.plot(self.prefit_x, v,
                                            color=self.plot_style['k_line']['color'],
                                            label=self.plot_style['k_line']['label'])
                    else:
                        ln, = self._ax.plot(self.prefit_x, v,
                                            color=self.plot_style['k_line']['color'],
                                            label='_nolegend_')
                    k_auto += 1
                self.auto_fit_obj.append(ln)
                sum += v

        # L lines
        if len(self.total_y_l):
            self._ax.hold(True)
            for i, (k, v) in enumerate(six.iteritems(self.total_y_l)):
                # only the first one has label
                if l_auto == 0:
                    ln, = self._ax.plot(self.prefit_x, v,
                                        color=self.plot_style['l_line']['color'],
                                        label=self.plot_style['l_line']['label'])
                else:
                    ln, = self._ax.plot(self.prefit_x, v,
                                        color=self.plot_style['l_line']['color'],
                                        label='_nolegend_')
                l_auto += 1
                self.auto_fit_obj.append(ln)
                sum += v

        # M lines
        if len(self.total_y_m):
            self._ax.hold(True)
            for i, (k, v) in enumerate(six.iteritems(self.total_y_m)):
                # only the first one has label
                if m_auto == 0:
                    ln, = self._ax.plot(self.prefit_x, v,
                                        color=self.plot_style['m_line']['color'],
                                        label=self.plot_style['m_line']['label'])
                else:
                    ln, = self._ax.plot(self.prefit_x, v,
                                        color=self.plot_style['m_line']['color'],
                                        label='_nolegend_')
                m_auto += 1
                self.auto_fit_obj.append(ln)
                sum += v

        if len(self.total_y) or len(self.total_y_l) or len(self.total_y_m):
            ln, = self._ax.plot(self.prefit_x, sum,
                                color=self.plot_style['auto_fit']['color'],
                                label=self.plot_style['auto_fit']['label'])
            self.auto_fit_obj.append(ln)

    @observe('show_autofit_opt')
    def update_auto_fit(self, change):
        if change['value']:
            if len(self.auto_fit_obj):
                for v in self.auto_fit_obj:
                    v.set_visible(True)
                    lab = v.get_label()
                    if lab != '_nolegend_':
                        v.set_label(lab.strip('_'))
        else:
            if len(self.auto_fit_obj):
                for v in self.auto_fit_obj:
                    v.set_visible(False)
                    lab = v.get_label()
                    if lab != '_nolegend_':
                        v.set_label('_' + lab)
        self._update_canvas()

    def plot_fit(self):
        #if len(self.fit_y):
        while(len(self.plot_fit_obj)):
            self.plot_fit_obj.pop().remove()
        ln, = self._ax.plot(self.fit_x, self.fit_y,
                            color=self.plot_style['fit']['color'],
                            label=self.plot_style['fit']['label'])
        self.plot_fit_obj.append(ln)

        shiftv = 1.5  # move residual down by some amount
        ln, = self._ax.plot(self.fit_x, self.residual - shiftv*(np.max(np.abs(self.residual))),
                            color=self.plot_style['fit']['color'],
                            label='residual')
        self.plot_fit_obj.append(ln)

        k_num = 0
        l_num = 0
        m_num = 0
        for k, v in six.iteritems(self.fit_all):
            k = str(k)
            if '_L' in k.upper():
                l_num += 1
                if l_num == 1:
                    ln, = self._ax.plot(self.fit_x, v,
                                        color=self.plot_style['l_line']['color'],
                                        label=self.plot_style['l_line']['label'])
                else:
                    ln, = self._ax.plot(self.fit_x, v,
                                        color=self.plot_style['l_line']['color'],
                                        label='_nolegend_')
                self.plot_fit_obj.append(ln)
            elif '_M' in k.upper():
                m_num += 1
                if m_num == 1:
                    ln, = self._ax.plot(self.fit_x, v,
                                        color=self.plot_style['m_line']['color'],
                                        label=self.plot_style['m_line']['label'])
                else:
                    ln, = self._ax.plot(self.fit_x, v,
                                        color=self.plot_style['m_line']['color'],
                                        label='_nolegend_')
                self.plot_fit_obj.append(ln)
            elif k == 'background':
                ln, = self._ax.plot(self.fit_x, v,
                                    color=self.plot_style['background']['color'],
                                    marker=self.plot_style['background']['marker'],
                                    markersize=self.plot_style['background']['markersize'],
                                    label=self.plot_style['background']['label'])
                self.plot_fit_obj.append(ln)
            elif k == 'compton':
                ln, = self._ax.plot(self.fit_x, v,
                                    color=self.plot_style['compton']['color'],
                                    label=self.plot_style['compton']['label'])
                self.plot_fit_obj.append(ln)
            elif k == 'elastic':
                ln, = self._ax.plot(self.fit_x, v,
                                    color=self.plot_style['elastic']['color'],
                                    label=self.plot_style['elastic']['label'])
                self.plot_fit_obj.append(ln)
            else:
                k_num += 1
                if k_num == 1:
                    ln, = self._ax.plot(self.fit_x, v,
                                        color=self.plot_style['k_line']['color'],
                                        label=self.plot_style['k_line']['label'])
                else:
                    ln, = self._ax.plot(self.fit_x, v,
                                        color=self.plot_style['k_line']['color'],
                                        label='_nolegend_')
                self.plot_fit_obj.append(ln)

    @observe('show_fit_opt')
    def _update_fit(self, change):
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

    def set_prefit_data(self, prefit_x,
                        total_y, total_y_l, total_y_m):
        """
        Parameters
        ----------
        prefit_x : array
            X axis with limited range
        total_y : dict
            Results for k lines, bg, and others
        total_y_l : dict
            Results for l lines
        total_y_m : dict
            Results for m lines
        """
        self.prefit_x = prefit_x
        # k lines
        self.total_y = total_y
        # l lines
        self.total_y_l = total_y_l
        # m lines
        self.total_y_m = total_y_m

        self._ax.set_xlim([self.prefit_x[0], self.prefit_x[-1]])
        self.plot_autofit()

