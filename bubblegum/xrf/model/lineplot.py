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

__author__ = 'Li Li'

import numpy as np
import six
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.pyplot as plt

from pprint import pprint

from .guessparam import Parameter

from atom.api import Atom, Str, observe, Typed, Int, List, Dict

from skxray.fitting.xrf_model import (k_line, l_line, m_line)
from skxray.constants.api import XrfElement as Element


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
    """
    data = Typed(np.ndarray)
    _fig = Typed(Figure)
    _ax = Typed(Axes)
    _canvas = Typed(object)
    element_id = Int(0)
    parameters = Dict()
    elist = List()
    plot_opt = Int(0)
    total_y = Dict()
    total_y_l = Dict()
    prefit_x = Typed(object)
    plot_title = Str()

    def __init__(self):
        self._fig = plt.figure()
        self._ax = self._fig.add_subplot(111)

    #def set_data(self, data):
    #    self.data = data

    @observe('plot_opt')
    def _new_opt(self, change):
        if change['type'] == 'update':
            self.plot_data()

    def plot_data(self, min_ratio=1e-6):
        """
        Parameters
        ----------
        min_ratio : float, opt
            define the range of plotting
        """
        plot_type = ['LinLog', 'Linear']
        #self._ax = self._fig.add_subplot(111)
        #self._fig.title = self.plot_title

        self._ax.hold(False)
        data_arr = np.asarray(self.data)
        x_v = (self.parameters['e_offset'].value +
               np.arange(len(data_arr)) *
               self.parameters['e_linear'].value +
               np.arange(len(data_arr))**2 *
               self.parameters['e_quadratic'].value)

        if plot_type[self.plot_opt] == 'Linear':
            self._ax.plot(x_v, data_arr, 'b-', label='experiment')
        else:
            self._ax.semilogy(x_v, data_arr, 'b-', label='experiment')

        #minv = np.min(data_arr)
        minv = np.max(data_arr)*min_ratio
        if len(self.elist) != 0:
            self._ax.hold(True)
            for i in range(len(self.elist)):
                if plot_type[self.plot_opt] == 'Linear':
                    self._ax.plot([self.elist[i][0], self.elist[i][0]],
                                  [minv, self.elist[i][1]*np.max(data_arr)],
                                  'r-', linewidth=2.0)
                    self._ax.set_ylim([minv, np.max(data_arr)*1.5])
                else:
                    self._ax.semilogy([self.elist[i][0], self.elist[i][0]],
                                      [minv, self.elist[i][1]*np.max(data_arr)],
                                      'r-', linewidth=2.0)
                    self._ax.set_ylim([minv, np.max(data_arr)*10.0])

        if len(self.total_y) != 0:
            self._ax.hold(True)
            if plot_type[self.plot_opt] == 'Linear':
                sum = 0
                for i, (k, v) in enumerate(six.iteritems(self.total_y)):
                    if k == 'background':
                        self._ax.plot(self.prefit_x, v, 'grey')
                    else:
                        if i == 0:
                            self._ax.plot(self.prefit_x, v, 'g-', label='k line')
                        else:
                            self._ax.plot(self.prefit_x, v, 'g-')
                    sum += v

                if len(self.total_y_l) != 0:
                    for i, (k, v) in enumerate(six.iteritems(self.total_y_l)):
                        if i == 0:
                            self._ax.plot(self.prefit_x, v, 'purple', label='l line')
                        else:
                            self._ax.plot(self.prefit_x, v, 'purple')
                        sum += v

                self._ax.plot(self.prefit_x, sum, 'k*', markersize=2, label='prefit')
                self._ax.set_ylim([minv, np.max(data_arr)*1.5])
            else:
                sum = 0
                for i, (k, v) in enumerate(six.iteritems(self.total_y)):
                    if k == 'background':
                        self._ax.semilogy(self.prefit_x, v, 'grey')
                    else:
                        if i == 0:
                            self._ax.semilogy(self.prefit_x, v, 'g-', label='k line')
                        else:
                            self._ax.semilogy(self.prefit_x, v, 'g-')
                    sum += v

                if len(self.total_y_l) != 0:
                    for i, (k, v) in enumerate(six.iteritems(self.total_y_l)):
                        if i == 0:
                            self._ax.semilogy(self.prefit_x, v, 'purple', label='l line')
                        else:
                            self._ax.semilogy(self.prefit_x, v, 'purple')
                        sum += v

                self._ax.semilogy(self.prefit_x, sum, 'k*', markersize=2, label='prefit')
                self._ax.set_ylim([minv, np.max(data_arr)*10.0])

            self._ax.set_xlim([self.prefit_x[0], self.prefit_x[-1]])

        self._ax.legend(loc=0)

        self._ax.set_title(self.plot_title)
        self._ax.set_xlabel('Energy [keV]')
        self._ax.set_ylabel('Counts')

        self._fig.canvas.draw()

    @observe('element_id')
    def set_element(self, change):
        if change['value'] == 0:
            self.elist = []
            self.plot_data()
            return

        self.elist = []

        total_list = k_line + l_line + m_line
        print('element name: {}'.format(self.element_id))
        ename = total_list[self.element_id-1]
        incident_energy = self.parameters['coherent_sct_energy']['value']

        if len(ename) <= 2:
            e = Element(ename)
            if e.cs(incident_energy)['ka1'] != 0:
                for i in range(4):
                    self.elist.append((e.emission_line.all[i][1],
                                       e.cs(incident_energy).all[i][1]/e.cs(incident_energy).all[0][1]))

        elif '_L' in ename:
            e = Element(ename[:-2])
            print e.cs(incident_energy)['la1']
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

        self.plot_data()

    def set_prefit_data(self, prefit_x,
                        total_y, total_y_l):
        """
        Parameters
        ----------
        prefit_x : array
            X axis with limited range
        total_y : dict
            Results for k lines
        total_y_l : dict
            Results for l and m lines
        """
        self.prefit_x = prefit_x
        # k lines
        self.total_y = total_y
        # l lines
        self.total_y_l = total_y_l
