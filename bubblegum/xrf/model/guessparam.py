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
import json
from collections import OrderedDict
import copy
import logging
logger = logging.getLogger(__name__)

from pprint import pprint
from atom.api import Atom, Str, observe, Typed, Int, Dict, List, Float, Enum

from skxray.fitting.xrf_model import (ModelSpectrum, set_range, k_line, l_line,
                                      m_line, get_linear_model, PreFitAnalysis)
from skxray.fitting.background import snip_method

bound_options = ['none', 'lohi', 'fixed', 'lo', 'hi']
class Parameter(Atom):
    # todo make sure that these are the only valid bound types
    bound_type = Enum(*bound_options)
    min = Float(-np.inf)
    max = Float(np.inf)
    value = Float()
    default_value = Float()
    free_more = Enum(*bound_options)
    adjust_element = Enum(*bound_options)
    e_calibration = Enum(*bound_options)
    linear = Enum(*bound_options)
    name = Str()
    description = Str()
    tool_tip = Str()

    @observe('name')
    def update_displayed_name(self, changed):
        if not self.description:
            self.description = self.name

    def __repr__(self):
        return ("Parameter(bound_type={}, min={}, max={}, value={}, "
               "default_value={}, free_more={}, adjust_element={}, "
               "e_calibration={}, linear={}, name={}, description={}, "
               "toop_tip={}".format(
            self.bound_type, self.min, self.max, self.value, self.default_value,
            self.free_more, self.adjust_element, self.e_calibration,
            self.linear, self.name, self.description, self.tool_tip))

    def to_dict(self):
        return {
            'bound_type': self.bound_type,
            'min': self.min,
            'max': self.max,
            'value': self.value,
            'default_value': self.default_value,
            'free_more': self.free_more,
            'adjust_element': self.adjust_element,
            'e_calibration': self.e_calibration,
            'linear': self.linear,
            'name': self.name,
            'description': self.description,
            'tool_tip': self.tool_tip,
        }

def format_dict(parameter_object_dict, element_list):
    """Format the dictionary that scikit-xray expects
    """
    param_dict = {key: value.to_dict() for key, value
                  in six.iteritems(parameter_object_dict)}
    elo = param_dict.pop('energy_bound_low')['value']
    ehi = param_dict.pop('energy_bound_high')['value']
    # reformat the dictionary that scikit-xray expects
    non_fitting_values = {'non_fitting_values': {
        'energy_bound_low': elo,
        'energy_bound_high': ehi,
        'element_list': element_list
    }}
    param_dict.update(non_fitting_values)

    return param_dict

class GuessParamModel(Atom):
    """
    This is auto fit model to guess the initial parameters.

    Attributes
    ----------
    parameters : `atom.Dict`
        A list of `Parameter` objects, subclassed from the `Atom` base class.
        These `Parameter` objects hold all relevant xrf information
    default_params : dict
        Save all the fitting parameters. Values will not be changed in this dict.
    data : array
        1D array of spectrum
    prefit_x : array
        xX axis with range defined by low and high limits.
    result_dict : dict
        Save all the auto fitting results for each element
    status_dict : dict
        Plotting status of each element
    total_y : dict
        Results from k lines
    total_y_l : dict
        Results from l lines
    parameter_file_path : str
        Path to parameter file
    param_status : str
        Loading status of parameter file
    """
    parameters = Dict()
    data = Typed(object)
    prefit_x = Typed(object)
    result_dict = Typed(object)
    status_dict = Dict(value=bool, key=str)
    total_y = Typed(object)
    total_y_l = Typed(object)
    param_status = Str()
    element_list = List()

    def __init__(self, default_parameters, *args, **kwargs):
        self.get_param(default_parameters)
        self.total_y_l = {}

    def get_param(self, default_parameters):
        non_fitting_values = default_parameters.pop('non_fitting_values')
        element_list = non_fitting_values.pop('element_list')
        if not isinstance(element_list, list):
            element_list = element_list.split(',')
        self.element_list = element_list
        elo = non_fitting_values.pop('energy_bound_low')
        ehi = non_fitting_values.pop('energy_bound_high')
        self.parameters = {
            'energy_bound_low': Parameter(name='E low (keV)', value=elo,
                                          default_value=elo),
            'energy_bound_high': Parameter(name='E high (keV)',
                                           default_value=ehi, value=ehi)
        }
        self.parameters.update({
            param_name: Parameter(name=param_name,
                                  default_value=param_dict['value'],
                                  **param_dict)
            for param_name, param_dict in six.iteritems(default_parameters)
        } )
        pprint(self.parameters)
        # sort by the parameter name
        # parameters.sort(key=lambda s: s.name.lower())
        # self.param_status = ('Parameter file {} is loaded'.format(
        #     self.parameter_file_path.split('/')[-1]))

    def set_data(self, data):
        """
        Parameters
        ----------
        data : numpy array
            1D array of spectrum intensity
        """
        self.data = np.asarray(data)

    def save_param(self, param):
        self.parameters = param

    def find_peak(self):
        """run automatic peak finding."""
        #for k,v in six.iteritems(self.parameters):
        #    print('{}:{}'.format(k,v))
        param_dict = format_dict(self.parameters, self.element_list)
        self.prefit_x, self.result_dict, bg = pre_fit_linear(self.data,
                                                             param_dict)

        self.result_dict.update(background=bg)

        # save the plotting status for a given element peak
        self.status_dict = {k: True for k in six.iterkeys(self.result_dict)}

    @observe('status_dict')
    def update_status_dict(self, changed):
        print('status dict changed: {}'.format(changed))

    def arange_prefit_result(self):
        # change range based on dict data
        self.total_y = self.result_dict.copy()

        # update plotting status based on self.status_dict
        for k, v in six.iteritems(self.status_dict):
            if v is False:
                del self.total_y[k]

        self.total_y_l = {}
        for k, v in six.iteritems(self.total_y):
            if '_L' in k or '_M' in k:
                self.total_y_l.update({k: v})
                del self.total_y[k]


def pre_fit_linear(y0, fitting_parameters):
    """
    Run prefit to get initial elements.

    Parameters
    ----------
    y0 : array
        Spectrum intensity
    fitting_parameters : dict
        Fitting parameters
    Returns
    -------
    x : array
        x axis
    result_dict : dict
        Fitting results
    bg : array
        Calculated background ground
    """
    x0 = np.arange(len(y0))
    x, y = set_range(fitting_parameters, x0, y0)
    # get background
    bg = snip_method(y, fitting_parameters['e_offset']['value'],
                     fitting_parameters['e_linear']['value'],
                     fitting_parameters['e_quadratic']['value'])

    y = y - bg

    element_list = k_line + l_line + m_line
    new_element = ', '.join(element_list)
    fitting_parameters['non_fitting_values']['element_list'] = new_element

    non_element = ['compton', 'elastic']
    total_list = element_list + non_element
    total_list = [str(v) for v in total_list]

    matv = get_linear_model(x, fitting_parameters)

    x = fitting_parameters['e_offset']['value'] + fitting_parameters['e_linear']['value']*x + \
        fitting_parameters['e_quadratic']['value'] * x**2

    PF = PreFitAnalysis(y, matv)
    out, res = PF.nnls_fit_weight()
    total_y = out * matv

    result_dict = OrderedDict()

    for i in range(len(total_list)):
        if '_L' in total_list[i] or total_list[i] in non_element:
            result_dict.update({total_list[i]: total_y[:, i]})
        else:
            result_dict.update({total_list[i] + '_K': total_y[:, i]})

    for k, v in six.iteritems(result_dict):
        if sum(v) == 0:
            del result_dict[k]
    return x, result_dict, bg
