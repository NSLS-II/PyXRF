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

from atom.api import Atom, Str, observe, Typed, Int, Dict, List

from skxray.fitting.xrf_model import (ModelSpectrum, set_range, k_line, l_line, m_line,
                                      get_linear_model, PreFitAnalysis)
from skxray.fitting.background import snip_method




class GuessParamModel(Atom):
    """
    This is auto fit model to guess the initial parameters.

    Attributes
    ----------
    param_d : dict
        The fitting parameters
    param_d_perm : dict
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
    parameter_file : str
        Path to parameter file
    param_status : str
        Loading status of parameter file
    """
    param_d = Dict()
    param_d_perm = Dict()
    data = Typed(object)
    prefit_x = Typed(object)
    result_dict = Typed(object)
    status_dict = Dict(value=bool, key=str)
    total_y = Typed(object)
    total_y_l = Typed(object)
    parameter_file = Str()
    param_status = Str('Use default parameter file.')

    def __init__(self, parameter_file, *args, **kwargs):
        self.parameter_file = parameter_file
        self.total_y_l = {}

    def get_param(self):
        try:
            with open(self.parameter_file, 'r') as json_data:
                self.param_d_perm = json.load(json_data)
            self.param_status = 'Parameter file {} is loaded.'.format(
                self.parameter_file.split('/')[-1])
            self.param_d = copy.deepcopy(self.param_d_perm)
        except ValueError:
            self.param_status = 'Parameter file can\'t be loaded.'

    @observe('parameter_file')
    def set_param(self, changed):
        self.get_param()

    def set_data(self, data):
        """
        Parameters
        ----------
        data : numpy array
            1D array of spectrum intensity
        """
        self.data = np.asarray(data)

    def save_param(self, param):
        self.param_d = param

    def find_peak(self):
        """run automatic peak finding."""
        #for k,v in six.iteritems(self.param_d):
        #    print('{}:{}'.format(k,v))
        self.prefit_x, self.result_dict, bg = pre_fit_linear(self.param_d,
                                                             self.data)

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


def pre_fit_linear(parameter_dict, y0):
    """
    Run prefit to get initial elements.

    Parameters
    ----------
    parameter_dict : dict
        Fitting parameters
    y0 : array
        Spectrum intensity

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
    x, y = set_range(parameter_dict, x0, y0)

    # get background
    bg = snip_method(y, parameter_dict['e_offset']['value'],
                     parameter_dict['e_linear']['value'],
                     parameter_dict['e_quadratic']['value'])

    y = y - bg

    element_list = k_line + l_line + m_line
    new_element = ', '.join(element_list)
    parameter_dict['non_fitting_values']['element_list'] = new_element

    non_element = ['compton', 'elastic']
    total_list = element_list + non_element
    total_list = [str(v) for v in total_list]

    matv = get_linear_model(x, parameter_dict)

    x = parameter_dict['e_offset']['value'] + parameter_dict['e_linear']['value']*x + \
        parameter_dict['e_quadratic']['value'] * x**2

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
