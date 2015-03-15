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
import time
import copy
import six
import os
from collections import OrderedDict

from atom.api import Atom, Str, observe, Typed, Int, List, Dict, Float
from skxray.fitting.xrf_model import (ModelSpectrum, update_parameter_dict,
                                      get_sum_area, set_parameter_bound,
                                      ParamController, set_range, get_linear_model,
                                      get_escape_peak, register_strategy)
from skxray.fitting.background import snip_method
from bubblegum.xrf.model.guessparam import dict_to_param, format_dict, fit_strategy_list
from lmfit import fit_report

import logging
logger = logging.getLogger(__name__)

#fit_strategy_list = ['fit_with_tail', 'free_more',
#                     'e_calibration', 'linear', 'adjust_element']


class Fit1D(Atom):

    file_path = Str()
    file_status = Str()
    param_dict = Dict()

    element_list = List()
    parameters = Dict()

    data = Typed(np.ndarray)
    fit_x = Typed(np.ndarray)
    fit_y = Typed(np.ndarray)
    residual = Typed(np.ndarray)
    comps = Dict()
    fit_strategy1 = Int(0)
    fit_strategy2 = Int(0)
    fit_strategy3 = Int(0)
    fit_strategy4 = Int(0)
    fit_strategy5 = Int(0)
    fit_result = Typed(object)
    data_title = Str()
    result_folder = Str()

    all_strategy = Typed(object) #Typed(OrderedDict)

    x0 = Typed(np.ndarray)
    y0 = Typed(np.ndarray)
    bg = Typed(np.ndarray)
    es_peak = Typed(np.ndarray)

    def __init__(self, *args, **kwargs):
        self.result_folder = kwargs['working_directory']
        self.all_strategy = OrderedDict()
    # def load_full_param(self):
    #     try:
    #         with open(self.file_path, 'r') as json_data:
    #             self.param_dict = json.load(json_data)
    #         self.file_status = 'Parameter file {} is loaded.'.format(self.file_path.split('/')[-1])
    #     except ValueError:
    #         self.file_status = 'Parameter file can\'t be loaded.'

    def get_new_param(self, param):
        self.param_dict = copy.deepcopy(param)
        self.element_list, self.parameters = dict_to_param(self.param_dict)

    @observe('parameters')
    def _update_param_dict(self, change):
        self.param_dict = format_dict(self.parameters, self.element_list)
        logger.info('param changed {}'.format(change['type']))

    #def update_param_dict(self):
    #    self.param_dict = format_dict(self.parameters, self.element_list)
    #    logger.info('param changed !!!')

    # @observe('file_path')
    # def update_param(self, change):
    #    self.load_full_param()

    @observe('data')
    def _update_data(self, change):
        self.data = np.asarray(self.data)

    @observe('fit_strategy1')
    def update_strategy1(self, change):
        print(change)
        self.all_strategy.update({'strategy1': change['value']})
        if change['value']:
            logger.info('Strategy at step 1 is: {}'.
                        format(fit_strategy_list[change['value']-1]))

    @observe('fit_strategy2')
    def update_strategy2(self, change):
        self.all_strategy.update({'strategy2': change['value']})
        if change['value']:
            logger.info('Strategy at step 2 is: {}'.
                        format(fit_strategy_list[change['value']-1]))

    @observe('fit_strategy3')
    def update_strategy3(self, change):
        self.all_strategy.update({'strategy3': change['value']})
        if change['value']:
            logger.info('Strategy at step 3 is: {}'.
                        format(fit_strategy_list[change['value']-1]))

    @observe('fit_strategy4')
    def update_strategy4(self, change):
        self.all_strategy.update({'strategy4': change['value']})
        if change['value']:
            logger.info('Strategy at step 4 is: {}'.
                        format(fit_strategy_list[change['value']-1]))

    @observe('fit_strategy5')
    def update_strategy5(self, change):
        self.all_strategy.update({'strategy5': change['value']})
        if change['value']:
            logger.info('Strategy at step 5 is: {}'.
                        format(fit_strategy_list[change['value']-1]))

    def update_param_with_result(self):
        update_parameter_dict(self.param_dict, self.fit_result)

    def define_range(self):
        x = np.arange(self.data.size)
        # ratio to transfer energy value back to channel value
        approx_ratio = 100
        lowv = self.param_dict['non_fitting_values']['energy_bound_low'] * approx_ratio
        highv = self.param_dict['non_fitting_values']['energy_bound_high'] * approx_ratio
        self.x0, self.y0 = set_range(x, self.data, lowv, highv)

    def get_background(self):
        self.bg = snip_method(self.y0,
                              self.param_dict['e_offset']['value'],
                              self.param_dict['e_linear']['value'],
                              self.param_dict['e_quadratic']['value'])

    def escape_peak(self):
        ratio = 0.005
        xe, ye = get_escape_peak(self.data, ratio, self.param_dict)
        lowv = self.param_dict['non_fitting_values']['energy_bound_low']
        highv = self.param_dict['non_fitting_values']['energy_bound_high']
        xe, self.es_peak = set_range(xe, ye, lowv, highv)
        logger.info('Escape peak is considered with ratio {}'.format(ratio))
        # align to the same length
        if self.y0.size > self.es_peak.size:
            temp = self.es_peak
            self.es_peak = np.zeros(len(self.y0.size))
            self.es_peak[:temp.size] = temp
        else:
            self.es_peak = self.es_peak[:self.y0.size]

    def fit_data(self, x0, y0, c_val=1e-2, fit_num=100, c_weight=1e3):
        MS = ModelSpectrum(self.param_dict)
        MS.model_spectrum()

        result = MS.model_fit(x0, y0, w=1/np.sqrt(c_weight+y0), maxfev=fit_num,
                              xtol=c_val, ftol=c_val, gtol=c_val)

        comps = result.eval_components(x=x0)
        self.combine_lines(comps)

        xnew = (result.values['e_offset'] + result.values['e_linear'] * x0 +
                result.values['e_quadratic'] * x0**2)
        self.fit_x = xnew
        self.fit_y = result.best_fit
        self.fit_result = result
        self.residual = self.fit_y - y0

    def fit_multiple(self):

        self.define_range()
        self.get_background()
        #self.escape_peak()

        #PC = ParamController(self.param_dict)
        #self.param_dict = PC.new_parameter

        y0 = self.y0 - self.bg #- self.es_peak

        t0 = time.time()
        logger.warning('Start fitting!')
        for k, v in six.iteritems(self.all_strategy):
            if v:
                stra_name = fit_strategy_list[v-1]
                logger.info('Fit with {}: {}'.format(k, stra_name))
                strategy = extract_strategy(self.param_dict, stra_name)
                register_strategy(stra_name, strategy)
                set_parameter_bound(self.param_dict, stra_name)
                self.comps.clear()
                self.fit_data(self.x0, y0)
                self.update_param_with_result()

        self.fit_y += self.bg #+ self.es_peak

        t1 = time.time()
        logger.warning('Time used for fitting is : {}'.format(t1-t0))
        self.save_result()

    def combine_lines(self, comps):
        """
        Combine results for different lines of the same element.
        And also add background.

        Parameters
        ----------
        comps : dict
            output results from lmfit
        """
        for e in self.param_dict['non_fitting_values']['element_list'].split(','):
            e = e.strip(' ')
            if '_' in e:
                e_temp = e.split('_')[0]
            else:
                e_temp = e
            intensity = 0
            for k, v in six.iteritems(comps):
                if e_temp in k:
                    del comps[k]
                    intensity += v
            self.comps[e] = intensity
        self.comps.update(comps)

        # add background
        self.comps.update({'background': self.bg})

        self.comps['elastic'] = self.comps['elastic_']
        del self.comps['elastic_']

    def save_result(self, fname=None):
        """
        Parameters
        ----------
        fname : str, optional
            name of output file
        """
        if not fname:
            fname = self.data_title+'_out.txt'
        filepath = os.path.join(self.result_folder, fname)
        with open(filepath, 'w') as myfile:
            myfile.write(fit_report(self.fit_result, sort_pars=True))
            logger.warning('Results are saved to {}'.format(filepath))


def extract_strategy(param, name):
    """
    Extract given strategy from param dict.

    Parameters
    ----------
    param : dict
        saving all parameters
    name : str
        strategy name
    """
    return {k: v[name] for k, v in six.iteritems(param) if k != 'non_fitting_values'}






# to be removed
class Param(Atom):
    name = Str()
    value = Float(0.0)
    min = Float(0.0)
    max = Float(0.0)
    bound_type = Str()
    free_more = Str()
    fit_with_tail = Str()
    adjust_element = Str()
    e_calibration = Str()
    linear = Str()

    def __init__(self):
        self.value = 0.0
        self.min = 0.0
        self.max = 10.0
