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
import json
import six
import os

from atom.api import Atom, Str, observe, Typed, Int, List, Dict, Float

from skxray.fitting.xrf_model import (k_line, l_line, m_line)
from skxray.constants.api import XrfElement as Element

from skxray.fitting.xrf_model import (ModelSpectrum, update_parameter_dict,
                                      get_sum_area, set_parameter_bound,
                                      ParamController, set_range, get_linear_model,
                                      PreFitAnalysis, k_line, l_line)
from skxray.fitting.background import snip_method
from lmfit import fit_report

import logging
logger = logging.getLogger(__name__)

fit_strategy_list = ['fit_with_tail', 'free_more',
                     'e_calibration', 'linear', 'adjust_element']


class Fit1D(Atom):

    file_path = Str()
    file_status = Str()
    param_dict = Dict()
    data = Typed(np.ndarray)
    fit_x = Typed(np.ndarray)
    fit_y = Typed(np.ndarray)
    residual = Typed(np.ndarray)
    comps = Dict()
    fit_strategy1 = Int(0)
    fit_strategy2 = Int(0)
    fit_strategy3 = Int(0)
    fit_result = Typed(object)
    strategy_list = List()
    data_title = Str()
    result_folder = Str()

    def __init__(self, *args, **kwargs):
        self.result_folder = kwargs['working_directory']
        self.strategy_list = fit_strategy_list

    def load_full_param(self):
        try:
            with open(self.file_path, 'r') as json_data:
                self.param_dict = json.load(json_data)
            self.file_status = 'Parameter file {} is loaded.'.format(self.file_path.split('/')[-1])
        except ValueError:
            self.file_status = 'Parameter file can\'t be loaded.'

    @observe('file_path')
    def update_param(self, change):
        self.load_full_param()

    @observe('fit_strategy1')
    def update_strategy1(self, change):
        if change['value'] == 0:
            return
        logger.info('Strategy at step 1 is: {}'.
                    format(self.strategy_list[change['value']-1]))
        set_parameter_bound(self.param_dict,
                            self.strategy_list[self.fit_strategy1-1])

    @observe('fit_strategy2')
    def update_strategy2(self, change):
        if change['value'] == 0:
            return
        logger.info('Strategy at step 2 is: {}'.
                    format(self.strategy_list[change['value']-1]))

    @observe('fit_strategy3')
    def update_strategy3(self, change):
        if change['value'] == 0:
            return
        logger.info('Strategy at step 3 is: {}'.
                    format(self.strategy_list[change['value']-1]))

    def update_param_with_result(self):
        update_parameter_dict(self.param_dict, self.fit_result)

    def fit_data(self):
        c_val = 1e-2
        self.data = np.asarray(self.data)
        x = np.arange(self.data.size)
        x0, y0 = set_range(self.param_dict, x, self.data)

        # get background
        bg = snip_method(y0,
                         self.param_dict['e_offset']['value'],
                         self.param_dict['e_linear']['value'],
                         self.param_dict['e_quadratic']['value'])

        MS = ModelSpectrum(self.param_dict)
        MS.model_spectrum()

        result = MS.model_fit(x0, y0-bg, w=1/np.sqrt(y0), maxfev=100,
                              xtol=c_val, ftol=c_val, gtol=c_val)

        comps = result.eval_components(x=x0)
        self.combine_lines(comps)
        self.comps.update({'background': bg})

        xnew = result.values['e_offset'] + result.values['e_linear'] * x0 +\
               result.values['e_quadratic'] * x0**2
        self.fit_x = xnew
        self.fit_y = result.best_fit + bg
        self.fit_result = result
        self.residual = self.fit_y - y0

    def fit_multiple(self):
        t0 = time.time()
        logger.warning('Start fitting!')
        if self.fit_strategy1 != 0 and \
            self.fit_strategy2+self.fit_strategy3 == 0:
            logger.info('Fit with 1 strategy')
            self.fit_data()

        elif self.fit_strategy1*self.fit_strategy2 != 0 \
            and self.fit_strategy3 == 0:
            logger.info('Fit with 2 strategies')
            self.fit_data()
            self.update_param_with_result()
            set_parameter_bound(self.param_dict,
                                self.strategy_list[self.fit_strategy2-1])
            self.fit_data()

        elif self.fit_strategy1*self.fit_strategy2*self.fit_strategy3 != 0:
            logger.info('Fit with 3 strategies')
            # first
            self.fit_data()
            # second
            self.update_param_with_result()
            set_parameter_bound(self.param_dict,
                                self.strategy_list[self.fit_strategy2-1])
            self.fit_data()
            # thrid
            self.update_param_with_result()
            set_parameter_bound(self.param_dict,
                                self.strategy_list[self.fit_strategy3-1])
            self.fit_data()

        t1 = time.time()
        logger.warning('Time used for fitting is : {}'.format(t1-t0))
        self.save_result()

    def combine_lines(self, comps):
        """
        Combine results for different lines of the same element.

        Parameters
        ----------
        comps : dict
            output results from lmfit
        """
        self.comps.clear()

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
