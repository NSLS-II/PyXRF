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

from atom.api import Atom, Str, observe, Typed, Int, List, Dict

from skxray.fitting.xrf_model import (k_line, l_line, m_line)
from skxray.constants.api import XrfElement as Element

from skxray.fitting.xrf_model import (ModelSpectrum, update_parameter_dict,
                                      get_sum_area, set_parameter_bound,
                                      ParamController, set_range, get_linear_model,
                                      PreFitAnalysis, k_line, l_line)
from skxray.fitting.background import snip_method
from lmfit import fit_report

out_folder = '/Users/Li/Research/X-ray/Research_work/all_code/nsls2_gui/nsls2_gui'

class Fit1D(Atom):

    file_path = Str()
    file_status = Str()
    param_dict = Dict()
    data = Typed(np.ndarray)
    fit_x = Typed(np.ndarray)
    fit_y = Typed(np.ndarray)
    comps = Dict()
    fit_strategy1 = Int(0)
    fit_strategy2 = Int(0)
    fit_strategy3 = Int(0)
    fit_result = Typed(object)
    strategy_list = List()
    data_title = Str()
    result_folder = Str(out_folder)

    def __init__(self):
        self.strategy_list = ['fit_with_tail', 'free_more', 'e_calibration', 'linear']

    def get_full_param(self):
        try:
            with open(self.file_path, 'r') as json_data:
                self.param_dict = json.load(json_data)
            self.file_status = 'Parameter file {} is loaded.'.format(self.file_path.split('/')[-1])
        except ValueError:
            self.file_status = 'Parameter file can\'t be loaded.'

    @observe('file_path')
    def update_param(self, change):
        self.get_full_param()

    @observe('fit_strategy1')
    def update_strategy1(self, change):
        if change['value'] == 0:
            return
        set_parameter_bound(self.param_dict,
                            self.strategy_list[self.fit_strategy1-1])

    @observe('fit_strategy2')
    def update_strategy2(self, change):
        print('strateg2 changed')

    @observe('fit_strategy3')
    def update_strategy3(self, change):
        print('strateg3 changed')

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

        print('Start fitting!')
        t0 = time.time()
        result = MS.model_fit(x0, y0-bg, w=1/np.sqrt(y0), maxfev=100,
                               xtol=c_val, ftol=c_val, gtol=c_val)
        t1 = time.time()
        print('time used: {}'.format(t1-t0))

        comps = result.eval_components(x=x0)
        self.combine_results(comps)

        xnew = result.values['e_offset'] + result.values['e_linear'] * x0 +\
               result.values['e_quadratic'] * x0**2
        self.fit_x = xnew
        self.fit_y = result.best_fit + bg
        self.fit_result = result

        #print(result1.fit_report())

    def fit_multiple(self):
        if self.fit_strategy1 != 0 and \
            self.fit_strategy2+self.fit_strategy3 == 0:
            print('fit with 1 strategy')
            self.fit_data()

        elif self.fit_strategy1*self.fit_strategy2 != 0 \
            and self.fit_strategy3 == 0:
            print('fit with 2 strategies')
            self.fit_data()
            self.update_param_with_result()
            set_parameter_bound(self.param_dict,
                                self.strategy_list[self.fit_strategy2-1])
            self.fit_data()

        elif self.fit_strategy1*self.fit_strategy2*self.fit_strategy3 != 0:
            print('fit with 3 strategies')
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

        self.save_result()

    def combine_results(self, comps):
        """
        Combine results for the same element.
        Parameters
        ----------
        comps : dict
            output results from lmfit
        """
        for e in self.param_dict['non_fitting_values']['element_list'].split(', '):
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

    def save_result(self):
        filepath = os.path.join(self.result_folder,
                                self.data_title+'_out.txt')
        with open(filepath, 'w') as myfile:
            myfile.write(fit_report(self.fit_result, sort_pars=True))
            print('Results are save to {}'.format(filepath))
