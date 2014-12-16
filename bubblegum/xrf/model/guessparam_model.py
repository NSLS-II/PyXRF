import numpy as np
import json
import six
from collections import OrderedDict
import copy

from atom.api import Atom, Str, observe, Typed, Int

from skxray.fitting.xrf_model import (ModelSpectrum, set_range, k_line, l_line, m_line,
                                      get_linear_model, PreFitAnalysis)
from skxray.fitting.background import snip_method

__author__ = 'Li Li'


# This is not a right way to define path. To be updated
param_path = '/Users/Li/Research/X-ray/Research_work/all_code/nsls2_gui/nsls2_gui/xrf_parameter.json'


class GuessParamModel(Atom):
    param_d = Typed(object)
    param_d_perm = Typed(object)
    #incident_e = Typed(object)
    data = Typed(object)
    prefit_x = Typed(object)
    result_dict = Typed(object)
    status_dict = Typed(object)
    total_y = Typed(object)
    total_y_l = Typed(object)
    prefit_bg = Typed(object)
    data_cut = Typed(object)

    def __init__(self,
                 filepath=param_path):
        json_data = open(filepath, 'r')
        self.param_d_perm = json.load(json_data)
        #self.param_d_perm = json.load(json_data)
        #self.incident_e = self.param_d['coherent_sct_energy']['value']
        self.param_d = copy.deepcopy(self.param_d_perm)
        self.total_y = {}
        self.total_y_l = {}

    def set_data(self, data):
        """
        Parameters:
            data: numpy array
                1D spectrum intensity
        """
        self.data = np.asarray(data)

    def find_peak(self):
        """run automatic peak finding."""
        self.prefit_x, self.result_dict = pre_fit_linear(self.param_d, self.data)

        x0 = np.arange(len(self.data))
        x0, self.data_cut = set_range(self.param_d, x0, self.data)

        self.prefit_bg = get_background(self.data_cut)
        self.result_dict.update(background=self.prefit_bg)

        # save the plotting status for a given element peak
        self.status_dict = {}
        for k, v in six.iteritems(self.result_dict):
            self.status_dict.update({k: True})

    def arange_prefit_result(self):
        """
        Plot peaks from pre fit results.
        """

        # change range based on dict data
        #self.x0, self.y0 = set_range(self.para_d, self.x0, self.y0)
        self.total_y = self.result_dict.copy()

        # update plotting status based on self.status_dict
        for k, v in six.iteritems(self.status_dict):
            if v is False:
                del self.total_y[k]

        self.total_y_l = {}
        for k, v in six.iteritems(self.total_y):
            if '_L' in k:
                self.total_y_l.update({k: v})
                del self.total_y[k]

        #self.total_y = np.array(self.total_y.values())
        #self.total_y = self.total_y.transpose()

        #self.total_y_l = np.array(self.total_y_l.values())
        #self.total_y_l = self.total_y_l.transpose()


def pre_fit_linear(parameter_dict, y0):
    """
    Run prefit to get initial elements.
    """

    # read json file
    x0 = np.arange(len(y0))

    x, y = set_range(parameter_dict, x0, y0)

    # get background
    bg = snip_method(y, 0, 0.01, 0)

    y = y - bg

    element_list = k_line + l_line
    new_element = ', '.join(element_list)
    parameter_dict['non_fitting_values']['element_list'] = new_element

    total_list = element_list + ['compton', 'elastic']
    total_list = [str(v) for v in total_list]

    matv = get_linear_model(x, parameter_dict)

    x = parameter_dict['e_offset']['value'] + parameter_dict['e_linear']['value']*x + \
        parameter_dict['e_quadratic']['value'] * x**2

    PF = PreFitAnalysis(y, matv)
    out, res = PF.nnls_fit_weight()
    total_y = out * matv

    result_dict = OrderedDict()
    for i in range(len(total_list)):
        if '_L' not in total_list[i]:
            # add '_K' to the end of element name
            result_dict.update({total_list[i]+'_K': total_y[:, i]})
        else:
            result_dict.update({total_list[i]: total_y[:, i]})

    for k, v in six.iteritems(result_dict):
        if sum(v) == 0:
            del result_dict[k]
    #sorted_result = sorted(six.iteritems(result_dict), key=lambda x: x[1], reverse=True)
    #sorted_v = [v for v in six.iteritems(result_dict) if v[1] != 0]
    #total_y = [v for v in total_y if sum(v) != 0]
    return x, result_dict #total_y, sorted_v


def DoFit(parameter_dict, x, y):

    bg = get_background(y)

    c_val = 1e-2

    # first fit
    MS = ModelSpectrum(parameter_dict)
    result = MS.model_fit(x, y-bg, w=1/np.sqrt(y), maxfev=100,
                          xtol=c_val, ftol=c_val, gtol=c_val)
    #fitname = list(result1.values.keys())
    return result, bg


def get_background(y):
    return snip_method(y, 0, 0.01, 0)


