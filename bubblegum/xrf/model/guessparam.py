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
import os

import logging
logger = logging.getLogger(__name__)

from atom.api import (Atom, Str, observe, Typed,
                      Int, Dict, List, Float, Enum, Bool)

from skxray.fitting.background import snip_method
from skxray.constants.api import XrfElement as Element
from skxray.fitting.xrf_model import (ModelSpectrum, ParamController,
                                      set_range, k_line, l_line, m_line,
                                      get_linear_model, PreFitAnalysis)

import logging
logger = logging.getLogger(__name__)


bound_options = ['none', 'lohi', 'fixed', 'lo', 'hi']


class Parameter(Atom):
    # todo make sure that these are the only valid bound types
    bound_type = Enum(*bound_options)
    min = Float(-np.inf)
    max = Float(np.inf)
    value = Float()
    default_value = Float()
    fit_with_tail = Enum(*bound_options)
    free_more = Enum(*bound_options)
    adjust_element = Enum(*bound_options)
    e_calibration = Enum(*bound_options)
    linear = Enum(*bound_options)
    name = Str()
    description = Str()
    tool_tip = Str()

    @observe('name', 'bound_type', 'min', 'max', 'value', 'default_value')
    def update_displayed_name(self, changed):
        pass
    #    print(changed)

    def __repr__(self):
        return ("Parameter(bound_type={}, min={}, max={}, value={}, "
                "default={}, free_more={}, adjust_element={}, "
                "e_calibration={}, linear={}, description={}, "
                "toop_tip={}".format(
            self.bound_type, self.min, self.max, self.value, self.default_value,
            self.free_more, self.adjust_element, self.e_calibration,
            self.linear, self.description, self.tool_tip))

    def to_dict(self):
        return {
            'bound_type': self.bound_type,
            'min': self.min,
            'max': self.max,
            'value': self.value,
            'default_value': self.default_value,
            'fit_with_tail': self.fit_with_tail,
            'free_more': self.free_more,
            'adjust_element': self.adjust_element,
            'e_calibration': self.e_calibration,
            'linear': self.linear,
            'name': self.name,
            'description': self.description,
            'tool_tip': self.tool_tip,
        }


def format_dict(parameter_object_dict, element_list):
    """
    Format the dictionary that scikit-xray expects.

    Parameters
    ----------
    parameter_object_dict : dict
    element_list : list
        Need to be transferred to str first, then save it to dict
    """
    param_dict = {key: value.to_dict() for key, value
                  in six.iteritems(parameter_object_dict)}
    elo = param_dict.pop('energy_bound_low')['value']
    ehi = param_dict.pop('energy_bound_high')['value']

    non_fitting_values = {'non_fitting_values': {
        'energy_bound_low': elo,
        'energy_bound_high': ehi,
        'element_list': ', '.join(element_list)
    }}
    param_dict.update(non_fitting_values)

    return param_dict


def dict_to_param(param_dict):
    """
    Transfer param dict to parameter object.

    Parameters
    param_dict : dict
        fitting parameter
    """
    temp_parameters = copy.deepcopy(param_dict)
    param = {}

    non_fitting_values = temp_parameters.pop('non_fitting_values')
    element_list = non_fitting_values.pop('element_list')
    if not isinstance(element_list, list):
        element_list = [e.strip(' ') for e in element_list.split(',')]
    #self.element_list = element_list

    elo = non_fitting_values.pop('energy_bound_low')
    ehi = non_fitting_values.pop('energy_bound_high')
    param = {
        'energy_bound_low': Parameter(value=elo,
                                      default_value=elo,
                                      description='E low limit [keV]'),
        'energy_bound_high': Parameter(value=ehi,
                                       default_value=ehi,
                                       description='E high limit [keV]')
    }

    for param_name, param_dict in six.iteritems(temp_parameters):
        param.update({
            param_name: Parameter(**param_dict)
        })
    return element_list, param


class PreFitStatus(Atom):
    """
    Data structure for pre fit analysis.

    Attributes
    ----------
    z : str
        z number of element
    spectrum : array
        spectrum of given element
    status : bool
        True as plot is visible
    stat_copy : bool
        copy of status
    maxv : float
        max value of a spectrum
    norm : float
        norm value respect to the strongest peak
    lbd_stat : bool
        define plotting status under a threshold value
    """
    z = Str()
    spectrum = Typed(np.ndarray)
    status = Bool(True)
    stat_copy = Bool(True)
    maxv = Float()
    norm = Float()
    lbd_stat = Bool()


class GuessParamModel(Atom):
    """
    This is auto fit model to guess the initial parameters.

    Attributes
    ----------
    parameters : `atom.Dict`
        A list of `Parameter` objects, subclassed from the `Atom` base class.
        These `Parameter` objects hold all relevant xrf information.
    data : array
        1D array of spectrum
    prefit_x : array
        xX axis with range defined by low and high limits.
    result_dict : dict
        Save all the auto fitting results for each element.
        It is a dictionary of object PreFitStatus.
    param_d : dict
        Parameters can be transferred into this dictionary.
    param_new : dict
        More information are saved, such as element position and width.
    total_y : dict
        Results from k lines
    total_y_l : dict
        Results from l lines
    total_y_m : dict
        Results from l lines
    e_list : str
        All elements used for fitting.
    file_path : str
        The path where file is saved.
    element_list : list
    """
    default_parameters = Dict()
    parameters = Dict() #Typed(OrderedDict) #OrderedDict()
    data = Typed(object)
    prefit_x = Typed(object)

    prefit_dict = Typed(OrderedDict)
    result_dict = Typed(OrderedDict)
    result_dict_names = List()

    param_d = Dict()
    param_new = Dict()
    total_y = Dict()
    total_y_l = Dict()
    total_y_m = Dict()
    e_list = Str()
    #save_file = Str()

    result_folder = Str()
    file_path = Str()

    #result_dict = Dict(key=str, value=Parameter)#OrderedDict()
    element_list = List()

    data_sets = Typed(OrderedDict)
    file_opt = Int()

    def __init__(self, *args, **kwargs):
        try:
            self.default_parameters = kwargs['default_parameters']
            self.element_list, self.parameters = dict_to_param(self.default_parameters)
            #self.get_param(default_parameters)
        except ValueError:
            logger.info('No default parameter files are chosen.')

        self.total_y_l = {}
        self.prefit_dict = OrderedDict()
        self.result_dict = OrderedDict()
        self.result_folder = kwargs['working_directory']

    def restore_default_param(self):
        self.element_list, self.parameters = dict_to_param(self.default_parameters)
        logger.info('Use default parameters.')

    def get_new_param(self, param_path):
        """
        Update parameters if new param_path is given.

        Parameters
        ----------
        param_path : str
        """
        with open(param_path, 'r') as json_data:
            new_param = json.load(json_data)
        self.element_list, self.parameters = dict_to_param(new_param)
        #self.element_list, self.parameters = self.get_param(new_param)

    def get_param(self, param_dict):
        """
        Transfer dict into the type of parameter class.

        Parameters
        ----------
        param_dict : dict
            Dictionary of parameters used for fitting.
        """
        self.element_list, self.parameters = dict_to_param(param_dict)

    def to_dict(self, param_obj, element_list=None):

        if not element_list:
            element_list = self.element_list
        return format_dict(param_obj, element_list)

    @observe('file_opt')
    def choose_file(self, change):
        if self.file_opt == 0:
            return
        names = self.data_sets.keys()
        self.data = self.data_sets[names[self.file_opt-1]].get_sum()

    def find_peak(self, threshv=0.1):
        """
        Run automatic peak finding, and save results as dict of object.
        """
        param_dict = format_dict(self.parameters, self.element_list)
        self.prefit_x, out_dict = pre_fit_linear(self.data, param_dict)

        max_dict = reduce(max, map(np.max, six.itervalues(out_dict)))

        self.prefit_dict.clear()
        for k, v in six.iteritems(out_dict):
            lb_check = bool(100*(np.max(v)/max_dict) > threshv)
            ps = PreFitStatus(z=get_Z(k), spectrum=v,
                              status=True, stat_copy=True,
                              maxv=np.max(v), norm=(np.max(v)/max_dict)*100,
                              lbd_stat=lb_check)
            self.prefit_dict.update({k: ps})

        #self.result_assumbler(out_dict)
        logger.info('The elements found from prefit {}'.format(self.prefit_dict.keys()))
        self.add_to_dict(self.prefit_dict)
        self.result_dict_names = self.result_dict.keys()

    def add_to_dict(self, dictv):
        """
        Summarize auto fitting results into a dict.
        """
        #max_dict = reduce(max, map(np.max, six.itervalues(dictv)))

        self.result_dict.clear()

        # for k, v in six.iteritems(dictv):
        #     lb_check = bool(100*(np.max(v)/max_dict) > threshv)
        #     ps = PreFitStatus(z=get_Z(k), spectrum=v,
        #                       status=True, stat_copy=True,
        #                       maxv=np.max(v), norm=(np.max(v)/max_dict)*100,
        #                      lbd_stat=lb_check)
        self.result_dict.update(dictv)

    def delete_item(self, k):
        try:
            del self.result_dict[k]
        except KeyError, e:
            logger.info(e)


    @observe('result_dict')
    def update_dict(self, change):
        if change['type'] == 'create':
            return
        print('result dict change: {}'.format(change['type']))
        self.result_dict_names = self.result_dict.keys()
        logger.info('elements to be used for fitting: {}'.format(self.result_dict.keys()))

    def get_activated_element(self):
        """
        Select elements from pre fit.
        """
        e = [k for (k, v) in six.iteritems(self.result_dict) if v.status and len(k)<=4]
        self.e_list = ', '.join(e)

    def save_elist(self):
        """
        Save selected list to param dict.
        """
        if len(self.e_list):
            elist_k = [v[:-2] for v in self.e_list.split(', ') if '_K' in v]
            elist_l_m = [v for v in self.e_list.split(', ') if '_K' not in v]
            elist = elist_k + elist_l_m
            logger.info('Input elements for fitting are {}'.format(elist))
        else:
            elist = self.element_list
            logger.warning('No input elements are given in auto fitting. '
                           'Use default elements: {}.'.format(elist))
        self.param_d = format_dict(self.parameters, elist)

    def create_full_param(self,
                          peak_std=0.07, peak_height=500.0):
        """
        Extend the param to full param dict with detailed elements
        information, and assign initial values from pre fit.

        Parameters
        ----------
        peak_std : float
            approximated std for element peak.
        peak_height : float
            initial value for element peak height
        """

        self.save_elist()

        PC = ParamController(self.param_d)
        PC.create_full_param()
        self.param_new = PC.new_parameter
        factor_to_area = np.sqrt(2*np.pi)*peak_std*0.5

        # update according to pre fit results
        if len(self.result_dict):
            for e in self.e_list.split(','):
                e = e.strip(' ')
                zname = e.split('_')[0]
                for k, v in six.iteritems(self.param_new):
                    if zname in k and 'area' in k:
                        if self.result_dict.has_key(e):
                            v['value'] = self.result_dict[e].maxv*factor_to_area
                        else:
                            v['value'] = peak_height*factor_to_area
            self.param_new['compton_amplitude']['value'] = self.result_dict['compton'].maxv*factor_to_area
            self.param_new['coherent_sct_amplitude']['value'] = self.result_dict['elastic'].maxv*factor_to_area

    def data_for_plot(self):
        """
        Save data in terms of K, L, M lines for plot.
        """
        self.total_y = {}
        self.total_y_l = {}
        self.total_y_m = {}
        new_dict = {k: v for (k, v) in six.iteritems(self.result_dict) if v.status}
        for k, v in six.iteritems(new_dict):
            if 'K' in k:
                self.total_y[k] = self.result_dict[k].spectrum
            elif 'L' in k:
                self.total_y_l[k] = self.result_dict[k].spectrum
            elif 'M' in k:
                self.total_y_m[k] = self.result_dict[k].spectrum
            else:
                self.total_y[k] = self.result_dict[k].spectrum

    def save(self, fname='param_default1.json'):
        """
        Save full param dict into a file at result directory.
        The name of the file is predefined.

        Parameters
        ----------
        fname : str, optional
            file name to save updated parameters
        """
        #self.save_elist()
        #self.create_full_param()
        fpath = os.path.join(self.result_folder, fname)
        with open(fpath, 'w') as outfile:
            json.dump(self.param_new, outfile,
                      sort_keys=True, indent=4)

    def save_as(self):
        """
        Save full param dict into a file.
        """
        #self.save_elist()
        #self.create_full_param()
        with open(self.file_path, 'w') as outfile:
            json.dump(self.param_new, outfile,
                      sort_keys=True, indent=4)

    def read_pre_saved(self, fname='param_default1.json'):
        """This is a bad idea."""

        fpath = os.path.join(self.result_folder, fname)
        with open(fpath, 'r') as infile:
            data = json.load(infile)
        return data


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
    """

    # Need to use deepcopy here to avoid unexpected change on parameter dict
    fitting_parameters = copy.deepcopy(fitting_parameters)

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

    e_select, matv = get_linear_model(x, fitting_parameters)

    total_list = e_select + non_element
    total_list = [str(v) for v in total_list]

    x = fitting_parameters['e_offset']['value'] + fitting_parameters['e_linear']['value']*x + \
        fitting_parameters['e_quadratic']['value'] * x**2

    PF = PreFitAnalysis(y, matv)
    out, res = PF.nnls_fit_weight()
    total_y = out * matv

    # use ordered dict
    result_dict = OrderedDict()

    for i in range(len(total_list)):
        if np.sum(total_y[:, i]) == 0:
            continue
        if '_L' in total_list[i] or '_M' in total_list[i] \
                or total_list[i] in non_element:
            result_dict.update({total_list[i]: total_y[:, i]})
        else:
            result_dict.update({total_list[i] + '_K': total_y[:, i]})
    result_dict.update(background=bg)
    return x, result_dict


def get_Z(ename):
    """
    Return element's Z number.

    Parameters
    ----------
    ename : str
        element name

    Returns
    -------
    int or None
        element Z number

    """

    strip_line = lambda ename: ename.split('_')[0]

    non_element = ['compton', 'elastic', 'background']
    if ename in non_element:
        return '-'
    else:
        e = Element(strip_line(ename))
        return str(e.Z)
