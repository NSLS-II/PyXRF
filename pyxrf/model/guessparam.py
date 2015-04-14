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
import json
from collections import OrderedDict
import copy
import os

from atom.api import (Atom, Str, observe, Typed,
                      Int, Dict, List, Float, Enum, Bool)

from skxray.fitting.background import snip_method
from skxray.constants.api import XrfElement as Element
from skxray.fitting.xrf_model import (ModelSpectrum, ParamController,
                                      compute_escape_peak, trim,
                                      construct_linear_model, linear_spectrum_fitting)

import logging
logger = logging.getLogger(__name__)


bound_options = ['none', 'lohi', 'fixed', 'lo', 'hi']
fit_strategy_list = ['fit_with_tail', 'free_more',
                     'e_calibration', 'linear',
                     'adjust_element1', 'adjust_element2', 'adjust_element3']


class Parameter(Atom):
    # todo make sure that these are the only valid bound types
    bound_type = Enum(*bound_options)
    min = Float(-np.inf)
    max = Float(np.inf)
    value = Float()
    default_value = Float()
    fit_with_tail = Enum(*bound_options)
    free_more = Enum(*bound_options)
    adjust_element1 = Enum(*bound_options)
    adjust_element2 = Enum(*bound_options)
    adjust_element3 = Enum(*bound_options)
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
                "default={}, free_more={}, adjust_element1={}, "
                "adjust_element2={}, adjust_element3={}, "
                "e_calibration={}, linear={}, description={}, "
                "toop_tip={}".format(
            self.bound_type, self.min, self.max, self.value, self.default_value,
            self.free_more, self.adjust_element1, self.adjust_element2,
            self.adjust_element3, self.e_calibration,
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
            'adjust_element1': self.adjust_element1,
            'adjust_element2': self.adjust_element2,
            'adjust_element3': self.adjust_element3,
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
        if 'default_value' in param_dict:
            param.update({param_name: Parameter(**param_dict)})
        else:
            param.update({
                param_name: Parameter(default_value=param_dict['value'],
                                      **param_dict)
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
    energy = Str()
    area = Float()
    spectrum = Typed(np.ndarray)
    status = Bool(False)
    stat_copy = Bool(False)
    maxv = Float()
    norm = Float()
    lbd_stat = Bool(False)


class ElementController(object):
    """
    This class performs basic ways to rank elements, show elements,
    calculate normed intensity, and etc.
    """

    def __init__(self):
        self.element_dict = OrderedDict()

    def delete_item(self, k):
        try:
            del self.element_dict[k]
            self.update_norm()
            logger.info('Item {} is deleted.'.format(k))
        except KeyError, e:
            logger.info(e)

    def order(self, option='z'):
        """
        Order dict in different ways.
        """
        if option == 'z':
            self.element_dict = OrderedDict(sorted(six.iteritems(self.element_dict),
                                                   key=lambda t: t[1].z))
        elif option == 'energy':
            self.element_dict = OrderedDict(sorted(six.iteritems(self.element_dict),
                                                   key=lambda t: t[1].energy))
        elif option == 'name':
            self.element_dict = OrderedDict(sorted(six.iteritems(self.element_dict),
                                                   key=lambda t: t[0]))
        elif option == 'maxv':
            self.element_dict = OrderedDict(sorted(six.iteritems(self.element_dict),
                                                   key=lambda t: t[1].maxv, reverse=True))

    def add_to_dict(self, dictv):
        self.element_dict.update(dictv)
        self.update_norm()

    def update_norm(self, threshv=0.1):
        """
        Calculate the norm intensity for each element peak.

        Parameters
        ----------
        threshv : float
            No value is shown when smaller than the shreshold value
        """
        #max_dict = reduce(max, map(np.max, six.itervalues(self.element_dict)))
        max_dict = np.max(np.array([v.maxv for v
                                    in six.itervalues(self.element_dict)]))

        for v in six.itervalues(self.element_dict):
            v.norm = v.maxv/max_dict*100
            v.lbd_stat = bool(v.norm > threshv)

    def delete_all(self):
        self.element_dict.clear()

    def get_element_list(self):
        current_elements = [v for v
                            in six.iterkeys(self.element_dict) if v.lower() != v]
        logger.info('Current Elements for fitting are {}'.format(current_elements))
        return current_elements

    def update_peak_ratio(self):
        """
        In case users change the max value.
        """
        for v in six.itervalues(self.element_dict):
            v.maxv = np.around(v.maxv, 1)
            v.spectrum = v.spectrum*v.maxv/np.max(v.spectrum)
        self.update_norm()

    def turn_on_all(self, option=True):
        """
        Set plotting status on for all lines.
        """
        if option:
            _plot = option
        else:
            _plot = False
        for v in six.itervalues(self.element_dict):
            v.status = _plot


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
    #parameters = Dict() #Typed(OrderedDict) #OrderedDict()
    data = Typed(object)
    prefit_x = Typed(object)

    result_dict = Typed(object) #Typed(OrderedDict)
    result_dict_names = List()

    #param_d = Dict()
    param_new = Dict()
    total_y = Dict()
    total_y_l = Dict()
    total_y_m = Dict()
    e_name = Str()
    add_element_intensity = Float()
    #save_file = Str()

    result_folder = Str()
    #file_path = Str()

    element_list = List()

    data_sets = Typed(OrderedDict)
    file_opt = Int()
    data_all = Typed(np.ndarray)

    EC = Typed(object)

    x0 = Typed(np.ndarray)
    y0 = Typed(np.ndarray)

    def __init__(self, **kwargs):
        try:
            self.default_parameters = kwargs['default_parameters']
            #self.element_list, self.parameters = dict_to_param(self.default_parameters)
            self.param_new = copy.deepcopy(self.default_parameters)
            self.element_list = get_element(self.param_new)
            #self.get_param(default_parameters)
        except ValueError:
            logger.info('No default parameter files are chosen.')
        self.result_folder = kwargs['working_directory']
        self.EC = ElementController()

    def get_new_param(self, param_path):
        """
        Update parameters if new param_path is given.

        Parameters
        ----------
        param_path : str
            path to save the file
        """
        with open(param_path, 'r') as json_data:
            self.param_new = json.load(json_data)
        self.element_list = get_element(self.param_new)
        self.EC.delete_all()
        self.define_range()
        self.create_spectrum_from_file(self.param_new, self.element_list)
        logger.info('Elements read from file are: {}'.format(self.element_list))

    def create_spectrum_from_file(self, param_dict, elemental_lines):
        """
        Create spectrum profile with given param dict from file.

        Parameters
        ----------
        param_dict : dict
            dict obtained from file
        elemental_lines : list
            e.g., ['Na_K', Mg_K', 'Pt_M'] refers to the
            K lines of Sodium, the K lines of Magnesium, and the M
            lines of Platinum
        """
        self.prefit_x, pre_dict, area_dict = calculate_profile(self.data,
                                                               param_dict,
                                                               elemental_lines)

        temp_dict = OrderedDict()
        for e in six.iterkeys(pre_dict):
            ename = e.split('_')[0]
            print(ename)
            if ename in ['background', 'escape']:
                spectrum = pre_dict[e]
                area = np.sum(spectrum)
                ps = PreFitStatus(z=get_Z(ename), energy=get_energy(e),
                                  area=area, spectrum=spectrum,
                                  maxv=np.around(np.max(spectrum), 1),
                                  norm=-1, lbd_stat=False)
                temp_dict[e] = ps
            else:
                for k, v in six.iteritems(param_dict):

                    if ename in k and 'area' in k:
                        spectrum = pre_dict[e]
                        area = area_dict[e]

                    elif ename == 'compton' and k == 'compton_amplitude':
                        spectrum = pre_dict[e]
                        area = area_dict[e]

                    elif ename == 'elastic' and k == 'coherent_sct_amplitude':
                        spectrum = pre_dict[e]
                        area = area_dict[e]

                    else:
                        continue

                    ps = PreFitStatus(z=get_Z(ename), energy=get_energy(e),
                                      area=area, spectrum=spectrum,
                                      maxv=np.around(np.max(spectrum), 1),
                                      norm=-1, lbd_stat=False)

                    temp_dict[e] = ps
        self.EC.add_to_dict(temp_dict)

    @observe('file_opt')
    def choose_file(self, change):
        if self.file_opt == 0:
            return
        names = self.data_sets.keys()
        self.data = self.data_sets[names[self.file_opt-1]].get_sum()
        self.define_range()
        #self.data_all = self.data_sets[names[self.file_opt-1]].raw_data

    def define_range(self):
        """
        Cut x range according to values define in param_dict.
        """
        lowv = self.param_new['non_fitting_values']['energy_bound_low']['value']
        highv = self.param_new['non_fitting_values']['energy_bound_high']['value']
        self.x0, self.y0 = define_range(self.data, lowv, highv)

    def manual_input(self):
        default_area = 1e5
        logger.info('{} peak is added'.format(self.e_name))

        if self.e_name == 'escape':
            self.param_new['non_fitting_values']['escape_ratio'] = (self.add_element_intensity
                                                                    / np.max(self.y0))
            es_peak = trim_escape_peak(self.data, self.param_new,
                                       self.prefit_x.size)
            ps = PreFitStatus(z=get_Z(self.e_name),
                              energy=get_energy(self.e_name),
                              area=np.sum(es_peak),
                              spectrum=es_peak,
                              maxv=np.max(es_peak), norm=-1,
                              lbd_stat=False)
        else:
            x, data_out, area_dict = calculate_profile(self.data, self.param_new,
                                                       elemental_lines=[self.e_name],
                                                       default_area=default_area)
            ratio_v = self.add_element_intensity / np.max(data_out[self.e_name])

            ps = PreFitStatus(z=get_Z(self.e_name),
                              energy=get_energy(self.e_name),
                              area=area_dict[self.e_name]*ratio_v,
                              spectrum=data_out[self.e_name]*ratio_v,
                              maxv=self.add_element_intensity, norm=-1,
                              lbd_stat=False)

        self.EC.add_to_dict({self.e_name: ps})

    def update_name_list(self):
        """
        When result_dict_names change, the looper in enaml will update.
        """
        # need to clean list first, in order to refresh the list in GUI
        self.result_dict_names = []
        self.result_dict_names = self.EC.element_dict.keys()
        logger.info('Current element names are {}'.format(self.result_dict_names))

    def find_peak(self, threshv=0.1):
        """
        Run automatic peak finding, and save results as dict of object.

        Parameters
        ----------
        threshv : float
            The value will not be shown on GUI if it is smaller than the threshold.
        """
        self.prefit_x, out_dict, area_dict = linear_spectrum_fitting(self.data,
                                                                     self.param_new)
        logger.info('Energy range: {}, {}'.format(
            self.param_new['non_fitting_values']['energy_bound_low']['value'],
            self.param_new['non_fitting_values']['energy_bound_high']['value']))

        prefit_dict = OrderedDict()
        for k, v in six.iteritems(out_dict):
            ps = PreFitStatus(z=get_Z(k), energy=get_energy(k),
                              area=area_dict[k], spectrum=v,
                              maxv=np.around(np.max(v), 1), norm=-1,
                              lbd_stat=False)
            prefit_dict.update({k: ps})

        logger.info('Automatic Peak Finding found elements as : {}'.format(
            prefit_dict.keys()))
        self.EC.delete_all()
        self.EC.add_to_dict(prefit_dict)

    def create_full_param(self):
        """
        Extend the param to full param dict including each element's
        information, and assign initial values from pre fit.
        """
        self.define_range()

        self.element_list = self.EC.get_element_list()
        self.param_new['non_fitting_values']['element_list'] = ', '.join(self.element_list)

        # remove elements not included in self.element_list
        self.param_new = param_dict_cleaner(self.param_new,
                                            self.element_list)

        # create full parameter list including elements
        # This part is a bit confusing and needs better treatment.
        PC = ParamController(self.param_new, self.element_list)
        # parameter values not updated based on param_new, so redo it
        param_temp = PC.params
        for k, v in six.iteritems(param_temp):
            if k == 'non_fitting_values':
                continue
            if self.param_new.has_key(k):
                v['value'] = self.param_new[k]['value']
        self.param_new = param_temp

        # to create full param dict, for GUI only
        create_full_dict(self.param_new, fit_strategy_list)

        # update according to pre fit results
        if len(self.EC.element_dict):
            for e in self.element_list:
                zname = e.split('_')[0]
                for k, v in six.iteritems(self.param_new):
                    if zname in k and 'area' in k:
                        v['value'] = self.EC.element_dict[e].area
            if 'compton' in self.EC.element_dict:
                self.param_new['compton_amplitude']['value'] = self.EC.element_dict['compton'].area
            if 'coherent_sct_amplitude' in self.EC.element_dict:
                self.param_new['coherent_sct_amplitude']['value'] = self.EC.element_dict['elastic'].area
            if 'escape' in self.EC.element_dict:
                self.param_new['non_fitting_values']['escape_ratio'] = (self.EC.element_dict['escape'].maxv
                                                                        / np.max(self.y0))

    def data_for_plot(self):
        """
        Save data in terms of K, L, M lines for plot.
        """
        self.total_y = {}
        self.total_y_l = {}
        self.total_y_m = {}
        new_dict = {k: v for (k, v)
                    in six.iteritems(self.EC.element_dict) if v.status}

        for k, v in six.iteritems(new_dict):
            if 'K' in k:
                self.total_y[k] = self.EC.element_dict[k].spectrum
            elif 'L' in k:
                self.total_y_l[k] = self.EC.element_dict[k].spectrum
            elif 'M' in k:
                self.total_y_m[k] = self.EC.element_dict[k].spectrum
            else:
                self.total_y[k] = self.EC.element_dict[k].spectrum


def save_as(file_path, data):
    """
    Save full param dict into a file.
    """
    with open(file_path, 'w') as outfile:
        json.dump(data, outfile,
                  sort_keys=True, indent=4)


def define_range(data, low, high):
    """
    Cut x range according to values define in param_dict.

    Parameters
    ----------
    data : array
        raw spectrum
    low : float
        low bound in KeV
    high : float
        high bound in KeV

    Returns
    -------
    x : array
        trimmed channel number
    y : array
        trimmed spectrum according to x
    """
    x = np.arange(data.size)
    # ratio to transfer energy value back to channel value
    approx_ratio = 100
    x0, y0 = trim(x, data, low*approx_ratio, high*approx_ratio)
    return x0, y0


def calculate_profile(y0, param,
                      elemental_lines, default_area=1e5):
    # Need to use deepcopy here to avoid unexpected change on parameter dict
    fitting_parameters = copy.deepcopy(param)

    x0 = np.arange(len(y0))

    # ratio to transfer energy value back to channel value
    approx_ratio = 100
    lowv = fitting_parameters['non_fitting_values']['energy_bound_low']['value'] * approx_ratio
    highv = fitting_parameters['non_fitting_values']['energy_bound_high']['value'] * approx_ratio

    x, y = trim(x0, y0, lowv, highv)

    e_select, matv, area_dict = construct_linear_model(x, fitting_parameters,
                                                       elemental_lines,
                                                       default_area=default_area)

    non_element = ['compton', 'elastic']
    total_list = e_select + non_element
    total_list = [str(v) for v in total_list]
    temp_d = {k: v for (k, v) in zip(total_list, matv.transpose())}

    # add background
    bg = snip_method(y, fitting_parameters['e_offset']['value'],
                     fitting_parameters['e_linear']['value'],
                     fitting_parameters['e_quadratic']['value'])
    temp_d['background'] = bg

    # add escape peak
    if fitting_parameters['non_fitting_values']['escape_ratio'] > 0:
        temp_d['escape'] = trim_escape_peak(y0, fitting_parameters, y.size)

    x = (fitting_parameters['e_offset']['value']
         + fitting_parameters['e_linear']['value'] * x
         + fitting_parameters['e_quadratic']['value'] * x**2)

    return x, temp_d, area_dict


def trim_escape_peak(data, param_dict, y_size):
    """
    Calculate escape peak within required range.

    Parameters
    ----------
    data : array
        raw spectrum
    param_dict : dict
        parameters for fitting
    y_size : int
        the size of trimmed spectrum

    Returns
    -------
    array :
        trimmed escape peak spectrum
    """
    ratio = param_dict['non_fitting_values']['escape_ratio']
    xe, ye = compute_escape_peak(data, ratio, param_dict)
    lowv = param_dict['non_fitting_values']['energy_bound_low']['value']
    highv = param_dict['non_fitting_values']['energy_bound_high']['value']
    xe, es_peak = trim(xe, ye, lowv, highv)
    logger.info('Escape peak is considered with ratio {}'.format(ratio))

    # align to the same length
    if y_size > es_peak.size:
        temp = es_peak
        es_peak = np.zeros(y_size)
        es_peak[:temp.size] = temp
    else:
        es_peak = es_peak[:y_size]
    return es_peak


def create_full_dict(param, name_list):
    """
    Create full param dict so each item has same nested dict.
    This is for GUI purpose only.

    .. warning :: This function mutates the input values.

    Pamameters
    ----------
    param : dict
        all parameters including element
    name_list : list
        strategy names
    """
    for n in name_list:
        for k, v in six.iteritems(param):
            if k == 'non_fitting_values':
                continue
            if n not in v:
                v.update({n: v['bound_type']})


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

    non_element = ['compton', 'elastic', 'background', 'escape']
    if ename.lower() in non_element:
        return '-'
    else:
        e = Element(strip_line(ename))
        return str(e.Z)


def get_energy(ename):
    strip_line = lambda ename: ename.split('_')[0]
    non_element = ['compton', 'elastic', 'background', 'escape']
    if ename in non_element:
        return '-'
    else:
        e = Element(strip_line(ename))
        if '_K' in ename:
            energy = e.emission_line['ka1']
        elif '_L' in ename:
            energy = e.emission_line['la1']
        elif '_M' in ename:
            energy = e.emission_line['ma1']

        return str(np.around(energy, 4))


def get_element(param):
    element_list = param['non_fitting_values']['element_list']
    return [e.strip(' ') for e in element_list.split(',')]


def factor_height2area(energy, param, std_correction=1):
    """
    Factor to transfer peak height to area.
    """
    temp_val = 2 * np.sqrt(2 * np.log(2))
    epsilon = param['non_fitting_values']['electron_hole_energy']
    sigma = np.sqrt((param['fwhm_offset']['value'] / temp_val)**2
                    + energy * epsilon * param['fwhm_fanoprime']['value'])
    return sigma*std_correction


def param_dict_cleaner(param, element_list):
    """
    Make sure param only contains element from element_list.

    Parameters
    ----------
    param : dict
        fitting parameters
    element_list : list
        list of elemental lines

    Returns
    -------
    dict :
        new param dict containing given elements
    """
    param_new = {}
    list_lower = [e.lower() for e in element_list]
    for k, v in six.iteritems(param):
        if k == 'non_fitting_values' or k == k.lower():
            param_new.update({k: v})
        else:
            if k[:4].lower() in list_lower:
                param_new.update({k: v})
    return param_new
