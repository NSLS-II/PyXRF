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

from atom.api import (Atom, Str, observe, Typed,
                      Int, Dict, List, Float, Enum, Bool)

from skxray.fitting.background import snip_method
from skxray.constants.api import XrfElement as Element
from skxray.fitting.xrf_model import (ModelSpectrum, ParamController,
                                      set_range, get_linear_model, pre_fit_linear)
#from bubblegum.xrf.model.fit_spectrum import fit_strategy_list

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
        max_dict = np.max(np.array([v.maxv for v in six.itervalues(self.element_dict)]))

        for v in six.itervalues(self.element_dict):
            v.norm = v.maxv/max_dict*100
            v.lbd_stat = bool(v.norm > threshv)

    def delete_all(self):
        self.element_dict.clear()

    def get_element_list(self):
        current_elements = [v for v in six.iterkeys(self.element_dict) if v.lower() != v]
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
    parameters = Dict() #Typed(OrderedDict) #OrderedDict()
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
    e_intensity = Float()
    #save_file = Str()

    result_folder = Str()
    file_path = Str()

    element_list = List()

    data_sets = Typed(OrderedDict)
    file_opt = Int()
    data_all = Typed(np.ndarray)

    EC = Typed(object)

    def __init__(self, *args, **kwargs):
        try:
            self.default_parameters = kwargs['default_parameters']
            self.element_list, self.parameters = dict_to_param(self.default_parameters)
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
        self.element_list, self.parameters = dict_to_param(self.param_new)
        self.EC.delete_all()
        self.create_spectrum_from_file(self.param_new)
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
        self.EC.delete_all()
        self.create_spectrum_from_file(param_dict)
        self.param_new = param_dict

    def create_spectrum_from_file(self, param_dict):
        """
        Create spectrum profile with given param dict from file.

        Parameters
        ----------
        param_dict : dict
            dict obtained from file
        """
        self.prefit_x, pre_dict = calculate_profile(self.data, param_dict)

        default_area = 1e5
        factor_to_area = factor_height2area()

        temp_dict = OrderedDict()
        for e in six.iterkeys(pre_dict):
            ename = e.split('_')[0]
            for k, v in six.iteritems(param_dict):

                if ename in k and 'area' in k:
                    ratio = v['value']/factor_to_area/np.max(pre_dict[e])
                    spectrum = pre_dict[e]*ratio
                    if 'ka1' in k:
                        e = e+'_K'

                elif ename == 'compton' and k == 'compton_amplitude':
                    ratio = v['value']/factor_to_area/np.max(pre_dict[e])
                    spectrum = pre_dict[e]*ratio

                elif ename == 'elastic' and k == 'coherent_sct_amplitude':
                    ratio = v['value']/factor_to_area/np.max(pre_dict[e])
                    spectrum = pre_dict[e]*ratio

                elif ename == 'background':
                    spectrum = pre_dict[e]

                else:
                    continue

                ps = PreFitStatus(z=get_Z(ename), energy=get_energy(e), spectrum=spectrum,
                                  maxv=np.around(np.max(spectrum), 1),
                                  norm=-1, lbd_stat=False)

                temp_dict.update({e: ps})

        self.EC.add_to_dict(temp_dict)

    @observe('file_opt')
    def choose_file(self, change):
        if self.file_opt == 0:
            return
        names = self.data_sets.keys()
        self.data = self.data_sets[names[self.file_opt-1]].get_sum()
        self.data_all = self.data_sets[names[self.file_opt-1]].raw_data

    def manual_input(self):
        logger.info('Element {} is added'.format(self.e_name))
        param_dict = format_dict(self.parameters, self.element_list)

        strip_line = lambda ename: ename.split('_')[0]

        if '_K' in self.e_name:
            e_item = strip_line(self.e_name)
        else:
            e_item = self.e_name

        x, data_out = calculate_profile(self.data, param_dict, elementlist=e_item)
        ps = PreFitStatus(z=get_Z(self.e_name), energy=get_energy(self.e_name),
                          spectrum=data_out[e_item]/1e5*self.e_intensity,
                          maxv=self.e_intensity, norm=-1,
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
        """
        param_dict = format_dict(self.parameters, self.element_list)
        self.prefit_x, out_dict = pre_fit_linear(self.data, param_dict)

        #max_dict = reduce(max, map(np.max, six.itervalues(out_dict)))
        prefit_dict = OrderedDict()
        for k, v in six.iteritems(out_dict):
            ps = PreFitStatus(z=get_Z(k), energy=get_energy(k), spectrum=v,
                              maxv=np.around(np.max(v), 1), norm=-1,
                              lbd_stat=False)
            prefit_dict.update({k: ps})

        logger.info('The elements found from prefit: {}'.format(prefit_dict.keys()))
        self.EC.add_to_dict(prefit_dict)

    # def save_elist(self):
    #     """
    #     Save selected list to param dict.
    #     """
    #     self.element_list = self.EC.get_element_list()
    #     temp_list = []
    #     for v in self.element_list:
    #         if '_K' in v:
    #             v = v.split('_')[0]
    #         temp_list.append(v)

        # self.param_d = format_dict(self.parameters, temp_list)

    def create_full_param(self, peak_std=0.07, peak_height=500.0):
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

        self.element_list = self.EC.get_element_list()
        temp_list = []
        for v in self.element_list:
            if '_K' in v:
                v = v.split('_')[0]
            temp_list.append(v)

        param_d = format_dict(self.parameters, temp_list)

        # create full parameter list including elements
        PC = ParamController(param_d)
        PC.create_full_param()
        self.param_new = PC.new_parameter

        # to create full param dict, for GUI only
        create_full_dict(self.param_new, fit_strategy_list)

        #factor_to_area = np.sqrt(2*np.pi)*peak_std*0.5
        factor_to_area = factor_height2area()

        # update according to pre fit results
        if len(self.EC.element_dict):
            for e in self.element_list:
                e = e.strip(' ')
                zname = e.split('_')[0]
                for k, v in six.iteritems(self.param_new):
                    if zname in k and 'area' in k:
                        if self.EC.element_dict[e].maxv > 0:
                            v['value'] = self.EC.element_dict[e].maxv*factor_to_area
                        else:
                            v['value'] = peak_height*factor_to_area
            if 'compton' in self.EC.element_dict:
                self.param_new['compton_amplitude']['value'] = (self.EC.element_dict['compton'].maxv
                                                                * factor_to_area)
            if 'coherent_sct_amplitude' in self.EC.element_dict:
                self.param_new['coherent_sct_amplitude']['value'] = (self.EC.element_dict['elastic'].maxv
                                                                     * factor_to_area)

    def data_for_plot(self):
        """
        Save data in terms of K, L, M lines for plot.
        """
        self.total_y = {}
        self.total_y_l = {}
        self.total_y_m = {}
        new_dict = {k: v for (k, v) in six.iteritems(self.EC.element_dict) if v.status}
        for k, v in six.iteritems(new_dict):
            if 'K' in k:
                self.total_y[k] = self.EC.element_dict[k].spectrum
            elif 'L' in k:
                self.total_y_l[k] = self.EC.element_dict[k].spectrum
            elif 'M' in k:
                self.total_y_m[k] = self.EC.element_dict[k].spectrum
            else:
                self.total_y[k] = self.EC.element_dict[k].spectrum

    def save(self, fname='param_default1.json'):
        """
        Save full param dict into a file at result directory.
        The name of the file is predefined.

        Parameters
        ----------
        fname : str, optional
            file name to save updated parameters
        """
        fpath = os.path.join(self.result_folder, fname)
        with open(fpath, 'w') as outfile:
            json.dump(self.param_new, outfile,
                      sort_keys=True, indent=4)

    def save_as(self):
        """
        Save full param dict into a file.
        """
        with open(self.file_path, 'w') as outfile:
            json.dump(self.param_new, outfile,
                      sort_keys=True, indent=4)

    def read_pre_saved(self, fname='param_default1.json'):
        """This is a bad idea."""
        fpath = os.path.join(self.result_folder, fname)
        with open(fpath, 'r') as infile:
            data = json.load(infile)
        return data


def factor_height2area(std=0.07):
    """
    Factor to transfer peak height to area.
    """
    return np.sqrt(2*np.pi)*std


def calculate_profile(y0, param, elementlist=None):
    # Need to use deepcopy here to avoid unexpected change on parameter dict
    fitting_parameters = copy.deepcopy(param)

    x0 = np.arange(len(y0))

    # ratio to transfer energy value back to channel value
    approx_ratio = 100
    lowv = fitting_parameters['non_fitting_values']['energy_bound_low'] * approx_ratio
    highv = fitting_parameters['non_fitting_values']['energy_bound_high'] * approx_ratio

    x, y = set_range(x0, y0, lowv, highv)

    if elementlist:
        fitting_parameters['non_fitting_values']['element_list'] = elementlist

    e_select, matv = get_linear_model(x, fitting_parameters, default_area=1e5)

    non_element = ['compton', 'elastic']
    total_list = e_select + non_element
    total_list = [str(v) for v in total_list]
    temp_d = {k: v for (k, v) in zip(total_list, matv.transpose())}

    # get background
    bg = snip_method(y, fitting_parameters['e_offset']['value'],
                     fitting_parameters['e_linear']['value'],
                     fitting_parameters['e_quadratic']['value'])

    temp_d.update(background=bg)
    #for i in len(total_list):
    #    temp_d[total_list[i]] = matv[:, i]

    x = fitting_parameters['e_offset']['value'] + fitting_parameters['e_linear']['value']*x \
        + fitting_parameters['e_quadratic']['value'] * x**2

    return x, temp_d


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

    non_element = ['compton', 'elastic', 'background']
    if ename in non_element:
        return '-'
    else:
        e = Element(strip_line(ename))
        return str(e.Z)


def get_energy(ename):
    strip_line = lambda ename: ename.split('_')[0]
    non_element = ['compton', 'elastic', 'background']
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


