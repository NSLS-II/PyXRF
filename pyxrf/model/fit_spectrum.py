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
import multiprocessing
import h5py

from atom.api import Atom, Str, observe, Typed, Int, List, Dict, Float
from skxray.fitting.xrf_model import (ModelSpectrum, update_parameter_dict,
                                      sum_area, set_parameter_bound,
                                      ParamController, K_LINE, L_LINE, M_LINE,
                                      nnls_fit, weighted_nnls_fit, trim,
                                      construct_linear_model, linear_spectrum_fitting,
                                      register_strategy, TRANSITIONS_LOOKUP)
from skxray.fitting.background import snip_method
from skxray.constants.api import XrfElement as Element
from pyxrf.model.guessparam import (calculate_profile, fit_strategy_list,
                                    trim_escape_peak, define_range)
from lmfit import fit_report

import logging
logger = logging.getLogger(__name__)


class Fit1D(Atom):
    """
    Fit 1D fluorescence spectrum. Users can choose multiple strategies
    for this fitting.
    """
    file_status = Str()
    param_dict = Dict()

    element_list = List()

    data_all = Typed(np.ndarray)
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

    working_directory = Str()
    result_folder = Str()

    all_strategy = Typed(object) #Typed(OrderedDict)

    x0 = Typed(np.ndarray)
    y0 = Typed(np.ndarray)
    bg = Typed(np.ndarray)
    es_peak = Typed(np.ndarray)
    cal_x = Typed(np.ndarray)
    cal_y = Typed(np.ndarray)
    cal_spectrum = Dict()

    # attributes used by the ElementEdit window
    selected_element = Str()
    elementinfo_list = List()

    function_num = Int(0)
    nvar = Int(0)
    chi2 = Float(0.0)
    red_chi2 = Float(0.0)
    global_param_list = List()

    def __init__(self, *args, **kwargs):
        self.working_directory = kwargs['working_directory']
        self.result_folder = kwargs['working_directory']
        self.all_strategy = OrderedDict()

    def result_folder_changed(self, changed):
        """
        Observer function to be connected to the fileio model
        in the top-level gui.py startup

        Parameters
        ----------
        changed : dict
            This is the dictionary that gets passed to a function
            with the @observe decorator
        """
        self.result_folder = changed['value']

    @observe('selected_element')
    def _selected_element_changed(self, changed):
        if len(self.selected_element) <= 4:
            element = self.selected_element.split('_')[0]
            self.elementinfo_list = sorted([e for e in self.param_dict.keys()
                                            if (element+'_' in e) and  # error between S_k or Si_k
                                            ('pileup' not in e)])  # Si_ka1 not Si_K
        else:
            element = self.selected_element  # for pileup peaks
            self.elementinfo_list = sorted([e for e in self.param_dict.keys()
                                            if element.replace('-', '_') in e])

    def get_new_param(self, param):
        self.param_dict = copy.deepcopy(param)
        element_list = self.param_dict['non_fitting_values']['element_list']
        self.element_list = [e.strip(' ') for e in element_list.split(',')]

        # global parameters
        # for GUI purpose only
        # if we do not clear the list first, there is not update on the GUI
        self.global_param_list = []
        self.global_param_list = sorted([k for k in six.iterkeys(self.param_dict)
                                         if k == k.lower() and k != 'non_fitting_values'])

        self.define_range()

        # register the strategy and extend the parameter list
        # to cover all given elements
        for strat_name in fit_strategy_list:
            strategy = extract_strategy(self.param_dict, strat_name)
            # register the strategy and extend the parameter list
            # to cover all given elements
            register_strategy(strat_name, strategy)
            #set_parameter_bound(self.param_dict, strat_name)

    @observe('data')
    def _update_data(self, change):
        self.data = np.asarray(self.data)

    @observe('fit_strategy1')
    def update_strategy1(self, change):
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
        """
        Cut x range according to values define in param_dict.
        """
        lowv = self.param_dict['non_fitting_values']['energy_bound_low']['value']
        highv = self.param_dict['non_fitting_values']['energy_bound_high']['value']
        self.x0, self.y0 = define_range(self.data, lowv, highv,
                                        self.param_dict['e_offset']['value'],
                                        self.param_dict['e_linear']['value'])

    def get_background(self):
        self.bg = snip_method(self.y0,
                              self.param_dict['e_offset']['value'],
                              self.param_dict['e_linear']['value'],
                              self.param_dict['e_quadratic']['value'])

    def get_profile(self):
        """
        Calculate profile based on current parameters.
        """
        #self.define_range()
        self.cal_x, self.cal_spectrum, area_dict = calculate_profile(self.x0,
                                                                     self.y0,
                                                                     self.param_dict,
                                                                     self.element_list)
        # add escape peak
        if self.param_dict['non_fitting_values']['escape_ratio'] > 0:
            self.cal_spectrum['escape'] = trim_escape_peak(self.data,
                                                           self.param_dict,
                                                           len(self.y0))

        self.cal_y = np.zeros(len(self.cal_x))
        for k, v in six.iteritems(self.cal_spectrum):
            self.cal_y += v

        self.residual = self.cal_y - self.y0

    def fit_data(self, x0, y0,
                 c_val=1e-3, fit_num=100, c_weight=1e3):
        MS = ModelSpectrum(self.param_dict, self.element_list)
        MS.assemble_models()

        result = MS.model_fit(x0, y0,
                              weights=1/np.sqrt(c_weight+y0),
                              maxfev=fit_num,
                              xtol=c_val, ftol=c_val, gtol=c_val)
        self.fit_x = (result.values['e_offset'] +
                      result.values['e_linear'] * x0 +
                      result.values['e_quadratic'] * x0**2)
        self.fit_y = result.best_fit
        self.fit_result = result
        self.residual = self.fit_y - y0

    def fit_multiple(self):
        """
        Fit data in sequence according to given strategies.
        The param_dict is extended to cover elemental parameters.
        """
        #self.define_range()
        self.get_background()

        #PC = ParamController(self.param_dict, self.element_list)
        #self.param_dict = PC.params
        #print('param keys {}'.format(self.param_dict.keys()))

        if self.param_dict['non_fitting_values']['escape_ratio'] > 0:
            self.es_peak = trim_escape_peak(self.data,
                                            self.param_dict,
                                            self.y0.size)
            y0 = self.y0 - self.bg - self.es_peak
        else:
            y0 = self.y0 - self.bg

        t0 = time.time()
        logger.info('Start fitting!')
        for k, v in six.iteritems(self.all_strategy):
            if v:
                strat_name = fit_strategy_list[v-1]
                logger.info('Fit with {}: {}'.format(k, strat_name))

                strategy = extract_strategy(self.param_dict, strat_name)
                # register the strategy and extend the parameter list
                # to cover all given elements
                register_strategy(strat_name, strategy)
                set_parameter_bound(self.param_dict, strat_name)

                self.fit_data(self.x0, y0)
                self.update_param_with_result()
        t1 = time.time()
        logger.warning('Time used for fitting is : {}'.format(t1-t0))

        # for GUI purpose only
        # if we do not clear the dict first, there is not update on the GUI
        param_temp = copy.deepcopy(self.param_dict)
        del self.param_dict['non_fitting_values']
        self.param_dict = param_temp

        self.comps.clear()
        comps = self.fit_result.eval_components(x=self.x0)
        self.comps = combine_lines(comps, self.element_list, self.bg)

        if self.param_dict['non_fitting_values']['escape_ratio'] > 0:
            self.fit_y += self.bg + self.es_peak
            self.comps['escape'] = self.es_peak
        else:
            self.fit_y += self.bg

        self.save_result()
        self.assign_fitting_result()

    def assign_fitting_result(self):
        self.function_num = self.fit_result.nfev
        self.nvar = self.fit_result.nvarys
        self.chi2 = np.around(self.fit_result.chisqr, 4)
        self.red_chi2 = np.around(self.fit_result.redchi, 4)

    def fit_single_pixel(self):
        """
        This function performs single pixel fitting.
        Multiprocess is considered.
        """
        save_name = 'pv250_slice1_data'
        #save_name = 'bnp_fly0148_data'
        #save_name = 'hxn_scan_01167_data'
        save_dict = {'fit_path': os.path.join(self.result_folder, save_name+'_pixel'),
                     'save_range': 50}

        strategy_pixel = 'linear'
        set_parameter_bound(self.param_dict, strategy_pixel)
        logger.info('Starting single pixel fitting')
        t0 = time.time()
        result_map = fit_pixel_fast_multi(self.data_all, self.param_dict,
                                          first_peak_area=True, **save_dict)
        t1 = time.time()
        logger.warning('Time used for pixel fitting is : {}'.format(t1-t0))

        # save data
        #fpath = os.path.join(self.result_folder, 'Root.h5')
        #write_to_hdf(fpath, result_map)

        # currently save data using pickle, need to be updated
        import pickle
        fpath = os.path.join(self.result_folder, save_name)
        pickle.dump(result_map, open(fpath, 'wb'))

    def fit_multi_files(self):
        """
        Fit data from multiple files.

        .. warning: this function is to be updated. Poor organized.
        """
        file_prefix = 'xsp3.0'
        working_directory = '/Users/Li/Downloads/aps13ide'
        result_file = 'xsp3_data'

        start_i = 100
        end_i = 110

        t0 = time.time()
        #result_map = fit_pixel_fast_multi(self.data_all, self.param_dict)

        result = fit_data_multi_files(working_directory, file_prefix,
                                      self.param_dict, start_i, end_i)

        t1 = time.time()
        logger.warning('Time used for pixel fitting multiple files is : {}'.format(t1-t0))

        import pickle
        fpath = os.path.join(self.result_folder, result_file)
        pickle.dump(result, open(fpath, 'wb'))

    def save_result(self, fname=None):
        """
        Save fitting results.

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


def combine_lines(components, element_list, background):
    """
    Combine results for different lines of the same element.
    And also add background, compton and elastic.

    Parameters
    ----------
    components : dict
        output results from lmfit
    element_list : list
        list of elemental lines
    background : array
        background calculated in given range

    Returns
    -------
    dict :
        combined results for elements and other related peaks.
    """
    new_components = {}
    for e in element_list:
        if len(e) <= 4:
            e_temp = e.split('_')[0]
            intensity = 0
            for k, v in six.iteritems(components):
                if (e_temp in k) and (e not in k):
                    intensity += v
            new_components[e] = intensity
        else:
            comp_name = 'pileup_' + e.replace('-', '_') + '_'  # change Si_K-Si_K to Si_K_Si_K
            new_components[e] = components[comp_name]

    # add background and elastic
    new_components['background'] = background
    new_components['compton'] = components['compton']
    new_components['elastic'] = components['elastic_']
    return new_components


def extract_strategy(param, name):
    """
    Extract given strategy from param dict.

    Parameters
    ----------
    param : dict
        saving all parameters
    name : str
        strategy name

    Returns
    -------
    dict :
        with given strategy as value
    """
    param_new = copy.deepcopy(param)
    return {k: v[name] for k, v in six.iteritems(param_new)
            if k != 'non_fitting_values'}


def fit_pixel_fast(dir_path, file_prefix,
                   fileID, param, interpath,
                   save_spectrum=True):
    """
    Single pixel fit of experiment data. No multiprocess is applied.

    .. warning :: This function is not optimized as it calls
    linear_spectrum_fitting function, where lots of repeated
    calculation are processed.

    Parameters
    ----------
    data : array
        3D data of experiment spectrum
    param : dict
        fitting parameters

    Returns
    -------
    dict :
        fitting values for all the elements
    """

    num_str = '{:03d}'.format(fileID)
    #logger.info('File number is {}'.format(fileID))
    filename = file_prefix + num_str
    file_path = os.path.join(dir_path, filename)
    with h5py.File(file_path, 'r') as f:
        data = f[interpath][:]
    #data = np.array(data[:, :, :])
    datas = data.shape

    #x0 = np.arange(datas[2])

    elist = param['non_fitting_values']['element_list'].split(', ')
    elist = [e.strip(' ') for e in elist]

    non_element = ['compton', 'elastic', 'background']
    total_list = elist + non_element

    result_map = dict()
    for v in total_list:
        if save_spectrum:
            result_map.update({v: np.zeros([datas[0], datas[1], datas[2]])})
        else:
            result_map.update({v: np.zeros([datas[0], datas[1]])})

    for i in xrange(datas[0]):
        #logger.info('Row number at {} out of total {}'.format(i, datas[0]))
        for j in xrange(datas[1]):
            #logger.info('Column number at {} out of total {}'.format(j, datas[1]))
            x, result, area_v = linear_spectrum_fitting(data[i, j, :], param,
                                                        elemental_lines=elist,
                                                        constant_weight=None)
            for v in total_list:
                if v in result:
                    if save_spectrum:
                        result_map[v][i, j, :len(result[v])] = result[v]
                    else:
                        result_map[v][i, j] = np.sum(result[v])

    return result_map


def fit_data_multi_files(dir_path, file_prefix,
                         param, start_i, end_i,
                         interpath='entry/instrument/detector/data'):
    """
    Fitting for multiple files with Multiprocessing.

    Parameters
    -----------
    dir_path : str
    file_prefix : str
    param : dict
    start_i : int
        start id of given file
    end_i: int
        end id of given file
    interpath : str
        path inside hdf5 file to fetch the data

    Returns
    -------
    result : list
        fitting result as list of dict
    """
    no_processors_to_use = multiprocessing.cpu_count()
    logger.info('cpu count: {}'.format(no_processors_to_use))
    #print 'Creating pool with %d processes\n' % no_processors_to_use
    pool = multiprocessing.Pool(no_processors_to_use)

    result_pool = [pool.apply_async(fit_pixel_fast,
                                    (dir_path, file_prefix,
                                     m, param, interpath))
                   for m in range(start_i, end_i+1)]

    results = []
    for r in result_pool:
        results.append(r.get())

    pool.terminate()
    pool.join()
    return results


def roi_sum_calculation(dir_path, file_prefix, fileID,
                        element_dict, interpath):
    """
    Parameters
    -----------
    dir_path : str
    file_prefix : str
    fileID : int
    element_dict : dict
        element name with low/high bound
    interpath : str
        path inside hdf5 file to fetch the data

    Returns
    -------
    result : dict
        roi sum for all given elements
    """
    num_str = '{:03d}'.format(fileID)
    #logger.info('File number is {}'.format(fileID))
    filename = file_prefix + num_str
    file_path = os.path.join(dir_path, filename)
    with h5py.File(file_path, 'r') as f:
        data = f[interpath][:]

    result_map = dict()
    #for v in six.iterkeys(element_dict):
    #    result_map[v] = np.zeros([datas[0], datas[1]])

    for k, v in six.iteritems(element_dict):
        result_map[k] = np.sum(data[:, :, v[0]: v[1]], axis=2)

    return result_map


def roi_sum_multi_files(dir_path, file_prefix,
                        start_i, end_i, element_dict,
                        interpath='entry/instrument/detector/data'):
    """
    Fitting for multiple files with Multiprocessing.

    Parameters
    -----------
    dir_path : str
    file_prefix : str
    start_i : int
        start id of given file
    end_i: int
        end id of given file
    element_dict : dict
        dict of element with [low, high] bounds as values
    interpath : str
        path inside hdf5 file to fetch the data

    Returns
    -------
    result : list
        fitting result as list of dict
    """
    no_processors_to_use = multiprocessing.cpu_count()
    logger.info('cpu count: {}'.format(no_processors_to_use))
    #print 'Creating pool with %d processes\n' % no_processors_to_use
    pool = multiprocessing.Pool(no_processors_to_use)

    result_pool = [pool.apply_async(roi_sum_calculation,
                                    (dir_path, file_prefix,
                                     m, element_dict, interpath))
                   for m in range(start_i, end_i+1)]

    results = []
    for r in result_pool:
        results.append(r.get())

    pool.terminate()
    pool.join()
    return results


def extract_result(data, element):
    """
    Extract fitting result returned from fitting of multi files.

    Parameters
    ----------
    data : list
        list of dict
    element : str
        elemental line
    """
    data_map = []
    for v in data:
        data_map.append(v[element])
    return np.array(data_map)


def fit_pixel(y, expected_matrix, constant_weight=10):
    """
    Non-negative linear fitting is applied for each pixel.

    Parameters
    ----------
    y : array
        spectrum of experiment data
    expected_matrix : array
        2D matrix of activated element spectrum
    constant_weight : float
        value used to calculate weight like so:
        weights = constant_weight / (constant_weight + spectrum)

    Returns
    -------
    results : array
        weights of different element
    residue : array
        error
    """
    if constant_weight:
        results, residue = weighted_nnls_fit(y, expected_matrix,
                                             constant_weight=constant_weight)
    else:
        results, residue = nnls_fit(y, expected_matrix)
    return results, residue


def fit_per_line(row_num, data,
                 matv, param):
    """
    Fit experiment data for a given row.

    Parameters
    ----------
    row_num : int
        which row to fit
    data : array
        3D data of experiment spectrum
    param : dict
        fitting parameters

    Returns
    -------
    array :
        fitting values for all the elements at a given row.
    """

    logger.info('Row number at {}'.format(row_num))
    out = []
    for i in range(data.shape[0]):
        bg = snip_method(data[i, :],
                         param['e_offset']['value'],
                         param['e_linear']['value'],
                         param['e_quadratic']['value'])
        y = data[i, :] - bg
        # setting constant weight to some value might cause error when fitting
        result, res = fit_pixel(y, matv,
                                constant_weight=None)
        result = list(result) + [np.sum(bg)]
        out.append(result)
    return np.array(out)


def fit_pixel_fast_multi(data, param, first_peak_area=False, **kwargs):
    """
    Multiprocess fit of experiment data.

    Parameters
    ----------
    data : array
        3D data of experiment spectrum
    param : dict
        fitting parameters
    first_peak_area : Bool, optional
        get overal peak area or only the first peak area, such as Ar_Ka1
    kwargs : dict
        the size of saved data and path

    Returns
    -------
    dict :
        fitting values for all the elements
    """

    no_processors_to_use = multiprocessing.cpu_count()
    #no_processors_to_use = 4

    logger.info('cpu count: {}'.format(no_processors_to_use))
    #print 'Creating pool with %d processes\n' % no_processors_to_use
    pool = multiprocessing.Pool(no_processors_to_use)

    y0 = data[0, 0, :]
    x0 = np.arange(len(y0))

    # transfer energy value back to channel value
    lowv = (param['non_fitting_values']['energy_bound_low']['value'] -
            param['e_offset']['value'])/param['e_linear']['value']
    highv = (param['non_fitting_values']['energy_bound_high']['value'] -
             param['e_offset']['value'])/param['e_linear']['value']

    lowv = int(lowv)
    highv = int(highv)

    x, y = trim(x0, y0, lowv, highv)
    start_i = x0[x0 == x[0]][0]
    end_i = x0[x0 == x[-1]][0]
    logger.info('label range: {}, {}'.format(start_i, end_i))

    elist = param['non_fitting_values']['element_list'].split(', ')
    elist = [e.strip(' ') for e in elist]

    e_select, matv, e_area = construct_linear_model(x, param, elist)
    mat_sum = np.sum(matv, axis=0)

    result_pool = [pool.apply_async(fit_per_line,
                                    (n, data[n, :, start_i:end_i+1], matv, param))
                   for n in range(data.shape[0])]

    results = []
    for r in result_pool:
        results.append(r.get())

    pool.terminate()
    pool.join()

    results = np.array(results)

    total_list = e_select + ['background']

    logger.info('The following peaks are saved: {}'.format(total_list))

    result_map = dict()
    for i in range(len(total_list)-1):
        if first_peak_area is not True:
            result_map.update({total_list[i]: results[:, :, i]*mat_sum[i]})
        else:
            if total_list[i] not in K_LINE+L_LINE+M_LINE:
                ratio_v = 1
            else:
                ratio_v = get_branching_ratio(total_list[i],
                                              param['coherent_sct_energy']['value'])
            result_map.update({total_list[i]: results[:, :, i]*mat_sum[i]*ratio_v})

    # add background
    result_map.update({total_list[-1]: results[:, :, -1]})

    # save summed spectrum only when required
    # the size is measured from the point of (0, 0)
    #sum_total = np.zeros([results.shape[0], results.shape[1], matv.shape[0]])
    if kwargs['save_range']:
        sum_total = np.zeros([kwargs['save_range'],
                              kwargs['save_range'],
                              matv.shape[0]])
        for m in range(sum_total.shape[0]):
            for n in range(sum_total.shape[1]):
                for i in range(len(total_list)-1):
                    sum_total[m, n, :] += results[m, n, i] * matv[:, i]
                bg = snip_method(data[m, n, start_i:end_i+1],
                                 param['e_offset']['value'],
                                 param['e_linear']['value'],
                                 param['e_quadratic']['value'])
                sum_total[m, n, :] += bg

        #fpath = os.path.join(kwargs['fit_path'])
        np.save(kwargs['fit_path'], sum_total)
        logger.info('Single pixel data is also save to : {}'.format(kwargs['fit_path']))

    #pickle.dump(result_map, open(fpath, 'wb'))
    return result_map


def get_branching_ratio(elemental_line, energy):
    """
    Calculate the ratio of branching ratio, such as ratio of
    branching ratio of Ka1 to sum of br of all K lines.

    Parameters
    ----------
    elemental_line : str
        e.g., 'Mg_K', refers to the K lines of Magnesium
    energy : float
        incident energy in keV

    Returns
    -------
    float :
        calculated ratio
    """

    name, line = elemental_line.split('_')
    e = Element(name)
    transition_lines = TRANSITIONS_LOOKUP[line.upper()]

    sum_v = 0
    for v in transition_lines:
        sum_v += e.cs(energy)[v]
    ratio_v = e.cs(energy)[transition_lines[0]]/sum_v
    return ratio_v


# def fit_pixel_fast_multi(data, param):
#     """
#     Multiprocess fit of experiment data.
#
#     Parameters
#     ----------
#     data : array
#         3D data of experiment spectrum
#     param : dict
#         fitting parameters
#
#     Returns
#     -------
#     dict :
#         fitting values for all the elements
#     """
#
#     #logger.info('Row number at {} out of total {}'.format(i, datas[0]))
#     #logger.info('no_processors_to_use = {}'.format(no_processors_to_use))
#     no_processors_to_use = multiprocessing.cpu_count()
#     logger.info('cpu count: {}'.format(no_processors_to_use))
#     #print 'Creating pool with %d processes\n' % no_processors_to_use
#     pool = multiprocessing.Pool(no_processors_to_use)
#
#     datas = data.shape
#
#     x0 = np.arange(datas[2])
#
#     elist = param['non_fitting_values']['element_list'].split(', ')
#     elist = [e.strip(' ') for e in elist]
#     elist = [e+'_K' for e in elist if ('_' not in e)]
#
#     non_element = ['compton', 'elastic', 'background']
#     total_list = elist + non_element
#
#     result_map = dict()
#     for v in total_list:
#         result_map.update({v: np.zeros([datas[0], datas[1]])})
#
#     result_pool = [pool.apply_async(fit_per_line,
#                                     (i, data, param)) for i in range(datas[0])]
#
#     results = []
#     for r in result_pool:
#         results.append(r.get())
#
#     pool.terminate()
#     pool.join()
#
#     results = np.array(results)
#
#     for v in total_list:
#         for i in xrange(datas[0]):
#             for j in xrange(datas[1]):
#                 result_map[v][i, j] = results[i, j].get(v, 0)
#
#     return result_map


def fit_pixel_slow_version(data, param, c_val=1e-2, fit_num=10, c_weight=1):
    datas = data.shape

    x0 = np.arange(datas[2])

    elist = param['non_fitting_values']['element_list'].split(', ')
    elist = [e.strip(' ') for e in elist]

    result_map = dict()
    for v in elist:
        result_map.update({v: np.zeros([datas[0], datas[1]])})

    MS = ModelSpectrum(param)
    MS.model_spectrum()

    for i in xrange(datas[0]):
        logger.info('Row number at {} out of total {}'.format(i, datas[0]))
        for j in xrange(datas[1]):
            logger.info('Column number at {} out of total {}'.format(j, datas[1]))
            y0 = data[i, j, :]
            result = MS.model_fit(x0, y0,
                                  w=1/np.sqrt(c_weight+y0))

                                  #maxfev=fit_num, xtol=c_val, ftol=c_val, gtol=c_val)
            #for k, v in six.iteritems(result.values):
            #    print('result {}: {}'.format(k, v))
            # save result
            for v in elist:
                if '_L' in v:
                    line_name = v.split('_')[0]+'_la1_area'
                elif '_M' in v:
                    line_name = v.split('_')[0]+'_ma1_area'
                else:
                    line_name = v+'_ka1_area'
                result_map[v][i, j] = result.values[line_name]

    return result_map


def write_to_hdf(fpath, data_dict, interpath='xrfmap/detsum'):
    """
    Add fitting results to existing h5 file. This is to be moved to filestore.

    Parameters
    ----------
    fpath : str
        path of the hdf5 file
    data_dict : dict
        dict of array
    interpath : str
        path inside h5py file
    """
    f = h5py.File(fpath, 'r+')
    dataGrp = f[interpath]

    data = []
    namelist = []
    for k, v in six.iteritems(data_dict):
        namelist.append(str(k))
        data.append(v)

    if 'xrf_fit' in dataGrp:
        del dataGrp['xrf_fit']

    data = np.array(data)
    ds_data = dataGrp.create_dataset('xrf_fit', data=data)
    ds_data.attrs['comments'] = 'All fitting values are saved.'

    if 'xrf_fit_name' in dataGrp:
        del dataGrp['xrf_fit_name']

    name_data = dataGrp.create_dataset('xrf_fit_name', data=namelist)
    name_data.attrs['comments'] = 'All elements for fitting are saved.'

    f.close()


def ccombine_data_to_hdf(fpath_read, file_prefix,
                         start_id, end_id,
                         interpath_read='entry/instrument/detector/data'):
    """
    Read data from each point scan, then save them to one hdf file.
    Following APS X13 beamline structure.
    """

    import h5py
    #import copy

    #datasum = np.zeros([100, 605, 4096])
    datasum = None
    for i in range(start_id, end_id+1):
        num_str = '{:03d}'.format(i)
        filename = file_prefix + num_str
        file_path = os.path.join(fpath_read, filename)
        with h5py.File(file_path, 'r') as f:
            data_temp = f[interpath_read][:]
            #data_temp = np.asarray(data_temp)
            #datasum.append(np.sum(data_temp, axis=1))
            if datasum is None:
                datasum = np.zeros([end_id-start_id+1,
                                    data_temp.shape[0],
                                    data_temp.shape[1],
                                    data_temp.shape[2]])
            datasum[i-start_id, :, :, :] = data_temp

    return datasum


def compare_result(m, n, start_i=151, end_i=1350, all=True, linear=True):

    import h5py
    import matplotlib.pyplot as plt

    x = np.arange(end_i-start_i)

    fpath = '/Users/Li/Downloads/xrf_data/Root.h5'
    myfile = h5py.File(fpath, 'r')
    data_exp = myfile['xrfmap/det1/counts']

    fpath_fit = '/Users/Li/Downloads/xrf_data/fit_data.npy'
    d_fit = np.load(fpath_fit)

    if not all:
        if linear:
            plt.plot(x, data_exp[m, n, start_i:end_i], x, d_fit[m, n, :])
        else:
            plt.semilogy(x, data_exp[m, n, start_i:end_i], x, d_fit[m, n, :])
        plt.show()

    else:
        if linear:
            plt.plot(x, np.sum(data_exp[:, :, start_i:end_i], axis=(0, 1)), x, np.sum(d_fit, axis=(0, 1)))
        else:
            plt.semilogy(x, np.sum(data_exp[:, :, start_i:end_i], axis=(0, 1)), x, np.sum(d_fit, axis=(0, 1)))
        plt.show()
