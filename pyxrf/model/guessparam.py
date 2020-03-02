from __future__ import (absolute_import, division,
                        print_function)

import numpy as np
import six
import json
from collections import OrderedDict
import copy
import math

from atom.api import (Atom, Str, observe, Typed,
                      Int, Dict, List, Float, Bool)

from skbeam.core.fitting.background import snip_method
from skbeam.fluorescence import XrfElement as Element
from skbeam.core.fitting.xrf_model import (ParamController,
                                           compute_escape_peak, trim,
                                           construct_linear_model,
                                           linear_spectrum_fitting)
from skbeam.core.fitting.xrf_model import (K_LINE, L_LINE, M_LINE)

import logging
logger = logging.getLogger()


bound_options = ['none', 'lohi', 'fixed', 'lo', 'hi']
fit_strategy_list = ['fit_with_tail', 'free_more',
                     'e_calibration', 'linear',
                     'adjust_element1', 'adjust_element2', 'adjust_element3']
autofit_param = ['e_offset', 'e_linear', 'fwhm_offset', 'fwhm_fanoprime']


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
            logger.debug('Item {} is deleted.'.format(k))
        except KeyError:
            pass

    def order(self, option='z'):
        """
        Order dict in different ways.
        """
        if option == 'z':
            self.element_dict = OrderedDict(sorted(
                six.iteritems(self.element_dict), key=lambda t: t[1].z))
        elif option == 'energy':
            self.element_dict = OrderedDict(sorted(
                six.iteritems(self.element_dict), key=lambda t: t[1].energy))
        elif option == 'name':
            self.element_dict = OrderedDict(sorted(
                six.iteritems(self.element_dict), key=lambda t: t[0]))
        elif option == 'maxv':
            self.element_dict = OrderedDict(sorted(
                six.iteritems(self.element_dict), key=lambda t: t[1].maxv, reverse=True))

    def add_to_dict(self, dictv):
        """
        This function updates the dictionary element if it already exists.
        """
        self.element_dict.update(dictv)
        logger.debug('Item {} is added.'.format(list(dictv.keys())))
        self.update_norm()

    def update_norm(self, threshv=0.0):
        """
        Calculate the norm intensity for each element peak.

        Parameters
        ----------
        threshv : float
            No value is shown when smaller than the threshold value
        """
        # Do nothing if no elements are selected
        if not self.element_dict:
            return

        # max_dict = reduce(max, map(np.max, six.itervalues(self.element_dict)))
        max_dict = np.max([v.maxv for v in six.itervalues(self.element_dict)])

        for v in six.itervalues(self.element_dict):
            v.norm = v.maxv/max_dict*100
            v.lbd_stat = bool(v.norm > threshv)

        # also delete smaller values
        # there is some bugs in plotting when values < 0.0
        self.delete_value_given_threshold(threshv=threshv)

    def delete_all(self):
        self.element_dict.clear()

    def is_element_in_list(self, element_line_name):
        """
        Check if element 'k' is in the list of selected elements
        """
        if element_line_name in self.element_dict.keys():
            return True
        else:
            return False

    def get_element_list(self):
        current_elements = [v for v
                            in six.iterkeys(self.element_dict)
                            if (v.lower() != v)]

        # logger.info('Current Elements for '
        #            'fitting are {}'.format(current_elements))
        return current_elements

    def update_peak_ratio(self):
        """
        In case users change the max value.
        """
        for v in six.itervalues(self.element_dict):
            max_spectrum = np.max(v.spectrum)
            if not math.isclose(max_spectrum, 0.0, abs_tol=1e-20):
                factor = v.maxv / max_spectrum
            else:
                factor = 0.0
            v.spectrum *= factor
            v.area *= factor
        self.update_norm()

    def turn_on_all(self, option=True):
        """
        Set plotting status on for all lines.
        """
        if option is True:
            _plot = option
        else:
            _plot = False
        for v in six.itervalues(self.element_dict):
            v.status = _plot

    def delete_value_given_threshold(self, threshv=0.1):
        """
        Delete elements smaller than threshold value. Non element
        peaks are not included.
        """
        remove_list = []
        non_element = ['compton', 'elastic', 'background']
        for k, v in six.iteritems(self.element_dict):
            if v.norm < threshv:
                remove_list.append(k)
        for name in remove_list:
            if name in non_element:
                continue
            del self.element_dict[name]

    def delete_unselected_items(self):
        remove_list = []
        for k, v in six.iteritems(self.element_dict):
            if v.status is False:
                remove_list.append(k)
        for name in remove_list:
            del self.element_dict[name]


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
        The list of element lines selected for fitting
    n_selected_elines_for_fitting : Int
        The number of element lines selected for fitting
    n_selected_pure_elines_for_fitting : Int
        The number of element lines selected for fitting
            excluding pileup peaks and user defined peaks.
            Only 'pure' lines like Ca_K, K_K etc.

    """
    default_parameters = Dict()
    data = Typed(np.ndarray)
    prefit_x = Typed(object)
    result_dict = Typed(object)
    result_dict_names = List()
    param_new = Dict()
    total_y = Typed(object)
    # total_l = Dict()
    # total_m = Dict()
    # total_pileup = Dict()
    e_name = Str()        # Element line name selected in combo box
    add_element_intensity = Float(1000.0)
    element_list = List()
    # data_sets = Typed(OrderedDict)
    EC = Typed(object)
    x0 = Typed(np.ndarray)
    y0 = Typed(np.ndarray)
    max_area_dig = Int(2)
    pileup_data = Dict()
    auto_fit_all = Dict()
    bound_val = Float(1.0)

    energy_bound_high_buf = Float(0.0)
    energy_bound_low_buf = Float(0.0)

    n_selected_elines_for_fitting = Int(0)
    n_selected_pure_elines_for_fitting = Int(0)

    def __init__(self, **kwargs):
        try:
            # default parameter is the original parameter, for user to restore
            self.default_parameters = kwargs['default_parameters']
            self.param_new = copy.deepcopy(self.default_parameters)
            self.element_list = get_element(self.param_new)
        except ValueError:
            logger.info('No default parameter files are chosen.')
        self.EC = ElementController()
        self.pileup_data = {'element1': 'Si_K',
                            'element2': 'Si_K',
                            'intensity': 0.0}

        # The following line is part of the fix for automated updating of the energy bound
        #     in 'Automatic Element Finding' dialog box
        self.energy_bound_high_buf = self.param_new['non_fitting_values']['energy_bound_high']['value']
        self.energy_bound_low_buf = self.param_new['non_fitting_values']['energy_bound_low']['value']

    def default_param_update(self, change):
        """
        Observer function to be connected to the fileio model
        in the top-level gui.py startup

        Parameters
        ----------
        changed : dict
            This is the dictionary that gets passed to a function
            with the @observe decorator
        """
        self.default_parameters = change['value']
        self.param_new = copy.deepcopy(self.default_parameters)
        self.element_list = get_element(self.param_new)

        # The following line is part of the fix for automated updating of the energy bound
        #     in 'Automatic Element Finding' dialog box
        self.energy_bound_high_buf = self.param_new['non_fitting_values']['energy_bound_high']['value']
        self.energy_bound_low_buf = self.param_new['non_fitting_values']['energy_bound_low']['value']

    # The following function is part of the fix for automated updating of the energy bound
    #     in 'Automatic Element Finding' dialog box
    @observe('energy_bound_high_buf')
    def _update_energy_bound_high_buf(self, change):
        self.param_new['non_fitting_values']['energy_bound_high']['value'] = change['value']

    @observe('energy_bound_low_buf')
    def _update_energy_bound_high_low(self, change):
        self.param_new['non_fitting_values']['energy_bound_low']['value'] = change['value']

    def param_from_db_update(self, change):
        self.default_parameters = change['value']
        print('update fitting param from db')
        self.update_new_param(self.default_parameters)

    def get_new_param_from_file(self, param_path):
        """
        Update parameters if new param_path is given.

        Parameters
        ----------
        param_path : str
            path to save the file
        """
        with open(param_path, 'r') as json_data:
            self.default_parameters = json.load(json_data)
        self.param_new = copy.deepcopy(self.default_parameters)
        self.element_list = get_element(self.param_new)
        self.EC.delete_all()
        self.define_range()
        self.create_spectrum_from_file(self.param_new, self.element_list)
        logger.info('Elements read from file are: {}'.format(self.element_list))

    def update_new_param(self, param):
        self.default_parameters = param
        self.param_new = copy.deepcopy(self.default_parameters)
        self.element_list = get_element(self.param_new)
        self.EC.delete_all()
        self.define_range()
        self.create_spectrum_from_file(self.param_new, self.element_list)

    def param_changed(self, change):
        """
        Observer function in the top-level gui.py startup

        Parameters
        ----------
        changed : dict
            This is the dictionary that gets passed to a function
            with the @observe decorator
        """
        self.param_new = change['value']

    def exp_data_update(self, change):
        """
        Observer function to be connected to the fileio model
        in the top-level gui.py startup

        Parameters
        ----------
        changed : dict
            This is the dictionary that gets passed to a function
            with the @observe decorator
        """
        self.data = change['value']

    @observe('bound_val')
    def _update_bound(self, change):
        if change['type'] != 'create':
            logger.info('Values smaller than bound {} can be cutted on Auto peak finding.'.format(self.bound_val))

    def define_range(self):
        """
        Cut x range according to values define in param_dict.
        """
        lowv = self.param_new['non_fitting_values']['energy_bound_low']['value']
        highv = self.param_new['non_fitting_values']['energy_bound_high']['value']
        self.x0, self.y0 = define_range(self.data, lowv, highv,
                                        self.param_new['e_offset']['value'],
                                        self.param_new['e_linear']['value'])

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
        self.prefit_x, pre_dict, area_dict = calculate_profile(self.x0,
                                                               self.y0,
                                                               param_dict,
                                                               elemental_lines)
        # add escape peak
        if param_dict['non_fitting_values']['escape_ratio'] > 0:
            pre_dict['escape'] = trim_escape_peak(self.data,
                                                  param_dict, len(self.y0))

        temp_dict = OrderedDict()
        for e in six.iterkeys(pre_dict):
            if e in ['background', 'escape']:
                spectrum = pre_dict[e]

                # summed spectrum here is not correct,
                # as the interval is assumed as 1, not energy interval
                # however area of background and escape is not used elsewhere, not important
                area = np.sum(spectrum)

                ps = PreFitStatus(z=get_Z(e), energy=get_energy(e),
                                  area=float(area), spectrum=spectrum,
                                  maxv=float(np.around(np.max(spectrum), self.max_area_dig)),
                                  norm=-1, lbd_stat=False)
                temp_dict[e] = ps

            elif '-' in e:  # pileup peaks
                e1, e2 = e.split('-')
                energy = float(get_energy(e1))+float(get_energy(e2))
                spectrum = pre_dict[e]
                area = area_dict[e]

                ps = PreFitStatus(z=get_Z(e), energy=str(energy),
                                  area=area, spectrum=spectrum,
                                  maxv=np.around(np.max(spectrum), self.max_area_dig),
                                  norm=-1, lbd_stat=False)
                temp_dict[e] = ps

            else:
                ename = e.split('_')[0]
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
                                      maxv=np.around(np.max(spectrum), self.max_area_dig),
                                      norm=-1, lbd_stat=False)

                    temp_dict[e] = ps
        self.EC.add_to_dict(temp_dict)

    def manual_input(self, userpeak_center=2.5):
        """
        Manually add an emission line (or peak).

        Parameters
        ----------
        userpeak_center: float
            Center of the user defined peak. Ignored if emission line other
            than 'userpeak' is added
        """

        default_area = 1e2

        # if self.e_name == 'escape':
        #     self.param_new['non_fitting_values']['escape_ratio'] = (self.add_element_intensity
        #                                                             / np.max(self.y0))
        #     es_peak = trim_escape_peak(self.data, self.param_new,
        #                                len(self.y0))
        #     ps = PreFitStatus(z=get_Z(self.e_name),
        #                       energy=get_energy(self.e_name),
        #                       # put float in front of area and maxv
        #                       # due to type conflicts in atom, which regards them as
        #                       # np.float32 if we do not put float in front.
        #                       area=float(np.around(np.sum(es_peak), self.max_area_dig)),
        #                       spectrum=es_peak,
        #                       maxv=float(np.around(np.max(es_peak), self.max_area_dig)),
        #                       norm=-1, lbd_stat=False)
        #     logger.info('{} peak is added'.format(self.e_name))
        #
        # else:

        # Add the new data entry to the parameter dictionary. This operation is necessary for 'userpeak'
        #   lines, because they need to be placed to the specific position (by setting 'delta_center'
        #   parameter, while regular element lines are placed to their default positions.
        d_energy = userpeak_center - 5.0

        # PC.params will contain a deepcopy of 'self.param_new' with the new line added
        PC = ParamController(self.param_new, [self.e_name])

        if "userpeak" in self.e_name.lower():
            # Default values for 'delta_center'
            dc = copy.deepcopy(PC.params[f"{self.e_name}_delta_center"])
            # Modify the default values in the dictionary of parameters
            PC.params[f"{self.e_name}_delta_center"]["value"] = d_energy
            PC.params[f"{self.e_name}_delta_center"]["min"] = d_energy - (dc["value"] - dc["min"])
            PC.params[f"{self.e_name}_delta_center"]["max"] = d_energy + (dc["max"] - dc["value"])

        param_tmp = PC.params

        # 'self.param_new' is used to provide 'hint' values for the model, but all active
        #    emission lines in 'elemental_lines' will be included in the model.
        #  The model will contain lines in 'elemental_lines', Compton and elastic
        x, data_out, area_dict = calculate_profile(self.x0,
                                                   self.y0,
                                                   param_tmp,
                                                   elemental_lines=[self.e_name],
                                                   default_area=default_area)

        # Check if element profile was calculated successfully.
        #   Calculation may fail if the selected line is not activated.
        #   The calculation is performed using ``xraylib` library, so there is no
        #   control over it.
        if self.e_name not in data_out:
            raise Exception(f"Failed to add the emission line '{self.e_name}': line is not activated.")

        # If model was generated successfully (the emission line was successfully added), then
        #   make temporary parameters permanent
        self.param_new = param_tmp

        ratio_v = self.add_element_intensity / np.max(data_out[self.e_name])

        ps = PreFitStatus(z=get_Z(self.e_name),
                          energy=get_energy(self.e_name),
                          area=area_dict[self.e_name]*ratio_v,
                          spectrum=data_out[self.e_name]*ratio_v,
                          maxv=self.add_element_intensity,
                          norm=-1,
                          status=True,    # for plotting
                          lbd_stat=False)

        self.EC.add_to_dict({self.e_name: ps})

    def generate_pileup_peak_name(self):
        """
        Returns name for the pileup peak. The element line with the lowest
        energy is placed first in the name.
        """
        # TODO: May be proper error processing should be added
        e1 = float(get_energy(self.pileup_data['element1']))
        e2 = float(get_energy(self.pileup_data['element2']))
        name1 = self.pileup_data['element1']
        name2 = self.pileup_data['element2']
        if e1 > e2:
            name1, name2 = name2, name1
        return name1 + '-' + name2

    def add_pileup(self):
        default_area = 1e2
        if self.pileup_data['intensity'] > 0:
            e_name = self.generate_pileup_peak_name()
            # parse elemental lines into multiple lines

            x, data_out, area_dict = calculate_profile(self.x0,
                                                       self.y0,
                                                       self.param_new,
                                                       elemental_lines=[e_name],
                                                       default_area=default_area)
            energy_float = float(get_energy(self.pileup_data['element1'])) + \
                float(get_energy(self.pileup_data['element2']))
            energy = f"{energy_float:.4f}"

            ratio_v = self.pileup_data['intensity'] / np.max(data_out[e_name])

            ps = PreFitStatus(z=get_Z(e_name),
                              energy=energy,
                              area=area_dict[e_name]*ratio_v,
                              spectrum=data_out[e_name]*ratio_v,
                              maxv=self.pileup_data['intensity'],
                              norm=-1,
                              status=True,    # for plotting
                              lbd_stat=False)
            logger.info('{} peak is added'.format(e_name))
        else:
            raise Exception("Pile-up peak intensity is negative")
        self.EC.add_to_dict({e_name: ps})

    def update_name_list(self):
        """
        When result_dict_names change, the looper in enaml will update.
        """
        # need to clean list first, in order to refresh the list in GUI
        self.result_dict_names = []
        self.result_dict_names = list(self.EC.element_dict.keys())

        peak_list = self.get_user_peak_list()
        # Create the list of selected emission lines such as Ca_K, K_K, etc.
        #   No pileup or user peaks
        pure_peak_list = [n for n in self.result_dict_names if n in peak_list]
        self.n_selected_elines_for_fitting = len(self.result_dict_names)
        self.n_selected_pure_elines_for_fitting = len(pure_peak_list)

        logger.info(f"The full list for fitting is {self.result_dict_names}")

    def find_peak(self, *, threshv=0.1, elemental_lines=None):
        """
        Run automatic peak finding, and save results as dict of object.

        Parameters
        ----------
        threshv: float
            The value will not be shown on GUI if it is smaller than the threshold.

        elemental_lines: list(str)
            The list of elemental lines to find. If ``None``, then all supported
            lines (K, L and M) are searched.
        """
        self.define_range()  # in case the energy calibraiton changes
        self.prefit_x, out_dict, area_dict = linear_spectrum_fitting(self.x0,
                                                                     self.y0,
                                                                     self.param_new,
                                                                     elemental_lines=elemental_lines)
        logger.info('Energy range: {}, {}'.format(
            self.param_new['non_fitting_values']['energy_bound_low']['value'],
            self.param_new['non_fitting_values']['energy_bound_high']['value']))

        prefit_dict = OrderedDict()
        for k, v in six.iteritems(out_dict):
            ps = PreFitStatus(z=get_Z(k),
                              energy=get_energy(k),
                              area=area_dict[k],
                              spectrum=v,
                              maxv=np.around(np.max(v), self.max_area_dig),
                              norm=-1,
                              lbd_stat=False)
            prefit_dict.update({k: ps})

        logger.info('Automatic Peak Finding found elements as : {}'.format(
            list(prefit_dict.keys())))
        self.EC.delete_all()
        self.EC.add_to_dict(prefit_dict)

    def create_full_param(self):
        """
        Extend the param to full param dict including each element's
        information, and assign initial values from pre fit.
        """
        self.define_range()
        self.element_list = self.EC.get_element_list()
        # self.param_new['non_fitting_values']['element_list'] = ', '.join(self.element_list)
        #
        # # first remove some nonexisting elements
        # # remove elements not included in self.element_list
        # self.param_new = param_dict_cleaner(self.param_new,
        #                                     self.element_list)
        #
        # # second add some elements to a full parameter dict
        # # create full parameter list including elements
        # PC = ParamController(self.param_new, self.element_list)
        # # parameter values not updated based on param_new, so redo it
        # param_temp = PC.params
        # for k, v in six.iteritems(param_temp):
        #     if k == 'non_fitting_values':
        #         continue
        #     if self.param_new.has_key(k):
        #         v['value'] = self.param_new[k]['value']
        # self.param_new = param_temp
        #
        # # to create full param dict, for GUI only
        # create_full_dict(self.param_new, fit_strategy_list)

        self.param_new = update_param_from_element(self.param_new, self.element_list)
        element_temp = [e for e in self.element_list if len(e) <= 4]
        pileup_temp = [e for e in self.element_list if '-' in e]
        userpeak_temp = [e for e in self.element_list if 'user' in e.lower()]

        # update area values in param_new according to results saved in ElementController
        if len(self.EC.element_dict):
            for k, v in six.iteritems(self.param_new):
                if 'area' in k:
                    if 'pileup' in k:
                        name_cut = k[7:-5]  # remove pileup_ and _area
                        for p in pileup_temp:
                            if name_cut == p.replace('-', '_'):
                                v['value'] = self.EC.element_dict[p].area
                    elif 'user' in k.lower():
                        for p in userpeak_temp:
                            if p in k:
                                v['value'] = self.EC.element_dict[p].area
                    else:
                        for e in element_temp:
                            k_name, k_line, _ = k.split('_')
                            e_name, e_line = e.split('_')
                            if k_name == e_name and e_line.lower() == k_line[0]:  # attention: S_k and As_k
                                v['value'] = self.EC.element_dict[e].area

            if 'compton' in self.EC.element_dict:
                self.param_new['compton_amplitude']['value'] = self.EC.element_dict['compton'].area
            if 'coherent_sct_amplitude' in self.EC.element_dict:
                self.param_new['coherent_sct_amplitude']['value'] = self.EC.element_dict['elastic'].area

            if 'escape' in self.EC.element_dict:
                self.param_new['non_fitting_values']['escape_ratio'] = (self.EC.element_dict['escape'].maxv
                                                                        / np.max(self.y0))
            else:
                self.param_new['non_fitting_values']['escape_ratio'] = 0.0

    def data_for_plot(self):
        """
        Save data in terms of K, L, M lines for plot.
        """
        self.total_y = None
        self.auto_fit_all = {}

        for k, v in six.iteritems(self.EC.element_dict):
            if v.status is True:
                self.auto_fit_all[k] = v.spectrum
                if self.total_y is None:
                    self.total_y = np.array(v.spectrum)  # need to copy an array
                else:
                    self.total_y += v.spectrum

        # for k, v in six.iteritems(new_dict):
        #     if '-' in k:  # pileup
        #         self.total_pileup[k] = self.EC.element_dict[k].spectrum
        #     elif 'K' in k:
        #         self.total_y[k] = self.EC.element_dict[k].spectrum
        #     elif 'L' in k:
        #         self.total_l[k] = self.EC.element_dict[k].spectrum
        #     elif 'M' in k:
        #         self.total_m[k] = self.EC.element_dict[k].spectrum
        #     else:
        #         self.total_y[k] = self.EC.element_dict[k].spectrum

    def get_user_peak_list(self, *, include_user_peaks=False):
        """
        Returns the list of element emission peaks
        """
        items = K_LINE + L_LINE + M_LINE

        if include_user_peaks:
            userpeak_list = ['Userpeak' + str(i) for i in range(1, 11)]  # 10 users peak
            items += userpeak_list

        return items


def save_as(file_path, data):
    """
    Save full param dict into a file.
    """
    with open(file_path, 'w') as outfile:
        json.dump(data, outfile,
                  sort_keys=True, indent=4)


def define_range(data, low, high, a0, a1):
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
    a0 : float
        offset term of energy calibration
    a1 : float
        linear term of energy calibration

    Returns
    -------
    x : array
        trimmed channel number
    y : array
        trimmed spectrum according to x
    """
    x = np.arange(data.size)

    #  ratio to transfer energy value back to channel value
    # approx_ratio = 100

    low_new = int(np.around((low - a0)/a1))
    high_new = int(np.around((high - a0)/a1))
    x0, y0 = trim(x, data, low_new, high_new)
    return x0, y0


def calculate_profile(x, y, param, elemental_lines,
                      default_area=1e5):
    """
    Calculate the spectrum profile based on given paramters. Use function
    construct_linear_model from xrf_model.

    Parameters
    ----------
    x : array
        channel array
    y : array
        spectrum intensity
    param : dict
        paramters
    elemental_lines : list
        such as Si_K, Pt_M
    required_length : optional, int
        the length of the array might change due to trim process, so
        predifine the length to a given value.
    default_area : float
        default value for the gaussian area of each element

    Returns
    -------
    x : array
        trimmed energy range
    temp_d : dict
        dict of array
    area_dict : dict
        dict of area for elements and other peaks
    """
    # Need to use deepcopy here to avoid unexpected change on parameter dict
    fitting_parameters = copy.deepcopy(param)

    total_list, matv, area_dict = construct_linear_model(x,
                                                         fitting_parameters,
                                                         elemental_lines,
                                                         default_area=default_area)

    temp_d = {k: v for (k, v) in zip(total_list, matv.transpose())}

    # add background
    bg = snip_method(y,
                     fitting_parameters['e_offset']['value'],
                     fitting_parameters['e_linear']['value'],
                     fitting_parameters['e_quadratic']['value'],
                     width=fitting_parameters['non_fitting_values']['background_width'])
    temp_d['background'] = bg

    x_energy = (fitting_parameters['e_offset']['value']
                + fitting_parameters['e_linear']['value'] * x
                + fitting_parameters['e_quadratic']['value'] * x**2)

    return x_energy, temp_d, area_dict


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


def create_full_dict(param, name_list,
                     fixed_list=['adjust_element2', 'adjust_element3']):
    """
    Create full param dict so each item has same nested dict.
    This is for GUI purpose only.

    Pamameters
    ----------
    param : dict
        all parameters including element
    name_list : list
        strategy names

    Returns
    -------
    dict: with update
    """
    param_new = copy.deepcopy(param)
    for n in name_list:
        for k, v in six.iteritems(param_new):
            if k == 'non_fitting_values':
                continue
            if n not in v:

                # enforce newly created parameter to be fixed
                # for strategy in fixed_list
                if n in fixed_list:
                    v.update({n: 'fixed'})
                else:
                    v.update({n: v['bound_type']})
    return param_new


def strip_line(ename):
    return ename.split('_')[0]


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

    non_element = ['compton', 'elastic', 'background', 'escape']
    if (ename.lower() in non_element) or '-' in ename or 'user' in ename.lower():
        return '-'
    else:
        e = Element(strip_line(ename))
        return str(e.Z)


def get_energy(ename):
    """
    Return energy value by given elemental name. Need to consider non-elemental case.
    """
    non_element = ['compton', 'elastic', 'background', 'escape']
    if (ename.lower() in non_element) or 'user' in ename.lower():
        return '-'
    else:
        e = Element(strip_line(ename))
        ename = ename.lower()
        if '_k' in ename:
            energy = e.emission_line['ka1']
        elif '_l' in ename:
            energy = e.emission_line['la1']
        elif '_m' in ename:
            energy = e.emission_line['ma1']

        return str(np.around(energy, 4))


def get_element(param):
    element_list = param['non_fitting_values']['element_list']
    return [e.strip(' ') for e in element_list.split(',')]


def param_dict_cleaner(parameter, element_list):
    """
    Make sure param only contains element from element_list.

    Parameters
    ----------
    parameter : dict
        fitting parameters
    element_list : list
        list of elemental lines

    Returns
    -------
    dict :
        new param dict containing given elements
    """
    param = copy.deepcopy(parameter)
    param_new = {}

    elist_lower = [e.lower() for e in element_list if len(e) <= 4]
    pileup_list = [e for e in element_list if '-' in e]
    userpeak_list = [e for e in element_list if 'user' in e.lower()]

    for k, v in six.iteritems(param):
        if k == 'non_fitting_values' or k == k.lower():
            param_new.update({k: v})
        elif 'pileup' in k:
            for p in pileup_list:
                if p.replace('-', '_') in k:
                    param_new.update({k: v})
        elif 'user' in k.lower():
            for p in userpeak_list:
                if p in k:
                    param_new.update({k: v})
        elif (k[:3].lower() in elist_lower) or (k[:4].lower() in elist_lower):
            param_new.update({k: v})

    return param_new


def update_param_from_element(param, element_list):
    """
    Clean up or extend param according to new element list.

    Parameters
    ----------
    param : dict
        fitting parameters
    element_list : list
        list of elemental lines

    Returns
    -------
    dict
    """
    param_new = copy.deepcopy(param)

    param_new['non_fitting_values']['element_list'] = ', '.join(element_list)

    # first remove some items not included in element_list
    param_new = param_dict_cleaner(param_new,
                                   element_list)

    # second add some elements to a full parameter dict
    # create full parameter list including elements
    PC = ParamController(param_new, element_list)
    # parameter values not updated based on param_new, so redo it
    param_temp = PC.params

    #  enforce adjust_element area to be fixed,
    #  while bound_type in xrf_model is defined as none for area
    # for k, v in six.iteritems(param_temp):
    #     if '_area' in k:
    #         v['bound_type'] = 'fixed'

    for k, v in six.iteritems(param_temp):
        if k == 'non_fitting_values':
            continue
        if k in param_new:
            param_temp[k] = param_new[k]
            # for k1 in six.iterkeys(v):
            #     v[k1] = param_new[k][k1]
    param_new = param_temp

    # to create full param dict, for GUI only
    param_new = create_full_dict(param_new, fit_strategy_list)
    return param_new
