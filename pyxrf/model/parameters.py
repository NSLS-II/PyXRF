from __future__ import absolute_import, division, print_function

import numpy as np
import json
from collections import OrderedDict
import copy
import math

from atom.api import Atom, Str, observe, Typed, Int, Dict, List, Float, Bool

from skbeam.fluorescence import XrfElement as Element
from skbeam.core.fitting.xrf_model import (
    ParamController,
    compute_escape_peak,
    trim,
    construct_linear_model,
    linear_spectrum_fitting,
)
from skbeam.core.fitting.xrf_model import K_LINE, L_LINE, M_LINE
from ..core.map_processing import snip_method_numba
from ..core.xrf_utils import check_if_eline_supported, get_eline_parameters, get_element_atomic_number

from ..core.utils import gaussian_sigma_to_fwhm, gaussian_fwhm_to_sigma

import logging

logger = logging.getLogger(__name__)


bound_options = ["none", "lohi", "fixed", "lo", "hi"]
fit_strategy_list = [
    "fit_with_tail",
    "free_more",
    "e_calibration",
    "linear",
    "adjust_element1",
    "adjust_element2",
    "adjust_element3",
]
autofit_param = ["e_offset", "e_linear", "fwhm_offset", "fwhm_fanoprime"]


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
        norm value in respect to the strongest peak
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
            logger.debug("Item {} is deleted.".format(k))
        except KeyError:
            pass

    def order(self, option="z"):
        """
        Order dict in different ways.
        """
        if option == "z":
            self.element_dict = OrderedDict(sorted(self.element_dict.items(), key=lambda t: t[1].z))
        elif option == "energy":
            self.element_dict = OrderedDict(sorted(self.element_dict.items(), key=lambda t: t[1].energy))
        elif option == "name":
            self.element_dict = OrderedDict(sorted(self.element_dict.items(), key=lambda t: t[0]))
        elif option == "maxv":
            self.element_dict = OrderedDict(
                sorted(self.element_dict.items(), key=lambda t: t[1].maxv, reverse=True)
            )

    def add_to_dict(self, dictv):
        """
        This function updates the dictionary element if it already exists.
        """
        self.element_dict.update(dictv)
        logger.debug("Item {} is added.".format(list(dictv.keys())))
        self.update_norm()

    def update_norm(self, threshv=0.0):
        """
        Calculate the normalized intensity for each element peak.

        Parameters
        ----------
        threshv : float
            No value is shown when smaller than the threshold value
        """
        # Do nothing if no elements are selected
        if not self.element_dict:
            return

        max_dict = np.max([v.maxv for v in self.element_dict.values()])

        for v in self.element_dict.values():
            v.norm = v.maxv / max_dict * 100
            v.lbd_stat = bool(v.norm > threshv)

        # also delete smaller values
        # there is some bugs in plotting when values < 0.0
        self.delete_peaks_below_threshold(threshv=threshv)

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
        current_elements = [v for v in self.element_dict.keys() if (v.lower() != v)]

        # logger.info('Current Elements for '
        #            'fitting are {}'.format(current_elements))
        return current_elements

    def update_peak_ratio(self):
        """
        If 'maxv' is modified, then the values of 'area' and 'spectrum' are adjusted accordingly:
        (1) maximum of spectrum is set equal to 'maxv'; (2) 'area' is scaled proportionally.
        It is important that only 'maxv' is changed before this function is called.
        """
        for v in self.element_dict.values():
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
        for v in self.element_dict.values():
            v.status = _plot

    def delete_peaks_below_threshold(self, threshv=0.1):
        """
        Delete elements smaller than threshold value. Non element
        peaks are not included.
        """
        remove_list = []
        non_element = ["compton", "elastic", "background"]
        for k, v in self.element_dict.items():
            if math.isnan(v.norm) or (v.norm >= threshv) or (k in non_element):
                continue
            # We don't want to delete userpeaks or pileup peaks (they are always added manually).
            if ("-" in k) or (k.lower().startswith("userpeak")):
                continue
            remove_list.append(k)
        for name in remove_list:
            del self.element_dict[name]
        return remove_list

    def delete_unselected_items(self):
        remove_list = []
        non_element = ["compton", "elastic", "background"]
        for k, v in self.element_dict.items():
            if k in non_element:
                continue
            if v.status is False:
                remove_list.append(k)
        for name in remove_list:
            del self.element_dict[name]
        return remove_list


class ParamModel(Atom):
    """
    The module used for maintain the set of fitting parameters.

    Attributes
    ----------
    parameters : `atom.Dict`
        A list of `Parameter` objects, subclassed from the `Atom` base class.
        These `Parameter` objects hold all relevant xrf information.
    data : array
        1D array of spectrum
    prefit_x : array
        xX axis with range defined by low and high limits.
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

    # Reference to FileIOModel object
    io_model = Typed(object)

    default_parameters = Dict()
    # data = Typed(np.ndarray)
    prefit_x = Typed(object)
    result_dict_names = List()
    param_new = Dict()
    total_y = Typed(object)
    # total_l = Dict()
    # total_m = Dict()
    # total_pileup = Dict()
    e_name = Str()  # Element line name selected in combo box
    add_element_intensity = Float(1000.0)
    element_list = List()
    # data_sets = Typed(OrderedDict)
    EC = Typed(object)
    x0 = Typed(np.ndarray)
    y0 = Typed(np.ndarray)
    max_area_dig = Int(2)
    auto_fit_all = Dict()
    bound_val = Float(1.0)

    energy_bound_high_buf = Float(0.0)
    energy_bound_low_buf = Float(0.0)

    n_selected_elines_for_fitting = Int(0)
    n_selected_pure_elines_for_fitting = Int(0)

    parameters_changed_cb = List()

    def __init__(self, *, default_parameters, io_model):
        try:
            self.io_model = io_model

            self.default_parameters = default_parameters
            self.param_new = copy.deepcopy(default_parameters)
            # TODO: do we set 'element_list' as a list of keys of 'EC.element_dict'
            self.element_list = get_element_list(self.param_new)
        except ValueError:
            logger.info("No default parameter files are chosen.")
        self.EC = ElementController()

        # The following line is part of the fix for automated updating of the energy bound
        #     in 'Automatic Element Finding' dialog box
        self.energy_bound_high_buf = self.param_new["non_fitting_values"]["energy_bound_high"]["value"]
        self.energy_bound_low_buf = self.param_new["non_fitting_values"]["energy_bound_low"]["value"]

    def add_parameters_changed_cb(self, cb):
        """
        Add callback to the list of callback function that are called after parameters are updated.
        """
        self.parameters_changed_cb.append(cb)

    def remove_parameters_changed_cb(self, cb):
        """
        Remove reference from the list of callback functions.
        """
        self.parameters_changed_cb = [_ for _ in self.parameters_changed_cb if _ != cb]

    def parameters_changed(self):
        """
        Run callback functions in the list. This method is expected to be called after the parameters
        are update to initiate necessary updates in the GUI.
        """
        for cb in self.parameters_changed_cb:
            cb()

    def default_param_update(self, default_parameters):
        """
        Replace the reference to the dictionary of default parameters.

        Parameters
        ----------
        default_parameters : dict
            Reference to complete and valid dictionary of default parameters.
        """
        self.default_parameters = default_parameters

    # The following function is part of the fix for automated updating of the energy bound
    #     in 'Automatic Element Finding' dialog box
    @observe("energy_bound_high_buf")
    def _update_energy_bound_high_buf(self, change):
        self.param_new["non_fitting_values"]["energy_bound_high"]["value"] = change["value"]
        self.define_range()

    @observe("energy_bound_low_buf")
    def _update_energy_bound_high_low(self, change):
        self.param_new["non_fitting_values"]["energy_bound_low"]["value"] = change["value"]
        self.define_range()

    def get_new_param_from_file(self, param_path):
        """
        Update parameters if new param_path is given.

        Parameters
        ----------
        param_path : str
            path to save the file
        """
        with open(param_path, "r") as json_data:
            self.param_new = json.load(json_data)
        self.create_spectrum_from_param_dict(reset=True)

        logger.info("Elements read from file are: {}".format(self.element_list))

    def update_new_param(self, param, reset=True):
        """
        Update the parameters based on the dictionary of parameters. Set ``reset=False``
        if selection status of elemental lines should be kept.

        Parameters
        ----------
        param : dict
            new dictionary of parameters
        reset : boolean
            reset (``True``) or clear (``False``) selection status of the element lines.
        """
        self.param_new = param
        self.create_spectrum_from_param_dict(reset=reset)

    @observe("bound_val")
    def _update_bound(self, change):
        if change["type"] != "create":
            logger.info(f"Peaks with values than the threshold {self.bound_val} will be removed from the list.")

    def define_range(self):
        """
        Cut x range according to values define in param_dict.
        """
        if self.io_model.data is None:
            return
        lowv = self.param_new["non_fitting_values"]["energy_bound_low"]["value"]
        highv = self.param_new["non_fitting_values"]["energy_bound_high"]["value"]
        self.x0, self.y0 = define_range(
            self.io_model.data,
            lowv,
            highv,
            self.param_new["e_offset"]["value"],
            self.param_new["e_linear"]["value"],
        )

    def create_spectrum_from_param_dict(self, reset=True):
        """
        Create spectrum profile with based on the current set of parameters.
        (``self.param_new`` -> ``self.EC`` and ``self.element_list``).
        Typical use: update self.param_new, then call this function.
        Set ``reset=False`` to keep selection status of the elemental lines.

        Parameters
        ----------
        reset : boolean
            clear or keep status of the elemental lines (in ``self.EC``).
        """
        param_dict = self.param_new
        self.element_list = get_element_list(param_dict)

        self.define_range()
        self.prefit_x, pre_dict, area_dict = calculate_profile(self.x0, self.y0, param_dict, self.element_list)
        # add escape peak
        if param_dict["non_fitting_values"]["escape_ratio"] > 0:
            pre_dict["escape"] = trim_escape_peak(self.io_model.data, param_dict, len(self.y0))

        temp_dict = OrderedDict()
        for e in pre_dict.keys():
            if e in ["background", "escape"]:
                spectrum = pre_dict[e]

                # summed spectrum here is not correct,
                # as the interval is assumed as 1, not energy interval
                # however area of background and escape is not used elsewhere, not important
                area = np.sum(spectrum)

                ps = PreFitStatus(
                    z=get_Z(e),
                    energy=get_energy(e),
                    area=float(area),
                    spectrum=spectrum,
                    maxv=float(np.around(np.max(spectrum), self.max_area_dig)),
                    norm=-1,
                    status=True,
                    lbd_stat=False,
                )
                temp_dict[e] = ps

            elif "-" in e:  # pileup peaks
                energy = self.get_pileup_peak_energy(e)
                energy = f"{energy:.4f}"
                spectrum = pre_dict[e]
                area = area_dict[e]

                ps = PreFitStatus(
                    z=get_Z(e),
                    energy=str(energy),
                    area=area,
                    spectrum=spectrum,
                    maxv=np.around(np.max(spectrum), self.max_area_dig),
                    norm=-1,
                    status=True,
                    lbd_stat=False,
                )
                temp_dict[e] = ps

            else:
                ename = e.split("_")[0]
                for k, v in param_dict.items():
                    energy = get_energy(e)  # For all peaks except Userpeaks

                    if ename in k and "area" in k:
                        spectrum = pre_dict[e]
                        area = area_dict[e]

                    elif ename == "compton" and k == "compton_amplitude":
                        spectrum = pre_dict[e]
                        area = area_dict[e]

                    elif ename == "elastic" and k == "coherent_sct_amplitude":
                        spectrum = pre_dict[e]
                        area = area_dict[e]

                    elif self.get_eline_name_category(ename) == "userpeak":
                        key = ename + "_delta_center"
                        energy = param_dict[key]["value"] + 5.0
                        energy = f"{energy:.4f}"
                    else:
                        continue

                    ps = PreFitStatus(
                        z=get_Z(ename),
                        energy=energy,
                        area=area,
                        spectrum=spectrum,
                        maxv=np.around(np.max(spectrum), self.max_area_dig),
                        norm=-1,
                        status=True,
                        lbd_stat=False,
                    )

                    temp_dict[e] = ps

        # Copy element status
        if not reset:
            element_status = {_: self.EC.element_dict[_].status for _ in self.EC.element_dict}

        self.EC.delete_all()
        self.EC.add_to_dict(temp_dict)

        if not reset:
            for key in self.EC.element_dict.keys():
                if key in element_status:
                    self.EC.element_dict[key].status = element_status[key]

        self.result_dict_names = list(self.EC.element_dict.keys())

    def get_selected_eline_energy_fwhm(self, eline):
        """
        Returns values of energy and fwhm for the peak 'eline' from the dictionary `self.param_new`.
        The emission line must exist in the dictionary. Primarily intended for use
        with user-defined peaks.

        Parameters
        ----------
        eline: str
            emission line (e.g. Ca_K) or peak name (e.g. Userpeak2, V_Ka1-Co_Ka1)
        """
        if eline not in self.EC.element_dict:
            raise ValueError(f"Emission line '{eline}' is not in the list of selected lines.")

        keys = self._generate_param_keys(eline)
        if not keys["key_dcenter"] or not keys["key_dsigma"]:
            raise ValueError(f"Failed to generate keys for the emission line '{eline}'.")

        energy = self.param_new[keys["key_dcenter"]]["value"] + 5.0
        dsigma = self.param_new[keys["key_dsigma"]]["value"]
        fwhm = gaussian_sigma_to_fwhm(dsigma) + self._compute_fwhm_base(energy)
        return energy, fwhm

    def get_pileup_peak_energy(self, eline):
        """
        Returns the energy (center) of pileup peak. Returns None if there is an error.

        Parameters
        ----------
        eline: str
            Name of the pileup peak, e.g. V_Ka1-Co_Ka1

        Returns
        -------
        float or None
            Energy in keV or None
        """
        incident_energy = self.param_new["coherent_sct_energy"]["value"]
        try:
            element_line1, element_line2 = eline.split("-")
            e1_cen = get_eline_parameters(element_line1, incident_energy)["energy"]
            e2_cen = get_eline_parameters(element_line2, incident_energy)["energy"]
            en = e1_cen + e2_cen
        except Exception:
            en = None
        return en

    def add_peak_manual(self, userpeak_center=2.5):
        """
        Manually add an emission line (or peak).

        Parameters
        ----------
        userpeak_center: float
            Center of the user defined peak. Ignored if emission line other
            than 'userpeak' is added
        """
        self._manual_input(userpeak_center=userpeak_center)
        self.update_name_list()
        self.data_for_plot()

    def remove_peak_manual(self):
        """
        Manually add an emission line (or peak). The name emission line (peak) to be deleted
        must be writtent to `self.e_name` before calling the function.
        """
        if self.e_name not in self.EC.element_dict:
            msg = (
                f"Line '{self.e_name}' is not in the list of selected lines,\n"
                f"therefore it can not be deleted from the list."
            )
            raise RuntimeError(msg)

        # Update parameter list
        self._remove_parameters_for_eline(self.e_name)

        # Update EC
        self.EC.delete_item(self.e_name)
        self.EC.update_peak_ratio()
        self.update_name_list()
        self.data_for_plot()

    def remove_elements_below_threshold(self, threshv=None):
        if threshv is None:
            threshv = self.bound_val

        deleted_elements = self.EC.delete_peaks_below_threshold(threshv=threshv)
        for eline in deleted_elements:
            self._remove_parameters_for_eline(eline)

        self.EC.update_peak_ratio()
        self.update_name_list()
        self.data_for_plot()

    def remove_elements_unselected(self):
        deleted_elements = self.EC.delete_unselected_items()
        for eline in deleted_elements:
            self._remove_parameters_for_eline(eline)

        self.EC.update_peak_ratio()
        self.update_name_list()
        self.data_for_plot()

    def _remove_parameters_for_eline(self, eline):
        """Remove entries for `eline` from the dictionary `self.param_new`"""
        if self.get_eline_name_category(eline) == "pileup":
            key_prefix = "pileup_" + self.e_name.replace("-", "_")
        else:
            key_prefix = eline

        # It is sufficient to compare using lowercase. It could be more reliable.
        key_prefix = key_prefix.lower()
        keys_to_delete = [_ for _ in self.param_new.keys() if _.lower().startswith(key_prefix)]
        for key in keys_to_delete:
            del self.param_new[key]

        # Add name to the name list
        _remove_element_from_list(eline, self.param_new)

    def _manual_input(self, userpeak_center=2.5):
        """
        Manually add an emission line (or peak).

        Parameters
        ----------
        userpeak_center: float
            Center of the user defined peak. Ignored if emission line other
            than 'userpeak' is added
        """

        if self.e_name in self.EC.element_dict:
            msg = f"Line '{self.e_name}' is in the list of selected lines. \nDuplicate entries are not allowed."
            raise RuntimeError(msg)

        default_area = 1e2

        # Add the new data entry to the parameter dictionary. This operation is necessary for 'userpeak'
        #   lines, because they need to be placed to the specific position (by setting 'delta_center'
        #   parameter, while regular element lines are placed to their default positions.
        d_energy = userpeak_center - 5.0

        # PC.params will contain a deepcopy of 'self.param_new' with the new line added
        PC = ParamController(self.param_new, [self.e_name])

        if self.get_eline_name_category(self.e_name) == "userpeak":
            energy = userpeak_center
            # Default values for 'delta_center'
            dc = copy.deepcopy(PC.params[f"{self.e_name}_delta_center"])
            # Modify the default values in the dictionary of parameters
            PC.params[f"{self.e_name}_delta_center"]["value"] = d_energy
            PC.params[f"{self.e_name}_delta_center"]["min"] = d_energy - (dc["value"] - dc["min"])
            PC.params[f"{self.e_name}_delta_center"]["max"] = d_energy + (dc["max"] - dc["value"])
        elif self.get_eline_name_category(self.e_name) == "pileup":
            energy = self.get_pileup_peak_energy(self.e_name)
        else:
            energy = get_energy(self.e_name)

        param_tmp = PC.params
        param_tmp = create_full_dict(param_tmp, fit_strategy_list)

        # Add name to the name list
        _add_element_to_list(self.e_name, param_tmp)

        # 'self.param_new' is used to provide 'hint' values for the model, but all active
        #    emission lines in 'elemental_lines' will be included in the model.
        #  The model will contain lines in 'elemental_lines', Compton and elastic
        x, data_out, area_dict = calculate_profile(
            self.x0, self.y0, param_tmp, elemental_lines=[self.e_name], default_area=default_area
        )

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

        ps = PreFitStatus(
            z=get_Z(self.e_name),
            energy=energy if isinstance(energy, str) else f"{energy:.4f}",
            area=area_dict[self.e_name] * ratio_v,
            spectrum=data_out[self.e_name] * ratio_v,
            maxv=self.add_element_intensity,
            norm=-1,
            status=True,  # for plotting
            lbd_stat=False,
        )

        self.EC.add_to_dict({self.e_name: ps})
        self.EC.update_peak_ratio()

    def _generate_param_keys(self, eline):
        """
        Returns prefix of the key from `param_new` dictionary based on emission line name
        If eline is actual emission line (like Ca_K), then the `key_dcenter` and `key_dsigma`
        point to 'a1' line (Ca_ka1). Function has to be extended if access to specific lines is
        required. Function is primarily intended for use with user-defined peaks.
        """

        category = self.get_eline_name_category(eline)
        if category == "pileup":
            eline = eline.replace("-", "_")
            key_area = "pileup_" + eline + "_area"
            key_dcenter = "pileup_" + eline + "delta_center"
            key_dsigma = "pileup_" + eline + "delta_sigma"
        elif category == "eline":
            eline = eline[:-1] + eline[-1].lower()
            key_area = eline + "a1_area"
            key_dcenter = eline + "a1_delta_center"
            key_dsigma = eline + "a1_delta_sigma"
        elif category == "userpeak":
            key_area = eline + "_area"
            key_dcenter = eline + "_delta_center"
            key_dsigma = eline + "_delta_sigma"
        elif eline == "compton":
            key_area = eline + "_amplitude"
            key_dcenter, key_dsigma = "", ""
        else:
            # No key exists (for "background", "escape", "elastic")
            key_area, key_dcenter, key_dsigma = "", "", ""
        return {"key_area": key_area, "key_dcenter": key_dcenter, "key_dsigma": key_dsigma}

    def modify_peak_height(self, maxv_new):
        """
        Modify the height of the emission line.

        Parameters
        ----------
        new_maxv: float
            New maximum value for the emission line `self.e_name`
        """

        ignored_peaks = {"escape"}
        if self.e_name in ignored_peaks:
            msg = f"Height of the '{self.e_name}' peak can not be changed."
            raise RuntimeError(msg)

        if self.e_name not in self.EC.element_dict:
            msg = (
                f"Attempt to modify maximum value for the emission line '{self.e_name},'\n"
                f"which is not currently selected."
            )
            raise RuntimeError(msg)

        key = self._generate_param_keys(self.e_name)["key_area"]
        maxv_current = self.EC.element_dict[self.e_name].maxv

        coef = maxv_new / maxv_current if maxv_current > 0 else 0
        # Only 'maxv' needs to be updated.
        self.EC.element_dict[self.e_name].maxv = maxv_new
        # The following function updates 'spectrum', 'area' and 'norm'.
        self.EC.update_peak_ratio()

        # Some of the parameters are represented only in EC, not in 'self.param_new'.
        #   (particularly "background" and "elastic")
        if key:
            self.param_new[key]["value"] *= coef

    def _compute_fwhm_base(self, energy):
        # Computes 'sigma' value based on default parameters and peak energy (for Userpeaks)
        #   does not include corrections for fwhm.
        # If both peak center (energy) and fwhm is updated, energy needs to be set first,
        #   since it is used in computation of ``fwhm_base``

        sigma = gaussian_fwhm_to_sigma(self.param_new["fwhm_offset"]["value"])

        sigma_sqr = energy + 5.0  # center
        sigma_sqr *= self.param_new["non_fitting_values"]["epsilon"]  # epsilon
        sigma_sqr *= self.param_new["fwhm_fanoprime"]["value"]  # fanoprime
        sigma_sqr += sigma * sigma  # We have computed the expression under sqrt

        sigma_total = np.sqrt(sigma_sqr)

        return gaussian_sigma_to_fwhm(sigma_total)

    def _update_userpeak_energy(self, eline, energy_new, fwhm_new):
        """
        Set new center for the user-defined peak at 'new_energy'
        """

        # According to the accepted peak model, as energy of the peak center grows,
        #   the peak becomes wider. The most user friendly solution is to automatically
        #   increase FWHM as the peak moves along the energy axis to the right and
        #   decrease otherwise. So generally, the user should first place the peak
        #   center at the desired energy, and then adjust FWHM.

        # We change energy, so we will have to change FWHM as well
        #  so before updating energy we will save the difference between
        #  the default (base) FWHM and the displayed FWHM

        name_userpeak_dcenter = eline + "_delta_center"
        old_energy = self.param_new[name_userpeak_dcenter]["value"]

        # This difference represents the required change in fwhm
        fwhm_difference = fwhm_new - self._compute_fwhm_base(old_energy)

        # Now we change energy.
        denergy = energy_new - 5.0

        v_center = self.param_new[name_userpeak_dcenter]["value"]
        v_max = self.param_new[name_userpeak_dcenter]["max"]
        v_min = self.param_new[name_userpeak_dcenter]["min"]
        # Keep the possible range for value change the same
        self.param_new[name_userpeak_dcenter]["value"] = denergy
        self.param_new[name_userpeak_dcenter]["max"] = denergy + v_max - v_center
        self.param_new[name_userpeak_dcenter]["min"] = denergy - (v_center - v_min)

        # The base value is updated now (since the energy has changed)
        fwhm_base = self._compute_fwhm_base(energy_new)
        fwhm = fwhm_difference + fwhm_base

        return fwhm

    def _update_userpeak_fwhm(self, eline, energy_new, fwhm_new):
        name_userpeak_dsigma = eline + "_delta_sigma"

        fwhm_base = self._compute_fwhm_base(energy_new)
        dfwhm = fwhm_new - fwhm_base

        dsigma = gaussian_fwhm_to_sigma(dfwhm)

        v_center = self.param_new[name_userpeak_dsigma]["value"]
        v_max = self.param_new[name_userpeak_dsigma]["max"]
        v_min = self.param_new[name_userpeak_dsigma]["min"]
        # Keep the possible range for value change the same
        self.param_new[name_userpeak_dsigma]["value"] = dsigma
        self.param_new[name_userpeak_dsigma]["max"] = dsigma + v_max - v_center
        self.param_new[name_userpeak_dsigma]["min"] = dsigma - (v_center - v_min)

    def _update_userpeak_energy_fwhm(self, eline, fwhm_new, energy_new):
        """
        Update energy and fwhm of the user-defined peak 'eline'. The 'delta_center'
        and 'delta_sigma' parameters in the `self.param_new` dictionary are updated.
        `area` should be updated after call to this function. This function also
        doesn't change entries in the `EC` dictionary.
        """
        # Ensure, that the values are greater than some small value to ensure that
        #   there is no computational problems.
        # Energy resolution for the existing beamlines is 0.01 keV, so 0.001 is small
        #   enough both for center energy and FWHM.
        energy_small_value = 0.001
        energy_new = max(energy_new, energy_small_value)
        fwhm_new = max(fwhm_new, energy_small_value)

        fwhm_new = self._update_userpeak_energy(eline, energy_new, fwhm_new)
        self._update_userpeak_fwhm(eline, energy_new, fwhm_new)

    def modify_userpeak_params(self, maxv_new, fwhm_new, energy_new):

        if self.get_eline_name_category(self.e_name) != "userpeak":
            msg = (
                f"Hight and width can be modified only for a user defined peak.\n"
                f"The function was called for '{self.e_name}' peak"
            )
            raise RuntimeError(msg)

        if self.e_name not in self.EC.element_dict:
            msg = (
                f"Attempt to modify maximum value for the emission line '{self.e_name},'\n"
                f"which is not currently selected."
            )
            raise RuntimeError(msg)

        # Some checks of the input values
        if maxv_new <= 0.0:
            raise ValueError("Peak height must be a positive number greater than 0.001.")
        if energy_new <= 0.0:
            raise ValueError("User peak energy must be a positive number greater than 0.001.")
        if fwhm_new <= 0:
            raise ValueError("User peak FWHM must be a positive number.")

        # Make sure that the energy of the user peak is within the selected fitting range
        energy_bound_high = self.param_new["non_fitting_values"]["energy_bound_high"]["value"]
        energy_bound_low = self.param_new["non_fitting_values"]["energy_bound_low"]["value"]
        if energy_new > energy_bound_high or energy_new < energy_bound_low:
            raise ValueError("User peak energy is outside the selected range.")

        # This updates 'delta_center' and 'delta_sigma' entries of the 'self.param_new' dictionary
        self._update_userpeak_energy_fwhm(self.e_name, fwhm_new, energy_new)

        default_area = 1e2
        key = self._generate_param_keys(self.e_name)["key_area"]

        # Set area to default area, change it later once the area is computed
        self.param_new[key]["value"] = default_area

        # 'self.param_new' is used to provide 'hint' values for the model, but all active
        #    emission lines in 'elemental_lines' will be included in the model.
        #  The model will contain lines in 'elemental_lines', Compton and elastic
        x, data_out, area_dict = calculate_profile(
            self.x0, self.y0, self.param_new, elemental_lines=[self.e_name], default_area=default_area
        )

        ratio_v = maxv_new / np.max(data_out[self.e_name])

        area = area_dict[self.e_name] * ratio_v
        self.param_new[key]["value"] = area

        ps = PreFitStatus(
            z=get_Z(self.e_name),
            energy=f"{energy_new:.4f}",
            area=area,
            spectrum=data_out[self.e_name] * ratio_v,
            maxv=maxv_new,
            norm=-1,
            status=True,  # for plotting
            lbd_stat=False,
        )

        self.EC.element_dict[self.e_name] = ps

        logger.debug(
            f"The parameters of the user defined peak. The new values:\n"
            f"   Energy: {energy_new} keV, FWHM: {fwhm_new}, Maximum: {maxv_new}\n"
        )

    def generate_pileup_peak_name(self, name1, name2):
        """
        Returns name for the pileup peak. The element line with the lowest
        energy is placed first in the name.
        """
        incident_energy = self.param_new["coherent_sct_energy"]["value"]
        e1 = get_eline_parameters(name1, incident_energy)["energy"]
        e2 = get_eline_parameters(name2, incident_energy)["energy"]
        if e1 > e2:
            name1, name2 = name2, name1
        return name1 + "-" + name2

    def update_name_list(self):
        """
        When result_dict_names change, the looper in enaml will update.
        """
        # need to clean list first, in order to refresh the list in GUI
        self.result_dict_names = []
        self.result_dict_names = list(self.EC.element_dict.keys())
        self.element_list = get_element_list(self.param_new)

        peak_list = self.get_user_peak_list()
        # Create the list of selected emission lines such as Ca_K, K_K, etc.
        #   No pileup or user peaks
        pure_peak_list = [n for n in self.result_dict_names if n in peak_list]
        self.n_selected_elines_for_fitting = len(self.result_dict_names)
        self.n_selected_pure_elines_for_fitting = len(pure_peak_list)

        logger.info(f"The update list of emission lines: {self.result_dict_names}")

    def get_eline_name_category(self, eline_name):
        """
        Returns the category to which `eline_name` belongs: `eline`, `userpeak`,
        `pileup` or `other`.

        Parameters
        ----------
        eline_name: str
            Name to be analyzed

        Returns
        -------
        str
            category: one of `("eline", "userpeak", "pileup" or "other")`
        """
        if check_if_eline_supported(eline_name):
            return "eline"
        elif eline_name.lower().startswith("userpeak"):
            return "userpeak"
        elif "-" in eline_name:  # This is specific to currently accepted naming convention
            return "pileup"
        else:
            return "other"

    def _sort_eline_list(self, element_list):
        """
        Sort the list of elements
        """
        names_elines, names_userpeaks, names_pileup_peaks, names_other = [], [], [], []
        for name in element_list:
            if self.get_eline_name_category(name) == "eline":
                try:
                    z = get_element_atomic_number(name.split("_")[0])
                except Exception:
                    z = 0
                names_elines.append([name, z])
            elif self.get_eline_name_category(name) == "userpeak":
                names_userpeaks.append(name)
            elif self.get_eline_name_category(name) == "pileup":
                names_pileup_peaks.append(name)
            else:
                names_other.append(name)

        names_elines.sort(key=lambda v: int(v[1]))  # Sort by Z (atomic number)
        names_elines = [_[0] for _ in names_elines]  # Get rid of Z
        names_userpeaks.sort()
        names_pileup_peaks.sort()
        names_other.sort()

        return names_elines + names_userpeaks + names_pileup_peaks + names_other

    def get_sorted_result_dict_names(self):
        """
        The function returns the list of selected emission lines. The emission lines are
        sorted in the following order: emission line names (sorted in the order of growing
        atomic number Z), userpeaks (in alphabetic order), pileup peaks (in alphabetic order),
        other peaks (in alphabetic order).

        Returns
        -------
        list(str)
            the list if emission line names
        """
        return self._sort_eline_list(self.result_dict_names)

    def get_sorted_element_list(self):
        """
        Returns sorted ``element_list``.
        """
        return self._sort_eline_list(self.element_list)

    def read_param_from_file(self, param_path):
        """
        Update parameters if new param_path is given.

        Parameters
        ----------
        param_path : str
            path to save the file
        """
        with open(param_path, "r") as json_data:
            param = json.load(json_data)
            self.update_new_param(param, reset=True)

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
        self.prefit_x, out_dict, area_dict = linear_spectrum_fitting(
            self.x0, self.y0, self.param_new, elemental_lines=elemental_lines
        )
        logger.info(
            "Energy range: {}, {}".format(
                self.param_new["non_fitting_values"]["energy_bound_low"]["value"],
                self.param_new["non_fitting_values"]["energy_bound_high"]["value"],
            )
        )

        prefit_dict = OrderedDict()
        for k, v in out_dict.items():
            ps = PreFitStatus(
                z=get_Z(k),
                energy=get_energy(k),
                area=area_dict[k],
                spectrum=v,
                maxv=np.around(np.max(v), self.max_area_dig),
                norm=-1,
                lbd_stat=False,
            )
            prefit_dict.update({k: ps})

        logger.info("Automatic Peak Finding found elements as : {}".format(list(prefit_dict.keys())))
        self.EC.delete_all()
        self.EC.add_to_dict(prefit_dict)

        self.create_full_param()

    def create_full_param(self):
        """
        Update current ``self.param_new`` with elements from ``self.EC`` (delete elements that
        are not in ``self.EC`` and update the existing elements.
        """
        self.define_range()
        # We set 'self.element_list' from 'EC' (because we want to set elements of 'self.param_new'
        #   from 'EC.element_dict'
        self.element_list = self.EC.get_element_list()

        self.param_new = update_param_from_element(self.param_new, self.element_list)
        element_temp = [e for e in self.element_list if len(e) <= 4]
        pileup_temp = [e for e in self.element_list if "-" in e]
        userpeak_temp = [e for e in self.element_list if "user" in e.lower()]

        # update area values in param_new according to results saved in ElementController
        if len(self.EC.element_dict):
            for k, v in self.param_new.items():
                if "area" in k:
                    if "pileup" in k:
                        name_cut = k[7:-5]  # remove pileup_ and _area
                        for p in pileup_temp:
                            if name_cut == p.replace("-", "_"):
                                v["value"] = self.EC.element_dict[p].area
                    elif "user" in k.lower():
                        for p in userpeak_temp:
                            if p in k:
                                v["value"] = self.EC.element_dict[p].area
                    else:
                        for e in element_temp:
                            k_name, k_line, _ = k.split("_")
                            e_name, e_line = e.split("_")
                            if k_name == e_name and e_line.lower() == k_line[0]:  # attention: S_k and As_k
                                v["value"] = self.EC.element_dict[e].area

            if "compton" in self.EC.element_dict:
                self.param_new["compton_amplitude"]["value"] = self.EC.element_dict["compton"].area
            if "coherent_sct_amplitude" in self.EC.element_dict:
                self.param_new["coherent_sct_amplitude"]["value"] = self.EC.element_dict["elastic"].area

            if "escape" in self.EC.element_dict:
                self.param_new["non_fitting_values"]["escape_ratio"] = self.EC.element_dict[
                    "escape"
                ].maxv / np.max(self.y0)
            else:
                self.param_new["non_fitting_values"]["escape_ratio"] = 0.0

    def data_for_plot(self):
        """
        Save data in terms of K, L, M lines for plot.
        """
        self.total_y = None
        self.auto_fit_all = {}

        for k, v in self.EC.element_dict.items():
            if v.status is True:
                self.auto_fit_all[k] = v.spectrum
                if self.total_y is None:
                    self.total_y = np.array(v.spectrum)  # need to copy an array
                else:
                    self.total_y += v.spectrum

        # for k, v in new_dict.items():
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

    def get_user_peak_list(self):
        """
        Returns the list of element emission peaks
        """
        return K_LINE + L_LINE + M_LINE

    def get_selected_emission_line_data(self):
        """
        Assembles the full emission line data for processing.

        Returns
        -------
        list(dict)
            Each dictionary includes the following data: "name" (e.g. Ca_ka1 etc.),
            "area" (estimated peak area based on current fitting results), "ratio"
            (ratio such as Ca_ka2/Ca_ka1)
        """
        # Full list of supported emission lines (such as Ca_K)
        supported_elines = self.get_user_peak_list()
        # Parameter keys start with full emission line name (eg. Ca_ka1)
        param_keys = list(self.param_new.keys())

        incident_energy = self.param_new["coherent_sct_energy"]["value"]

        full_line_list = []
        for eline in self.EC.element_dict.keys():
            if eline not in supported_elines:
                continue
            area = self.EC.element_dict[eline].area
            lines = [_ for _ in param_keys if _.lower().startswith(eline.lower())]
            lines = set(["_".join(_.split("_")[:2]) for _ in lines])
            for ln in lines:
                eline_info = get_eline_parameters(ln, incident_energy)
                data = {"name": ln, "area": area, "ratio": eline_info["ratio"], "energy": eline_info["energy"]}
                full_line_list.append(data)
        return full_line_list

    def guess_pileup_peak_components(self, energy, tolerance=0.05):
        """
        Provides a guess on components of pileup peak based on the set of selected emission lines,
        and selected energy.

        Parameters
        ----------
        energy: float
            Approximate (selected) energy of pileup peak location
        tolerance: float
            Allowed deviation of the sum of component energies from the selected energy, keV

        Returns
        -------
        tuple(str, str, float)
            Component emission lines (such as Ca_ka1, K_ka1 etc) and the energy of
            the resulting pileup peak.
        """

        line_data = self.get_selected_emission_line_data()
        energy_min, energy_max = energy - tolerance, energy + tolerance

        # Not very efficient algorithm, which tries all combinations of lines
        pileup_components, areas = [], []
        for n1, line1 in enumerate(line_data):
            for n2 in range(n1, len(line_data)):
                line2 = line_data[n2]
                if energy_min < line1["energy"] + line2["energy"] < energy_max:
                    if line1 == line2:
                        area = line1["area"] * line1["ratio"]
                    else:
                        area = line1["area"] * line1["ratio"] + line2["area"] * line2["ratio"]
                    pileup_components.append((line1["name"], line2["name"], line1["energy"] + line2["energy"]))
                    areas.append(area)

        if len(areas):
            # Find index with maximum area
            n = areas.index(max(areas))
            return pileup_components[n]
        else:
            return None


def save_as(file_path, data):
    """
    Save full param dict into a file.
    """
    with open(file_path, "w") as outfile:
        json.dump(data, outfile, sort_keys=True, indent=4)


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

    low_new = int(np.around((low - a0) / a1))
    high_new = int(np.around((high - a0) / a1))
    x0, y0 = trim(x, data, low_new, high_new)
    return x0, y0


def calculate_profile(x, y, param, elemental_lines, default_area=1e5):
    """
    Calculate the spectrum profile based on given parameters. Use function
    construct_linear_model from xrf_model.

    Parameters
    ----------
    x : array
        channel array
    y : array
        spectrum intensity
    param : dict
        parameters
    elemental_lines : list
        such as Si_K, Pt_M
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

    total_list, matv, area_dict = construct_linear_model(
        x, fitting_parameters, elemental_lines, default_area=default_area
    )

    temp_d = {k: v for (k, v) in zip(total_list, matv.transpose())}

    # add background
    bg = snip_method_numba(
        y,
        fitting_parameters["e_offset"]["value"],
        fitting_parameters["e_linear"]["value"],
        fitting_parameters["e_quadratic"]["value"],
        width=fitting_parameters["non_fitting_values"]["background_width"],
    )
    temp_d["background"] = bg

    x_energy = (
        fitting_parameters["e_offset"]["value"]
        + fitting_parameters["e_linear"]["value"] * x
        + fitting_parameters["e_quadratic"]["value"] * x**2
    )

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
    ratio = param_dict["non_fitting_values"]["escape_ratio"]
    xe, ye = compute_escape_peak(data, ratio, param_dict)
    lowv = param_dict["non_fitting_values"]["energy_bound_low"]["value"]
    highv = param_dict["non_fitting_values"]["energy_bound_high"]["value"]
    xe, es_peak = trim(xe, ye, lowv, highv)
    logger.info("Escape peak is considered with ratio {}".format(ratio))

    # align to the same length
    if y_size > es_peak.size:
        temp = es_peak
        es_peak = np.zeros(y_size)
        es_peak[: temp.size] = temp
    else:
        es_peak = es_peak[:y_size]
    return es_peak


def create_full_dict(param, name_list, fixed_list=["adjust_element2", "adjust_element3"]):
    """
    Create full param dict so each item has the same nested dict.
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
        for k, v in param_new.items():
            if k == "non_fitting_values":
                continue
            if n not in v:
                # enforce newly created parameter to be fixed
                # for strategy in fixed_list
                if n in fixed_list:
                    v.update({n: "fixed"})
                else:
                    v.update({n: v["bound_type"]})
    return param_new


def strip_line(ename):
    return ename.split("_")[0]


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

    non_element = ["compton", "elastic", "background", "escape"]
    if (ename.lower() in non_element) or "-" in ename or "user" in ename.lower():
        return "-"
    else:
        e = Element(strip_line(ename))
        return str(e.Z)


def get_energy(ename):
    """
    Return energy value by given elemental name. Need to consider non-elemental case.
    """
    non_element = ["compton", "elastic", "background", "escape"]
    if (ename.lower() in non_element) or "user" in ename.lower():
        return "-"
    else:
        e = Element(strip_line(ename))
        ename = ename.lower()
        if "_k" in ename:
            energy = e.emission_line["ka1"]
        elif "_l" in ename:
            energy = e.emission_line["la1"]
        elif "_m" in ename:
            energy = e.emission_line["ma1"]

        return str(np.around(energy, 4))


def get_element_list(param):
    """Extract elements from parameter class object"""
    element_list = param["non_fitting_values"]["element_list"]
    element_list = [e.strip(" ") for e in element_list.split(",")]
    # Unfortunately, "".split(",") returns [""] instead of [], but we need [] !!!
    if element_list == [""]:
        element_list = []
    return element_list


def _set_element_list(element_list, param):
    element_list = ", ".join(element_list)
    param["non_fitting_values"]["element_list"] = element_list


def _add_element_to_list(eline, param):
    """Add element to list in the parameter class object"""
    elist = get_element_list(param)
    elist_lower = [_.lower() for _ in elist]
    if eline.lower() not in elist_lower:
        elist.append(eline)
    _set_element_list(elist, param)


def _remove_element_from_list(eline, param):
    """Add element to list in the parameter class object"""
    elist = get_element_list(param)
    elist_lower = [_.lower() for _ in elist]
    try:
        index = elist_lower.index(eline.lower())
        elist.pop(index)
        _set_element_list(elist, param)
    except ValueError:
        pass


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

    elines_list = [e for e in element_list if len(e) <= 4]
    elines_lower = [e.lower() for e in elines_list]
    pileup_list = [e for e in element_list if "-" in e]
    userpeak_list = [e for e in element_list if "user" in e.lower()]

    new_element_set = set()

    for k, v in param.items():
        if k == "non_fitting_values" or k == k.lower():
            param_new.update({k: v})
        elif "pileup" in k:
            for p in pileup_list:
                if p.replace("-", "_") in k:
                    param_new.update({k: v})
                    new_element_set.add(p)
        elif "user" in k.lower():
            for p in userpeak_list:
                if p in k:
                    param_new.update({k: v})
                    new_element_set.add(p)
        elif k[:3].lower() in elines_lower:
            index = elines_lower.index(k[:3].lower())
            param_new.update({k: v})
            new_element_set.add(elines_list[index])
        elif k[:4].lower() in elines_lower:
            index = elines_lower.index(k[:4].lower())
            param_new.update({k: v})
            new_element_set.add(elines_list[index])

    new_element_list = list(new_element_set)
    _set_element_list(new_element_list, param_new)

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

    for eline in element_list:
        _add_element_to_list(eline, param_new)

    # first remove some items not included in element_list
    param_new = param_dict_cleaner(param_new, element_list)

    # second add some elements to a full parameter dict
    # create full parameter list including elements
    PC = ParamController(param_new, element_list)
    # parameter values not updated based on param_new, so redo it
    param_temp = PC.params

    #  enforce adjust_element area to be fixed,
    #  while bound_type in xrf_model is defined as none for area
    # for k, v in param_temp.items():
    #     if '_area' in k:
    #         v['bound_type'] = 'fixed'

    for k, v in param_temp.items():
        if k == "non_fitting_values":
            continue
        if k in param_new:
            param_temp[k] = param_new[k]
            # for k1 in v.keys():
            #     v[k1] = param_new[k][k1]
    param_new = param_temp

    # to create full param dict, for GUI only
    param_new = create_full_dict(param_new, fit_strategy_list)
    return param_new
