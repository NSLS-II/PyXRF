from __future__ import absolute_import, division, print_function

import numpy as np
from collections import OrderedDict
import os
import re

from atom.api import Atom, Str, observe, List, Int, Bool, Typed

from skbeam.fluorescence import XrfElement as Element
from skbeam.core.fitting.xrf_model import K_LINE, L_LINE, M_LINE

from .fileio import save_fitdata_to_hdf
from .fit_spectrum import get_energy_bin_range
from ..core.map_processing import compute_selected_rois, TerminalProgressBar

import logging

logger = logging.getLogger(__name__)


class ROISettings(Atom):
    """
    This class defines basic data structure for roi calculation.

    Attributes
    ----------
    prefix : str
        prefix name
    line_val : float
        emission energy of primary line
    left_val : float
        left boundary
    right_val : float
        right boundary
    default_left : float
    default_right : float
    step : float
        min step value to change
    show_plot : bool
        option to plot
    """

    prefix = Str()
    line_val = Int()
    left_val = Int()
    right_val = Int()
    default_left = Int()
    default_right = Int()
    step = Int(1)
    show_plot = Bool(False)

    @observe("left_val")
    def _value_update(self, change):
        if change["type"] == "create":
            return
        logger.debug("left value is changed {}".format(change))

    @observe("show_plot")
    def _plot_opt(self, change):
        if change["type"] == "create":
            return
        logger.debug("show plot is changed {}".format(change))


class ROIModel(Atom):
    """
    Control roi calculation according to given inputs.

    Parameters
    ----------
    parameters : Dict
        parameter values used for fitting
    data_dict : Dict
        dict of 3D data
    element_for_roi : str
        inputs given by users
    element_list_roi : list
        list of elements after parsing
    roi_dict : dict
        dict of ROISettings object
    enable_roi_computation : Bool
        enables/disables GUI element that start ROI computation
        At least one element must be selected and all entry in the element
          list must be valid before ROI may be computed

    result_folder : Str
        directory which contains HDF5 file, in which results of processing are saved
    hdf_path : Str
        full path to the HDF5 file, in which results are saved
    hdf_name : Str
        name of the HDF file, in which results are saved

    data_title : str
        The title of the selected dataset (from ``fileio`` module)
    data_title_base : str
        The title changed for internal use (suffix is removed)
    data_title_adjusted : str
        The title changed for internal use (suffix 'sum' is removed if it exists)
    suffix_name_roi : str
        The suffix may have values 'sum', 'det1', 'det2' etc.
    """

    # Reference to ParamModel object
    param_model = Typed(object)
    # Reference to FileIOModel object
    io_model = Typed(object)

    element_for_roi = Str()
    element_list_roi = List()
    roi_dict = OrderedDict()
    enable_roi_computation = Bool(False)

    subtract_background = Bool(False)

    result_folder = Str()

    hdf_path = Str()
    hdf_name = Str()

    data_title = Str()
    data_title_base = Str()
    data_title_adjusted = Str()
    suffix_name_roi = Str()

    def filename_update(self, change):
        """
        Observer function to be connected to the fileio model
        in the top-level gui.py startup

        Parameters
        ----------
        changed : dict
            This is the dictionary that gets passed to a function
            with the @observe decorator
        """
        self.hdf_name = change["value"]
        # output to .h5 file
        self.hdf_path = os.path.join(self.result_folder, self.hdf_name)

    def result_folder_changed(self, change):
        """
        Observer function to be connected to the fileio model
        in the top-level gui.py startup

        Parameters
        ----------
        changed : dict
            This is the dictionary that gets passed to a function
            with the @observe decorator
        """
        self.result_folder = change["value"]

    def data_title_update(self, change):
        """
        Observer function to be connected to the fileio model
        in the top-level gui.py startup

        Parameters
        ----------
        changed : dict
            This is the dictionary that gets passed to a function
            with the @observe decorator
        """
        self.data_title = change["value"]

        # It is assumed, that ``self.data_title`` was created in the ``fileio`` module
        #   and has dataset label attached to the end of it.
        #   The labels are ``sum``, ``det1``, ``det2`` etc. depending on the number
        #   of detector channels.
        self.suffix_name_roi = self.data_title.split("_")[-1]

        self.data_title_base = "_".join(self.data_title.split("_")[:-1])

        if self.suffix_name_roi == "sum":
            # If suffix is 'sum', then remove the suffix
            self.data_title_adjusted = self.data_title_base
        else:
            # Else keep the original title
            self.data_title_adjusted = self.data_title

    def __init__(self, *, param_model, io_model):
        # Initialize with an empty string (no elements selected)
        self.param_model = param_model
        self.io_model = io_model
        self.element_for_roi = ""
        self.enable_roi_computation = False

    @observe("element_for_roi")
    def _update_element(self, change):
        """
        Get element information as a string and parse it as a list.
        This element information means the ones for roi setup.
        """
        self.element_for_roi = self.element_for_roi.strip(" ")
        # Remove leading and trailing ','
        self.element_for_roi = self.element_for_roi.strip(",")
        # Remove leading and trailing '.'
        self.element_for_roi = self.element_for_roi.strip(".")
        try:
            if len(self.element_for_roi) == 0:
                logger.debug("No elements entered.")
                self.remove_all_roi()
                self.element_list_roi = []
                self.enable_roi_computation = False
                return
            elif "," in self.element_for_roi:
                element_list = [v.strip(" ") for v in self.element_for_roi.split(",")]
            else:
                element_list = [v for v in self.element_for_roi.split(" ")]

            # with self.suppress_notifications():
            #     self.element_list_roi = element_list
            logger.debug("Current elements for ROI sum are: {}".format(element_list))
            self.update_roi(element_list)
            self.element_list_roi = element_list
            self.enable_roi_computation = True
        except Exception as ex:
            logger.warning(f"Incorrect specification of element lines for ROI computation: {ex}")
            self.enable_roi_computation = False

    def select_elements_from_list(self, element_list):
        self.element_for_roi = ", ".join(element_list)

    def use_all_elements(self):
        self.element_for_roi = ", ".join(K_LINE + L_LINE)  # +M_LINE)

    def clear_selected_elements(self):
        self.element_for_roi = ""

    def remove_all_roi(self):
        self.roi_dict.clear()

    def update_roi(self, element_list, std_ratio=4):
        """
        Update elements without touching old ones.

        Parameters
        ----------
        element_list : list
            list of elements for roi
        std_ratio : float, optional
            Define the range of roi for given element.

        Notes
        -----
        The unit of energy is in ev in this function. The reason is
        SpinBox in Enaml can only read integer as input. To be updated.
        """

        eline_list = K_LINE + L_LINE + M_LINE

        for v in element_list:
            if v in self.roi_dict:
                continue

            if v not in eline_list:
                raise ValueError(f"Emission line {v} is unknown")

            if "_K" in v:
                temp = v.split("_")[0]
                e = Element(temp)
                val = int(e.emission_line["ka1"] * 1000)
            elif "_L" in v:
                temp = v.split("_")[0]
                e = Element(temp)
                val = int(e.emission_line["la1"] * 1000)
            elif "_M" in v:
                temp = v.split("_")[0]
                e = Element(temp)
                val = int(e.emission_line["ma1"] * 1000)

            delta_v = int(self.get_sigma(val / 1000) * 1000)

            roi = ROISettings(
                prefix=self.suffix_name_roi,
                line_val=val,
                left_val=val - delta_v * std_ratio,
                right_val=val + delta_v * std_ratio,
                default_left=val - delta_v * std_ratio,
                default_right=val + delta_v * std_ratio,
                step=1,
                show_plot=False,
            )

            self.roi_dict.update({v: roi})

        # remove old items not included in element_list
        for k in self.roi_dict.copy().keys():
            if k not in element_list:
                del self.roi_dict[k]

    def get_sigma(self, energy, epsilon=2.96):
        """
        Calculate the std at given energy.
        """
        temp_val = 2 * np.sqrt(2 * np.log(2))
        return np.sqrt(
            (self.param_model.param_new["fwhm_offset"]["value"] / temp_val) ** 2
            + energy * epsilon * self.param_model.param_new["fwhm_fanoprime"]["value"]
        )

    def get_roi_sum(self):
        """
        Save roi sum into a dict.

        Returns
        -------
        dict
            nested dict as output
        """
        roi_result = {}

        datav = self.io_model.data_sets[self.data_title].raw_data

        logger.info(f"Computing ROIs for dataset {self.data_title} ...")

        snip_param = {
            "e_offset": self.param_model.param_new["e_offset"]["value"],
            "e_linear": self.param_model.param_new["e_linear"]["value"],
            "e_quadratic": self.param_model.param_new["e_quadratic"]["value"],
            "b_width": self.param_model.param_new["non_fitting_values"]["background_width"],
        }

        n_bin_low, n_bin_high = get_energy_bin_range(
            num_energy_bins=datav.shape[2],
            low_e=self.param_model.param_new["non_fitting_values"]["energy_bound_low"]["value"],
            high_e=self.param_model.param_new["non_fitting_values"]["energy_bound_high"]["value"],
            e_offset=self.param_model.param_new["e_offset"]["value"],
            e_linear=self.param_model.param_new["e_linear"]["value"],
        )

        # Prepare the 'roi_dict' parameter for computations
        roi_dict = {
            _: (self.roi_dict[_].left_val / 1000.0, self.roi_dict[_].right_val / 1000.0)
            for _ in self.roi_dict.keys()
        }

        roi_dict_computed = compute_selected_rois(
            data=datav,
            data_sel_indices=(n_bin_low, n_bin_high),
            roi_dict=roi_dict,
            snip_param=snip_param,
            use_snip=self.subtract_background,
            chunk_pixels=5000,
            n_chunks_min=4,
            progress_bar=TerminalProgressBar("Computing ROIs: "),
            client=None,
        )

        # Save ROI data to HDF5 file
        self.saveROImap_to_hdf(roi_dict_computed)

        # Add scalers to the ROI dataset, so that they can be selected from Image Wizard.
        # We don't want to save scalers to the file, since they are already in the file.
        # So we add scalers after data is saved.
        scaler_key = f"{self.data_title_base}_scaler"
        if scaler_key in self.io_model.img_dict:
            roi_dict_computed.update(self.io_model.img_dict[scaler_key])

        roi_result[f"{self.data_title_adjusted}_roi"] = roi_dict_computed

        logger.info("ROI is computed.")
        return roi_result

    def saveROImap_to_hdf(self, data_dict_roi):

        # Generate the path to computed ROIs in the HDF5 file
        det_name = "detsum"  # Assume that ROIs are computed using the sum of channels

        # Search for channel name in the data title. Channels are named
        #   det1, det2, ... , i.e. 'det' followed by integer number.
        # The channel name is always located at the end of the ``data_title``.
        # If the channel name is found, then build the path using this name.
        srch = re.search("det\d+$", self.data_title)  # noqa: W605
        if srch:
            det_name = srch.group(0)
        inner_path = f"xrfmap/{det_name}"

        try:
            save_fitdata_to_hdf(
                self.hdf_path,
                data_dict_roi,
                datapath=inner_path,
                data_saveas="xrf_roi",
                dataname_saveas="xrf_roi_name",
            )
        except Exception as ex:
            logger.error(f"Failed to save ROI data to file '{self.hdf_path}'\n    Exception: {ex}")
        else:
            logger.info(f"ROI data was successfully saved to file '{self.hdf_name}'")
