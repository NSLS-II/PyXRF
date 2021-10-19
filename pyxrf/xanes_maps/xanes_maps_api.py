import os
import re
import sys
import numpy as np
import csv
from pystackreg import StackReg
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, TextBox
from matplotlib.patches import Rectangle, FancyArrow
import time as ttime
import tifffile
import jsonschema
import pandas as pd

from ..model.load_data_from_db import make_hdf
from ..model.command_tools import pyxrf_batch
from ..model.fileio import read_hdf_APS
from ..core.utils import grid_interpolate, normalize_data_by_scaler, convert_time_to_nexus_string
from ..core.xrf_utils import check_if_eline_is_activated, check_if_eline_supported
from ..core.fitting import fit_spectrum, rfactor_compute
from ..core.yaml_param_files import create_yaml_parameter_file, read_yaml_parameter_file
from ..core.map_processing import dask_client_create

import logging

logger = logging.getLogger(__name__)


def build_xanes_map(
    start_id=None,
    end_id=None,
    *,
    parameter_file_path=None,
    create_parameter_file=False,
    allow_exceptions=False,
    **kwargs,
):
    r"""
    The function builds XANES maps based on a set of XRF scans. The maps may be built based
    on data from the following sources:

    -- database (set the parameter ``sequence`` to ``load_and_process``). The data
    for the specified range of scan IDs ``start_id`` .. ``end_id`` is loaded using
    databroker, saved to .h5 files placed in subdirectory ``xrf_subdir`` of the working
    directory ``wd`` and processed using ``pyxrf_batch`` with the set of parameters from
    ``xrf_fitting_param_fln`` to generate XRF maps. XANES maps are computed from the stack of
    the XRF maps. The maps are interpolated to uniform grid of position coordinates
    (important if positions of data points are unevenly spaced), aligned along the spatial
    coordinates and fitted with the element references from file ``ref_file_name``. The
    resulting XANES maps and XRF map stack are saved to the hard drive in the directory ``wd``.
    (Currently only stacked TIFF files are supported). The results are also plotted in
    multiple Matplotlib figures.

    -- .h5 data files in subfolder ``xrf_subdir`` of the directory ``wd`` (set the parameter
    ``sequence`` to ``process``). This option should be selected if data is already loaded
    on local hard drive, since downloading a stack of images may significantly increase
    the total processing time. Select this option to process existing raw data files,
    or process data files with different set of parameters to generate XRF maps.

    -- processed .h5 files in subfolder ``xrf_subdir`` of the directory ``wd`` (set the parameter
    ``sequence`` to ``build_xanes_map``. This is the quickest processing option and should be
    used if properly generated XRF maps for the selected set of element emission lines is
    already present in the .h5 files.

    Example of function call for the batch of scans with IDs in the range 92276-92335:

    build_xanes_map(92276, 92335, xrf_fitting_param_fln="param_335", scaler_name="sclr1_ch4",
    sequence="load_and_process", ref_file_name="refs_Fe_P23.csv", fitting_method="nnls",
    emission_line="Fe_K", emission_line_alignment="P_K", incident_energy_shift_keV=-0.0013)

    By default, the function will not overwrite the existing files and skip downloading
    of the data. To overwrite the existing files, set the parameter ``file_overwrite_existing=True``.

    There are two ways of setting processing parameters:

    -- Method 1. The parameter values different from default may be sent to ``build_xanes_map`` as
    function arguments (as in the example above).

    -- Method 2. Use YAML file with processing parameters. The YAML file may be created by
    calling ``build_xanes_map`` with parameter ``create_parameter_file=True`` and supplying
    the name of the parameter file ``parameter_file_path`` (absolute or relative path).
    The file contains the description of each parameter and brief notes on editing YAML files.
    Parameter values are initially set to default values. Some values must be changed before
    processing can be run (e.g. ``emission_line`` must be set to some valid name). Once
    parameters in YAML file are set, the function may be called as
    ``build_xanes_map(parameter_file_path="path-to-file")``.
    It is not requiremed that each parameter is specified in the YAML file: the parameters
    that are not specified will be assigned the default value. If the YAML file contains
    parameters that are unsupported by the program, the list of such parameters and their
    values will be printed. Unsupported parameters are ignored by the program.
    Additional parameter values may be sent as function arguments: such parameters have
    precedence before the default parameters and the parameters from YAML file.
    For example, the function call

    build_xanes_map(parameter_file_path="path-to-file", incident_energy_shift_keV=0.001)

    will override the value of the incident energy shift specified in the YAML file.

    Options:

    Typical processing sequence includes operation of XRF map interpolation and stack alignment.
    But those operations may be disabled by setting parameters ``interpolation_enable`` and
    ``alignment enable`` to False. For example, if the position coordinates of XRF maps form
    uniform grid, then disabling interpolation will reduce processing time.

    There are two parameters that control stack alignment. The stack alignment for all emission
    lines is performed based on the emission line selected for XANES (parameter ``emission_line``).
    If maps for different emission lines are better suited for alignment, then the default behavior
    can be changed by setting the parameter ``emission_line_alignment``. The other parameter
    controls the order of the stack alignment. The default order (starting from the top of
    the stack or XRF map measured with highest energy) may be changed by setting the parameter
    ``alignment_starts_from="bottom"``. In this case the alignment will start from the map
    acquired with the lowest incident energy.

    Incident energy used for generation of XRF maps is automatically computed as the highest of
    the energies: incident energy from the respective .h data file or the lowest energy which
    activates the emission line of interest (line specified by the parameter ``emission_line``).
    The default behavior may be changed by setting True the parameter
    ``use_incident_energy_from_param_file`` (use fixed energy from parameter file set by
    ``xrf_fitting_param_fln``) or specifying lower bound for incident energy ``incident_energy_low_bound``
    (if the energy read from the .h5 data file is less than ``incident_energy_low_bound``, then
    use the lower bound, otherwise use the value of energy from the data file.

    If ``plot_results`` is set to False, then the results will not be plotted (no Matplotlib
    windows will be opened). The processing results will still be saved to files with selected
    format(s).

    Format of the reference file (see parameter ``ref_file_name``)
    --------------------------------------------------------------

    The element references for XANES fitting must be formatted as columns of CSV file.
    The first column of the file must contain incident energy values. The first line of the file
    must contain labels for each column. The labels must be comma separated. If labels contain
    separator ',', they must be enclosed in double quotes '"'. (Note, that spreadsheet programs
    tend to use other characters as opening and closing quotes. Those characters look similar
    to double quotes, but are not recognized by the CSV reading software. It is better to
    avoid using separator ',' in the labels.) Labels often represent chemical formulas.
    Since Matplotlib renders LaTeX formatted strings, the labels can use LaTeX formatting
    for better quality plotting.

    The example of the CSV file with LaTeX formatted labels:

    Energy,$Fe_3P$,$LiFePO_4$,$Fe_2O_3$
    7061.9933,0.0365,0.0235,0.0153
    7063.9925,0.0201,0.0121,0.00378
    7065.9994,0.0181,0.0111,0.00327
    7067.9994,0.0161,0.0101,0.00238
    7070.0038,0.0144,0.00949,0.00233
    ... etc. ...

    By default, the function is catch all exceptions and prints the error messages on the screen.
    This is the most user-friendly mode of operation, since routine error messages are presented
    in readable form. If the function is used as part of a script, the exceptions may be
    allowed by setting the parameter ``allow_exceptions=True``.

    Parameters
    ----------

    start_id : int
        scan ID of the first run of the sequence.
        Default: ``None``

    end_id : int
        scan ID of the last run of the sequence

        When processing data files from the local drive, both ``start_id`` and ``end_id``
        may be set to ``None`` (default). In this case all files in the ``xrf_subdir``
        subdirectory of the working directory ``wd`` are loaded and processed. The values
        of ``start_id`` and ``end_id`` will limit the range of processed scans.
        The values of ``start_id`` and ``end_id`` must be set to proper values in order
        to load data from the databroker.
        Default: ``None``

    file_overwrite_existing : boolean
        Indicates if the existing file should be deleted and replaced with the new file
        with the same name: ``True`` the file will always be replaced, ``False`` existing
        file will be skipped. This parameter is passed directly to ``make_hdf``, which
        is responsible for downloading data.

    xrf_fitting_param_fln : str
        the name of the JSON parameter file. The parameters are used for automated
        processing of data with ``pyxrf_batch``. The parameter file is typically produced
        by PyXRF. The parameter is not used for XANES analysis and may be skipped if
        if XRF maps are already generated (``sequence="build_xanes_maps"``). The file
        name may include absolute or relative path from current directory
        (not the directory ``wd``).
        Default: ``None``

    xrf_subtract_baseline : bool
        Subtract baseline before spectral decomposition while generating XRF maps.
        Typically baseline subtraction is enabled. Disabling it will reduce time
        used for generation of XRF maps. (This parameter is identical to 'use_snip'
        of `pyxrf_batch`.
        Default: True

    scaler_name : str
        the name of the scaler used for normalization. The name should be valid, i.e.
        present in each scan data. It may be set to None: in this case no normalization
        will be performed.
        Default: ``None``

    wd : str
        working directory: if ``wd`` is not specified then current directory will be
        used as the working directory for processing. The working directory is used
        to store raw and processed HDF5 files (in subdirectory specified by
        ``xrf_subdir``) and write processing results. Path to configuration files
        is specified separately.
        Default: ``None``

    xrf_subdir : str
        subdirectory inside the working directory in which raw data files will be loaded.
        If the "process" or "build_xanes_map" sequence is executed, then the program
        looks for raw or processed files in this subfolder. In majority of cases, the
        default value is sufficient.
        Default: ``"xrf_data"``

    results_dir_suffix : str
        suffix for the directory where the results are saved. The suffix is generated
        automatically if the parameter is not specified or ``None``. The directory will be
        created inside the directory ``wd`` with the name ``nanoXANES_Analysis_{suffix}``.
        Default: ``None``

    sequence : str
        the sequence of operations performed by the function:

        -- ``load_and_process`` - loading data and full processing,

        -- ``process`` - full processing of data, including xrf mapping
        and building the energy map,

        -- ``build_xanes_map`` - build the energy map using xrf mapping data,
        all data must be processed.

        Default: ``"build_xanes_map"``

    emission_line : str
        the name of the selected emission line ("Ca_K", "Fe_K", etc.). The emission line
        of interest. This is the REQUIRED parameter. Processing can not be performed unless
        valid emission line is selected. The emission line must be in the list used for
        XRF mapping.
        Default: ``None``

    emission_line_alignment : str
        the name of the stack of maps used as a reference for alignment of all other stacks.
        The name can represent a valid emission line (e.g. "Ca_K", "Fe_K", etc.) or arbitrary
        stack name (e.g. "total_cnt"). The map with the specified name must be present
        in each processed data file. If ``None``, then the name of the line specified as
        ``emission_line`` used for alignment.
        Default: ``None``

    incident_energy_shift_keV : float
        shift (in keV) applied to incident energy axis of the observed data before
        XANES fitting. The positive shift value shifts observed data in the direction of
        higher energies. The shift may be used to compensate for the difference in the
        adjustments of measurement setups used for acquisition of references and
        observed dataset.
        Default: ``0``

    alignment_starts_from : str
        The order of alignment of the image stack: "top" - start from the top of the stack
        (scan with highest energy) and proceed to the bottom of the stack (lowest energy),
        "bottom" - start from the bottom of the stack (lowest energy) and proceed to the top.
        Starting from the top typically produces better results, because scans at higher
        energy are more defined. If the same emission line is selected for XANES and
        for alignment reference, then alignment should always be performed starting from
        the top of the stack.
        Default: ``"top"``

    interpolation_enable : bool
        enable interpolation of XRF maps to uniform grid before alignment of maps.
        Default: ``True``

    alignment_enable : bool
        enable alignment of the stack of maps. In typical processing workflow the alignment
        should be enabled.
        Default: ``True``

    normalize_alignment_stack : bool
        enable normalization of the image stack used for alignment
        Default: ``True``

    subtract_pre_edge_baseline : bool
        indicates if pre-edge baseline is subtracted from XANES spectra before fitting.
        Currently the subtracted baseline is constant, computed as a median value of spectrum
        points in the pre-edge region.
        Default : ``True``

    ref_file_name : str
        file name with emission line references. If ``ref_file_name`` is not provided,
        then no XANES maps are generated. The rest of the processing is still performed
        as expected. The file name may include absolute or relative path from the current
        directory (not directory ``wd``).
        Default: ``None``

    fitting_method : str
        method used for fitting XANES spectra. The currently supported methods are
        'nnls' and 'admm'.
        Default: ``"nnls"``

    fitting_descent_rate : float
        optimization parameter: descent rate for the fitting algorithm.
        Used only for 'admm' algorithm (rate = 1/lambda), ignored for 'nnls' algorithm.
        Default: ``0.2``

    incident_energy_low_bound : float
        files in the set are processed using the value of incident energy equal to
        the greater of the values of ``incident_energy_low_bound`` or incident energy
        from file metadata. If None, then the lower energy bound is found automatically
        as the largest value of energy in the set which still activates the selected
        emission line (specified as the ``emission_line`` parameter). This parameter
        overrides the parameter ``use_incident_energy_from_param_file``.
        Default: ``None``

    use_incident_energy_from_param_file : bool
        indicates if incident energy from parameter file will be used to process all
        files: True - use incident energy from parameter files, False - use incident
        energy from data files. If ``incident_energy_low_bound`` is specified, then
        this parameter is ignored.
        Default: ``False``

    plot_results : bool
        indicates if results (image stack and XANES maps) are to be plotted. If set to
        False, the processing results are saved to file (if enabled) and the program
        exits without showing data plots.
        Default: ``True``

    plot_use_position_coordinates : bool
        results (image stack and XANES maps) are plotted vs. position coordinates if
        the parameter is set to True, otherwise images are plotted vs. pixel number
        Default: ``True``

    plot_position_axes_units : str
        units for position coordinates along X and Y axes. The units are used while
        plotting the results vs. position coordinates. The string specifying units
        may contain LaTeX expressions: for example ``"$\mu $m"`` will print units
        of ``micron`` as part of X and Y axes labels.
        Default: ``"$\mu $m"``

    output_file_formats : list(str)
        list of output file formats. Currently only "tiff" format is supported
        (XRF map stack and XANES maps are saved as stacked TIFF files).
        Default: ``["tiff"]``

    output_save_all : bool
        indicates if processing results for every emission line, which is selected for XRF
        fitting, are saved. If set to ``True``, the (aligned) stacks of XRF maps are saved
        for each element. If set to False, then only the stack for the emission line
        selected for XANES fitting is saved (argument ``emission_line``).
        Default value: False

    dask_client : dask.distributed.Client
        Dask client object. If None, then Dask client is created automatically.
        If a batch of files is processed, then creating Dask client and
        passing the reference to it to the processing functions will save
        execution time:

        .. code:: python

            from pyxrf.api import dask_client_create
            client = dask_client_create()  # Create Dask client
            # <-- code that runs computations -->
            client.close()  # Close Dask client

    parameter_file_path : str
        absolute or relative path to the YAML file, which contains the processing parameters.
        Relative path is specified from the current directory (not ``wd``).

    create_parameter_file : bool
        indicates whether a new parameter file needs to be created.

    allow_exceptions : bool
        True - raise exception in case of errors, False - print error message and exit

    Returns
    -------

    no value is returned

    Raises
    ------

    Raise exceptions if processing can not be completed and exceptions are allowed
    (``allow_exceptions=True``). The error message indicates the reason of failure.
    If exceptions are disabled, then error message is printed.
    """

    # First process the case of creating new parameter file (with default values)
    #   This step is using 'create_parameter_file' and 'parameter_file_path' arguments
    #   and ignores the rest.
    if create_parameter_file is True:
        # It is allowed to specify path to the new parameter file as the first argument.
        # This will work only if 'create_parameter_file' is set True.
        if not parameter_file_path and isinstance(start_id, str):
            parameter_file_path = start_id
        if parameter_file_path:
            try:
                create_yaml_parameter_file(
                    file_path=parameter_file_path,
                    function_docstring=_build_xanes_map_api.__doc__,
                    param_value_dict=_build_xanes_map_param_default,
                )
            except Exception as ex:
                logger.error(f"Error occurred while creating file '{parameter_file_path}': {ex}")
                if allow_exceptions:
                    raise
        else:
            msg = "New parameter file can't be created: parameter file path is not specified"
            if allow_exceptions:
                raise RuntimeError(msg)
            else:
                logger.error(msg)
        return

    # Move 'start_id' and 'end_id' to kwargs
    kwargs["start_id"] = start_id
    kwargs["end_id"] = end_id

    # Assemble the dictionary of arguments for the processing function
    arguments = _build_xanes_map_param_default
    # Modify default values to values from yaml parameter file (if file name is given)
    if parameter_file_path:
        param_file_args = read_yaml_parameter_file(file_path=parameter_file_path)
        # The set of parameters listed in 'param_file_args' may not match the set of supported parameters.
        # Unsupported parameters are ignored. The list of unsupported parameter names is printed
        # (those could be misspelled valid parameter names, so the information may be valuable)
        param_file_args_unsupported = {
            key: value for key, value in param_file_args.items() if key not in arguments
        }
        msg = [f"{key}: {value}" for key, value in param_file_args_unsupported.items()]
        if msg:
            msg = (
                f"Parameter file '{parameter_file_path}' contains unsupported parameters:\n"
                + "    "
                + "\n    ".join(msg)
            )
            logger.warning(msg)
        # The supported parameters are replacing the default parameters.
        args_modified = {key: value for key, value in param_file_args.items() if key in arguments}
        arguments.update(args_modified)
    # Verify if all arguments ('kwargs') of the function are supported. If the function is
    #   called with invalid arguments, then return from the function or raise the exception.
    #   The 'unsupported_args' list will keep the order of the arguments the same as in 'kwargs'.
    unsupported_args = {key: value for key, value in kwargs.items() if key not in arguments}
    if unsupported_args:  # Error occurred: at least one argument is not supported
        msg = [f"{key}: {value}" for key, value in unsupported_args.items()]
        msg = "\n    ".join(msg)
        msg = "The function is called with invalid arguments:\n    " + msg
        if allow_exceptions:
            raise RuntimeError(msg)
        else:
            logger.error(msg)
            return
    else:
        arguments.update(kwargs)

    # Now call the processing function
    try:
        # Validate the argument dictionary (exception will be raised if there is not match with the schema)
        jsonschema.validate(instance=arguments, schema=_build_xanes_map_param_schema)
        _build_xanes_map_api(**arguments)
    except BaseException as ex:
        if allow_exceptions:
            raise
        else:
            msg = f"Processing is incomplete! Exception was raised during execution:\n   {ex}"
            logger.error(msg)
    else:
        logger.info("Processing was completed successfully.")


# Default input parameters for '_build_xanes_map_api' and 'build_xanes_map' functions
_build_xanes_map_param_default = {
    "start_id": None,
    "end_id": None,
    "file_overwrite_existing": False,
    "xrf_fitting_param_fln": None,
    "xrf_subtract_baseline": True,
    "scaler_name": None,
    "wd": None,
    "xrf_subdir": "xrf_data",
    "results_dir_suffix": None,
    "sequence": "build_xanes_map",
    "emission_line": None,
    "emission_line_alignment": None,
    "incident_energy_shift_keV": 0,
    "alignment_starts_from": "top",
    "interpolation_enable": True,
    "alignment_enable": True,
    "normalize_alignment_stack": True,
    "subtract_pre_edge_baseline": True,
    "ref_file_name": None,
    "fitting_method": "nnls",
    "fitting_descent_rate": 0.2,
    "incident_energy_low_bound": None,
    "use_incident_energy_from_param_file": False,
    "plot_results": True,
    "plot_use_position_coordinates": True,
    "plot_position_axes_units": r"$\mu $m",
    "output_file_formats": ["tiff"],
    "output_save_all": False,
    "dask_client": None,
}

# Default input parameters for '_build_xanes_map_api' and 'build_xanes_map' functions
_build_xanes_map_param_schema = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "start_id",
        "end_id",
        "file_overwrite_existing",
        "xrf_fitting_param_fln",
        "xrf_subtract_baseline",
        "scaler_name",
        "wd",
        "xrf_subdir",
        "results_dir_suffix",
        "sequence",
        "emission_line",
        "emission_line_alignment",
        "incident_energy_shift_keV",
        "alignment_starts_from",
        "interpolation_enable",
        "alignment_enable",
        "normalize_alignment_stack",
        "subtract_pre_edge_baseline",
        "ref_file_name",
        "fitting_method",
        "fitting_descent_rate",
        "incident_energy_low_bound",
        "use_incident_energy_from_param_file",
        "plot_results",
        "plot_use_position_coordinates",
        "plot_position_axes_units",
        "output_file_formats",
        "output_save_all",
    ],
    "properties": {
        "start_id": {"type": ["integer", "null"], "exclusiveMinimum": 0},
        "end_id": {"type": ["integer", "null"], "exclusiveMinimum": 0},
        "file_overwrite_existing": {"type": "boolean"},
        "xrf_fitting_param_fln": {"type": ["string", "null"]},
        "xrf_subtract_baseline": {"type": "boolean"},
        "scaler_name": {"type": ["string", "null"]},
        "wd": {"type": ["string", "null"]},
        "xrf_subdir": {"type": ["string", "null"]},
        "results_dir_suffix": {"type": ["string", "null"]},
        "sequence": {"type": "string"},
        "emission_line": {"type": ["string", "null"]},
        "emission_line_alignment": {"type": ["string", "null"]},
        "incident_energy_shift_keV": {"type": "number"},
        "alignment_starts_from": {"type": "string", "enum": ["top", "bottom"]},
        "interpolation_enable": {"type": "boolean"},
        "alignment_enable": {"type": "boolean"},
        "normalize_alignment_stack": {"type": "boolean"},
        "subtract_pre_edge_baseline": {"type": "boolean"},
        "ref_file_name": {"type": ["string", "null"]},
        "fitting_method": {"type": "string", "enum": ["nnls", "admm"]},
        "fitting_descent_rate": {"type": "number", "exclusiveMinimum": 0.0},
        "incident_energy_low_bound": {"type": ["number", "null"], "exclusiveMinimum": 0.0},
        "use_incident_energy_from_param_file": {"type": "boolean"},
        "plot_results": {"type": "boolean"},
        "plot_use_position_coordinates": {"type": "boolean"},
        "plot_position_axes_units": {"type": "string"},
        "output_file_formats": {
            "type": "array",
            "uniqueItems": True,
            "items": {"type": "string", "enum": ["tiff"]},
        },
        "output_save_all": {"type": "boolean"},
        "dask_client": {},
    },
}


def _build_xanes_map_api(
    *,
    start_id=None,
    end_id=None,
    file_overwrite_existing=False,
    xrf_fitting_param_fln=None,
    xrf_subtract_baseline=True,
    scaler_name=None,
    wd=None,
    xrf_subdir="xrf_data",
    results_dir_suffix=None,
    sequence="build_xanes_map",
    emission_line=None,  # This is the REQUIRED parameter
    emission_line_alignment=None,
    incident_energy_shift_keV=0,
    alignment_starts_from="top",
    interpolation_enable=True,
    alignment_enable=True,
    normalize_alignment_stack=True,
    subtract_pre_edge_baseline=True,
    ref_file_name=None,
    fitting_method="nnls",
    fitting_descent_rate=0.2,
    incident_energy_low_bound=None,
    use_incident_energy_from_param_file=False,
    plot_results=True,
    plot_use_position_coordinates=True,
    plot_position_axes_units=r"$\mu $m",
    output_file_formats=["tiff"],
    output_save_all=False,
    dask_client=None,
):
    r"""
    The function builds XANES map from a set of XRF maps. It also may perform loading of data
    from databroker and processing of raw data to obtain XRF maps. For detailed descriptions,
    see docstring for the wrapper function ``build_xanes_map``.

    Note, that the list and the description of parameters in this docstring is used for
    generation of the YAML parameter file, so it has to be maintained properly. The set of
    parameters should match the list of keys in ``_build_xanes_map_param_default`` and
    the parameter values should satisfy the schema ``_build_xanes_map_param_schema``
    Also, the list of parameters in the docstring for ``build_xanes_map`` is displayed to the user and
    used for generation of documentation. So both docstrings should match and be maintained to
    contain complete information.

    Parameters
    ----------

    start_id : int
        scan ID of the first run of the sequence.
        Default: ``None``

    end_id : int
        scan ID of the last run of the sequence

        When processing data files from the local drive, both ``start_id`` and ``end_id``
        may be set to ``None`` (default). In this case all files in the ``xrf_subdir``
        subdirectory of the working directory ``wd`` are loaded and processed. The values
        of ``start_id`` and ``end_id`` will limit the range of processed scans.
        The values of ``start_id`` and ``end_id`` must be set to proper values in order
        to load data from the databroker.
        Default: ``None``

    file_overwrite_existing : boolean
        Indicates if the existing file should be deleted and replaced with the new file
        with the same name: ``True`` the file will always be replaced, ``False`` existing
        file will be skipped. This parameter is passed directly to ``make_hdf``, which
        is responsible for downloading data.

    xrf_fitting_param_fln : str
        the name of the JSON parameter file. The parameters are used for automated
        processing of data with ``pyxrf_batch``. The parameter file is typically produced
        by PyXRF. The parameter is not used for XANES analysis and may be skipped if
        if XRF maps are already generated (``sequence="build_xanes_maps"``). The file
        name may include absolute or relative path from current directory
        (not the directory ``wd``).
        Default: ``None``

    xrf_subtract_baseline : bool
        Subtract baseline before spectral decomposition while generating XRF maps.
        Typically baseline subtraction is enabled. Disabling it will reduce time
        used for generation of XRF maps. (This parameter is identical to 'use_snip'
        of `pyxrf_batch`.
        Default: True

    scaler_name : str
        the name of the scaler used for normalization. The name should be valid, i.e.
        present in each scan data. It may be set to None: in this case no normalization
        will be performed.
        Default: ``None``

    wd : str
        working directory: if ``wd`` is not specified then current directory will be
        used as the working directory for processing. The working directory is used
        to store raw and processed HDF5 files (in subdirectory specified by
        ``xrf_subdir``) and write processing results. Path to configuration files
        is specified separately.
        Default: ``None``

    xrf_subdir : str
        subdirectory inside the working directory in which raw data files will be loaded.
        If the "process" or "build_xanes_map" sequence is executed, then the program
        looks for raw or processed files in this subfolder. In majority of cases, the
        default value is sufficient.
        Default: ``"xrf_data"``

    results_dir_suffix : str
        suffix for the directory where the results are saved. The suffix is generated
        automatically if the parameter is not specified or ``None``. The directory will be
        created inside the directory ``wd`` with the name ``nanoXANES_Analysis_{suffix}``.
        Default: ``None``

    sequence : str
        the sequence of operations performed by the function:

        -- ``load_and_process`` - loading data and full processing,

        -- ``process`` - full processing of data, including xrf mapping
        and building the energy map,

        -- ``build_xanes_map`` - build the energy map using xrf mapping data,
        all data must be processed.

        Default: ``"build_xanes_map"``

    emission_line : str
        the name of the selected emission line ("Ca_K", "Fe_K", etc.). The emission line
        of interest. This is the REQUIRED parameter. Processing can not be performed unless
        valid emission line is selected. The emission line must be in the list used for
        XRF mapping.
        Default: ``None``

    emission_line_alignment : str
        the name of the stack of maps used as a reference for alignment of all other stacks.
        The name can represent a valid emission line (e.g. "Ca_K", "Fe_K", etc.) or arbitrary
        stack name (e.g. "total_cnt"). The map with the specified name must be present
        in each processed data file. If ``None``, then the name of the line specified as
        ``emission_line`` used for alignment.
        Default: ``None``

    incident_energy_shift_keV : float
        shift (in keV) applied to incident energy axis of the observed data before
        XANES fitting. The positive shift value shifts observed data in the direction of
        higher energies. The shift may be used to compensate for the difference in the
        adjustments of measurement setups used for acquisition of references and
        observed dataset.
        Default: ``0``

    alignment_starts_from : str
        The order of alignment of the image stack: "top" - start from the top of the stack
        (scan with highest energy) and proceed to the bottom of the stack (lowest energy),
        "bottom" - start from the bottom of the stack (lowest energy) and proceed to the top.
        Starting from the top typically produces better results, because scans at higher
        energy are more defined. If the same emission line is selected for XANES and
        for alignment reference, then alignment should always be performed starting from
        the top of the stack.
        Default: ``"top"``

    interpolation_enable : bool
        enable interpolation of XRF maps to uniform grid before alignment of maps.
        Default: ``True``

    alignment_enable : bool
        enable alignment of the stack of maps. In typical processing workflow the alignment
        should be enabled.
        Default: ``True``

    normalize_alignment_stack : bool
        enable normalization of the image stack used for alignment
        Default: ``True``

    subtract_pre_edge_baseline : bool
        indicates if pre-edge baseline is subtracted from XANES spectra before fitting.
        Currently the subtracted baseline is constant, computed as a median value of spectrum
        points in the pre-edge region.
        Default : ``True``

    ref_file_name : str
        file name with emission line references. If ``ref_file_name`` is not provided,
        then no XANES maps are generated. The rest of the processing is still performed
        as expected. The file name may include absolute or relative path from the current
        directory (not directory ``wd``).
        Default: ``None``

    fitting_method : str
        method used for fitting XANES spectra. The currently supported methods are
        'nnls' and 'admm'.
        Default: ``"nnls"``

    fitting_descent_rate : float
        optimization parameter: descent rate for the fitting algorithm.
        Used only for 'admm' algorithm (rate = 1/lambda), ignored for 'nnls' algorithm.
        Default: ``0.2``

    incident_energy_low_bound : float
        files in the set are processed using the value of incident energy equal to
        the greater of the values of ``incident_energy_low_bound`` or incident energy
        from file metadata. If None, then the lower energy bound is found automatically
        as the largest value of energy in the set which still activates the selected
        emission line (specified as the ``emission_line`` parameter). This parameter
        overrides the parameter ``use_incident_energy_from_param_file``.
        Default: ``None``

    use_incident_energy_from_param_file : bool
        indicates if incident energy from parameter file will be used to process all
        files: True - use incident energy from parameter files, False - use incident
        energy from data files. If ``incident_energy_low_bound`` is specified, then
        this parameter is ignored.
        Default: ``False``

    plot_results : bool
        indicates if results (image stack and XANES maps) are to be plotted. If set to
        False, the processing results are saved to file (if enabled) and the program
        exits without showing data plots.
        Default: ``True``

    plot_use_position_coordinates : bool
        results (image stack and XANES maps) are plotted vs. position coordinates if
        the parameter is set to True, otherwise images are plotted vs. pixel number
        Default: ``True``

    plot_position_axes_units : str
        units for position coordinates along X and Y axes. The units are used while
        plotting the results vs. position coordinates. The string specifying units
        may contain LaTeX expressions: for example ``"$\mu $m"`` will print units
        of ``micron`` as part of X and Y axes labels.
        Default: ``"$\mu $m"``

    output_file_formats : list(str)
        list of output file formats. Currently only "tiff" format is supported
        (XRF map stack and XANES maps are saved as stacked TIFF files).
        Default: ``["tiff"]``

    output_save_all : bool
        indicates if processing results for every emission line, which is selected for XRF
        fitting, are saved. If set to ``True``, the (aligned) stacks of XRF maps are saved
        for each element. If set to False, then only the stack for the emission line
        selected for XANES fitting is saved (argument ``emission_line``).
        Default value: False

    dask_client : dask.distributed.Client
        Dask client object. If None, then Dask client is created automatically.
        If a batch of files is processed, then creating Dask client and
        passing the reference to it to the processing functions will save
        execution time: `client = Client(processes=True, silence_logs=logging.ERROR)`

    Returns
    -------

    no value is returned

    Raises
    ------

    Raise exceptions if processing can not be completed. The error message indicates the reason
    of failure.
    """

    if wd is None:
        wd = "."
    else:
        wd = os.path.expanduser(wd)
    wd = os.path.abspath(wd)

    if ref_file_name:
        ref_file_name = os.path.expanduser(ref_file_name)
        ref_file_name = os.path.abspath(ref_file_name)

    if not xrf_subdir:
        raise ValueError(
            "The parameter 'xrf_subdir' is None or contains an empty string ('_build_xanes_map_api')."
        )

    if not scaler_name:
        logger.warning(
            "Scaler was not specified. The processing will still be performed,"
            "but the DATA WILL NOT BE NORMALIZED!"
        )

    if not emission_line:
        raise ValueError(
            "Emission line is not specified (parameter 'emission_line'). XANES maps can not be built."
        )

    # Convert all tags specifying output format to lower case
    for n, fmt in enumerate(output_file_formats):
        output_file_formats[n] = fmt.lower()
    # Now check every value agains the list of supported formats.
    supported_formats = ("tiff",)
    for fmt in output_file_formats:
        assert (
            fmt in supported_formats
        ), f"Output format '{fmt}' is not supported. Check values of the parameter 'output_file_formats'"

    alignment_starts_from_values = ["top", "bottom"]
    alignment_starts_from = alignment_starts_from.lower()
    if alignment_starts_from not in alignment_starts_from_values:
        raise ValueError(
            "The parameter 'alignment_starts_from' has illegal value "
            f"'{alignment_starts_from}' ('_build_xanes_map_api')."
        )

    # Selected emission lines for XANES and image stack alignment
    eline_selected = emission_line
    if emission_line_alignment:
        eline_alignment = emission_line_alignment
    else:
        eline_alignment = eline_selected

    # Check emission line names. They must be in the list of supported emission lines.
    #   The check is case-sensitive.
    if not check_if_eline_supported(eline_selected):
        raise ValueError(
            f"The emission line '{eline_selected}' does not exist or is not supported. "
            f"Check the value of the parameter 'eline_selected' ('_build_xanes_map_api')."
        )
    # The map used for stack alignment doesn't have to represent fluorescence due to an emission line
    #   It can contain some other quantity (e.g. 'total_cnt'). So just print a warning.
    if not check_if_eline_supported(eline_alignment):
        logger.warning(
            f"The map '{eline_alignment}' selected for alignment does not represent a supported emission line"
        )

    # Check fitting method
    fitting_method = fitting_method.lower()
    supported_fitting_methods = ("nnls", "admm")
    if fitting_method not in supported_fitting_methods:
        raise ValueError(
            f"The fitting method '{fitting_method}' is not supported. "
            f"Supported methods: {supported_fitting_methods}"
        )

    # Depending on the selected sequence, determine which steps must be executed
    seq_load_data = True
    seq_process_xrf_data = True
    seq_build_xrf_map_stack = True
    seq_generate_xanes_map = True
    if sequence == "load_and_process":
        pass
    elif sequence == "process":
        seq_load_data = False
    elif sequence == "build_xanes_map":
        seq_load_data = False
        seq_process_xrf_data = False
    else:
        raise ValueError(
            f"Unknown sequence name '{sequence}' is passed as a parameter "
            "to the function '_build_xanes_map_api'."
        )

    if seq_process_xrf_data:
        if not xrf_fitting_param_fln:
            raise ValueError(
                "Parameter file name is not specified and XRF maps can not be generated: "
                "set the value the parameter 'xrf_fitting_param_fln' of '_build_xanes_map_api'."
            )
        xrf_fitting_param_fln = os.path.expanduser(xrf_fitting_param_fln)
        xrf_fitting_param_fln = os.path.abspath(xrf_fitting_param_fln)

    # No XANES maps will be generated if references are not provided
    #                 (this is one of the built-in options, not an error)
    if not ref_file_name:
        seq_generate_xanes_map = False
        ref_energy = None
        ref_data = None
        ref_labels = None
    else:
        # Load reference file. (If there is a problem reading the reference file, it is better
        #   to learn it before the processing starts.)
        ref_energy, ref_data, ref_labels = read_ref_data(ref_file_name)

    # XRF data will be placed in the subdirectory 'xrf_data' of the directory 'wd'
    wd_xrf = os.path.join(wd, xrf_subdir)

    if seq_load_data:
        _load_data_from_databroker(
            start_id=start_id, end_id=end_id, wd_xrf=wd_xrf, file_overwrite_existing=file_overwrite_existing
        )
        logger.info("Loading data from databroker: success.")
    else:
        logger.info("Loading of data from databroker: skipped.")

    if seq_process_xrf_data:
        _process_xrf_data(
            start_id=start_id,
            end_id=end_id,
            wd_xrf=wd_xrf,
            xrf_fitting_param_fln=xrf_fitting_param_fln,
            xrf_subtract_baseline=xrf_subtract_baseline,
            eline_selected=eline_selected,
            incident_energy_low_bound=incident_energy_low_bound,
            use_incident_energy_from_param_file=use_incident_energy_from_param_file,
            dask_client=dask_client,
        )
        logger.info("Processing data files (computing XRF maps): success.")
    else:
        logger.info("Processing data files (computing XRF maps): skipped.")

    if seq_build_xrf_map_stack:
        processing_results = _compute_xanes_maps(
            start_id=start_id,
            end_id=end_id,
            wd_xrf=wd_xrf,
            eline_selected=eline_selected,
            eline_alignment=eline_alignment,
            scaler_name=scaler_name,
            interpolation_enable=interpolation_enable,
            alignment_enable=alignment_enable,
            normalize_alignment_stack=normalize_alignment_stack,
            seq_generate_xanes_map=seq_generate_xanes_map,
            fitting_method=fitting_method,
            fitting_descent_rate=fitting_descent_rate,
            incident_energy_shift_keV=incident_energy_shift_keV,
            alignment_starts_from=alignment_starts_from,
            ref_energy=ref_energy,
            ref_data=ref_data,
            subtract_pre_edge_baseline=subtract_pre_edge_baseline,
        )

        res_dir = _generate_output_dir_name(wd=wd, dir_suffix=results_dir_suffix)
        os.makedirs(res_dir, exist_ok=True)

        _save_xanes_processing_results(
            wd=res_dir,
            eline_selected=eline_selected,
            ref_labels=ref_labels,
            output_file_formats=output_file_formats,
            processing_results=processing_results,
            output_save_all=output_save_all,
        )

        if plot_results:
            _plot_processing_results(
                ref_energy=ref_energy,
                ref_data=ref_data,
                ref_labels=ref_labels,
                plot_position_axes_units=plot_position_axes_units,
                plot_use_position_coordinates=plot_use_position_coordinates,
                eline_selected=eline_selected,
                processing_results=processing_results,
                fitting_method=fitting_method,
                fitting_descent_rate=fitting_descent_rate,
                incident_energy_shift_keV=incident_energy_shift_keV,
                subtract_pre_edge_baseline=subtract_pre_edge_baseline,
                wd=wd,
            )
        else:
            logger.info("Plotting results: skipped.")

    logger.info("Processing is complete.")


def _load_data_from_databroker(*, start_id, end_id, wd_xrf, file_overwrite_existing):
    r"""
    Implements the first step of processing sequence: loading the batch of scan data
    from databroker.

    Parameters
    ----------

    start_id : int
        first scan ID of the batch of scans

    end_id : int
        last scan ID of the batch of scans

    wd_xrf : str
        full (absolute) name of the directory, where loaded .h5 files are placed

    file_overwrite_existing : bool, keyword parameter
        Indicates if the existing file should be deleted and replaced with the new file
        with the same name: ``True`` the file will always be replaced, ``False`` existing
        file will be skipped.
    """

    # Try to create the directory (does nothing if the directory exists)
    os.makedirs(wd_xrf, exist_ok=True)
    files_h5 = [fl.path for fl in os.scandir(path=wd_xrf) if fl.name.lower().endswith(".h5")]
    if files_h5:
        logger.warning(
            f"The temporary directory '{wd_xrf}' is not empty. Deleting {len(files_h5)} files (.h5) ..."
        )
        for fln in files_h5:
            logger.info(f"Removing raw xrf data file: '{fln}'.")
            os.remove(path=fln)
    make_hdf(
        start_id,
        end_id,
        wd=wd_xrf,
        completed_scans_only=True,
        file_overwrite_existing=file_overwrite_existing,
        create_each_det=False,
        save_scaler=True,
    )


def _process_xrf_data(
    *,
    start_id,
    end_id,
    wd_xrf,
    xrf_fitting_param_fln,
    xrf_subtract_baseline,
    eline_selected,
    incident_energy_low_bound,
    use_incident_energy_from_param_file,
    dask_client,
):
    r"""
    Implements the second step of the processing sequence: processing of XRF scans
    and generation of XRF maps

    Parameters
    ----------

    start_id : int
        first scan ID of the batch of scans

    end_id : int
        last scan ID of the batch of scans

    wd_xrf : str
        full (absolute) name of the directory, where loaded .h5 files are placed

    xrf_fitting_param_fln : str
        the name of the JSON parameter file. The parameters are used for automated
        processing of data with ``pyxrf_batch``. The parameter file is typically produced
        by PyXRF.

    dask_client : dask.distributed.Client
        Dask client object. If None, then Dask client is created automatically.
        If a batch of files is processed, then creating Dask client and
        passing the reference to it to the processing functions will save
        execution time: `client = Client(processes=True, silence_logs=logging.ERROR)`

    eline_selected : str
        the name of the selected emission line ("Ca_K", "Fe_K", etc.). The emission line
        of interest.

    incident_energy_low_bound : float
        files in the set are processed using the value of incident energy equal to
        the greater of the values of ``incident_energy_low_bound`` or incident energy
        from file metadata. If None, then the lower energy bound is found automatically
        as the largest value of energy in the set which still activates the selected
        emission line (specified as the ``emission_line`` parameter). This parameter
        overrides the parameter ``use_incident_energy_from_param_file``.

    use_incident_energy_from_param_file : bool
        indicates if incident energy from parameter file will be used to process all
        files: True - use incident energy from parameter files, False - use incident
        energy from data files. If ``incident_energy_low_bound`` is specified, then
        this parameter is ignored.

    dask_client : dask.distributed.Client
        Dask client object. If None, then Dask client is created automatically.
        If a batch of files is processed, then creating Dask client and
        passing the reference to it to the processing functions will save
        execution time: `client = Client(processes=True, silence_logs=logging.ERROR)`
    """
    # Make sure that the directory with xrf data exists
    if not os.path.isdir(wd_xrf):
        # Unfortunately there is no way to continue if there is no directory with data
        raise IOError(f"XRF data directory '{wd_xrf}' does not exist.")

    # Load scan metadata (only ids, energies and file names are extracted)
    scan_ids, scan_energies, _, files_h5 = _load_dataset_from_hdf5(
        start_id=start_id, end_id=end_id, wd_xrf=wd_xrf, load_fit_results=False
    )

    # Sort the lists based on incident beam energy. (The loaded data is sorted in the
    #   alphabetical order of file names, which may not match the ascending order or
    #   incident energy values.
    scan_energies, sorted_indexes = list(zip(*sorted(zip(scan_energies, range(len(scan_energies))))))
    scan_energies = list(scan_energies)
    files_h5 = [files_h5[n] for n in sorted_indexes]
    scan_ids = [scan_ids[n] for n in sorted_indexes]

    scan_energies_adjusted = scan_energies.copy()

    ignore_metadata = False
    if incident_energy_low_bound is not None:
        for n, v in enumerate(scan_energies_adjusted):
            if v < incident_energy_low_bound:
                scan_energies_adjusted[n] = incident_energy_low_bound
    elif use_incident_energy_from_param_file:
        # If 'pyxrf_batch' is called with 'incident_energy' set to None, and
        #   'ignore_datafile_metadata' is True, then
        #   the value of the incident energy from the parameter file is used
        scan_energies_adjusted = [None] * len(scan_energies)
        ignore_metadata = True
    else:
        scan_energies_adjusted = adjust_incident_beam_energies(scan_energies, eline_selected)

    # Create Dask client for processing the batch of files unless one is provided
    if dask_client is None:
        logger.info("Creating local Dask client for processing the batch of files ...")
        dask_client = dask_client_create()
        client_is_local = True
    else:
        client_is_local = False

    # Process data files from the list. Use adjusted energy value.
    for fln, energy in zip(files_h5, scan_energies_adjusted):
        # Process .h5 files in the directory 'wd_xrf'. Processing results are saved
        #   as additional datasets in the original .h5 files.
        pyxrf_batch(
            data_files=fln,  # Process only one data file
            param_file_name=xrf_fitting_param_fln,
            ignore_datafile_metadata=ignore_metadata,
            incident_energy=energy,  # This value overrides incident energy from other sources
            use_snip=xrf_subtract_baseline,
            wd=wd_xrf,
            save_tiff=False,
            dask_client=dask_client,
        )

    # Close the client if it was created locally
    if client_is_local:
        logger.info("Closing Dask client ...")
        dask_client.close()


def _compute_xanes_maps(
    *,
    start_id,
    end_id,
    wd_xrf,
    eline_selected,
    eline_alignment,
    alignment_starts_from,
    scaler_name,
    ref_energy,
    ref_data,
    fitting_method,
    fitting_descent_rate,
    incident_energy_shift_keV,
    interpolation_enable,
    alignment_enable,
    normalize_alignment_stack,
    subtract_pre_edge_baseline,
    seq_generate_xanes_map,
):
    r"""
    Implements the third step of the processing sequence: computation of XANES maps based
    on the set of XRF maps from scan in the range ``start_id`` .. ``end_id``.

    Parameters
    ----------

    start_id : int
        first scan ID of the batch of scans

    end_id : int
        last scan ID of the batch of scans

    wd_xrf : str
        full (absolute) name of the directory, where loaded .h5 files are placed

    eline_selected : str
        the name of the selected emission line ("Ca_K", "Fe_K", etc.). The emission line
        of interest.

    eline_alignment : str
        the name of the emission line ("Ca_K", "Fe_K", etc.) used for alignment of image stack.
        The emission line may be the same as ``eline_selected``.

    alignment_starts_from : str
        The order of alignment of the image stack: "top" - start from the top of the stack
        (scan with highest energy) and proceed to the bottom of the stack (lowest energy),
        "bottom" - start from the bottom of the stack (lowest energy) and proceed to the top.
        This is user defined parameter, which is passed as an argument to the program.

    scaler_name : str
        the name of the scaler used for normalization. The name should be valid, i.e.
        present in each scan data. It may be set to None: in this case no normalization
        will be performed.

    ref_energy : ndarray(float), 1D
        values of incident energy for XANES reference data specified in ``ref_data``. If
        ``ref_data`` has shape (N, M), then ``ref_energy`` must have N elements.

    ref_data : ndarray(float), 2D
        reference data for the element states, used for XANES fitting. The array of shape (N, M)
        contains reference data for M element states specified at N energy points.

    fitting_method : str
        method used for fitting XANES spectra. The currently supported methods are
        'nnls' and 'admm'.

    fitting_descent_rate : float
        optimization parameter: descent rate for the fitting algorithm.
        Used only for 'admm' algorithm (rate = 1/lambda), ignored for 'nnls' algorithm.

    incident_energy_shift_keV : float
        shift (in keV) applied to incident energy axis of the observed data before
        XANES fitting. The positive shift value shifts observed data in the direction of
        higher energies. The shift may be used to compensate for the difference in the
        adjustments of measurement setups used for acquisition of references and
        observed dataset.

    interpolation_enable : bool
        enable interpolation of XRF maps to uniform grid before alignment of maps.

    alignment_enable : bool
        enable alignment of the stack of maps. In typical processing workflow the alignment
        should be enabled.

    normalize_alignment_stack : bool
        enable normalization of the image stack used for alignment
        Default: ``True``

    subtract_pre_edge_baseline : bool
        indicates if pre-edge baseline is subtracted from XANES spectra before fitting.
        Currently the subtracted baseline is constant, computed as a median value of spectrum
        points in the pre-edge region.

    seq_generate_xanes_map : bool
        indicates if XANES maps should be generated based on the aligned stack. If set to False,
        then the step of generation XANES maps is skipped.

    Returns
    -------

    Dictionary with the results of processing. For structure of the dictionary entries look
    at the end of the function code.
    """

    logger.info("Building energy map ...")

    scan_ids, scan_energies, scan_img_dict, files_h5 = _load_dataset_from_hdf5(
        start_id=start_id, end_id=end_id, wd_xrf=wd_xrf
    )

    # The following function checks dataset for consistency. If additional checks
    #   needs to be performed, they should be added to the implementation of this function.
    _check_dataset_consistency(
        scan_ids=scan_ids,
        scan_img_dict=scan_img_dict,
        files_h5=files_h5,
        scaler_name=scaler_name,
        eline_selected=eline_selected,
        eline_alignment=eline_alignment,
    )

    logger.info("Checking dataset for consistency: success.")

    # Sort the lists based on energy. Prior to this point the data was arranged in the
    #   alphabetical order of files.
    scan_energies, sorted_indexes = list(zip(*sorted(zip(scan_energies, range(len(scan_energies))))))
    files_h5 = [files_h5[n] for n in sorted_indexes]
    scan_ids = [scan_ids[n] for n in sorted_indexes]
    scan_img_dict = [scan_img_dict[n] for n in sorted_indexes]

    logger.info("Sorting dataset: success.")

    # Apply shift to scan energies
    scan_energies_shifted = [_ + incident_energy_shift_keV for _ in scan_energies]

    # Create the lists of positional data for all scans
    positions_x_all = np.asarray([element["positions"]["x_pos"] for element in scan_img_dict])
    positions_y_all = np.asarray([element["positions"]["y_pos"] for element in scan_img_dict])

    # Find uniform grid that can be applied to the whole dataset (mostly for data plotting)
    positions_x_uniform, positions_y_uniform = _get_uniform_grid(positions_x_all, positions_y_all)
    logger.info("Generating common uniform grid: success.")

    # Create the arrays of XRF amplitudes for each emission line and normalize them
    eline_list, eline_data = _get_eline_data(
        scan_img_dict=scan_img_dict,
        scaler_name=scaler_name,
        eline_selected=eline_selected,
        eline_alignment=eline_alignment,
    )
    logger.info("Extracting XRF maps for emission lines: success.")

    if interpolation_enable:
        # Interpolate each image. I tried to use common uniform grid to do interpolation,
        #   but it didn't work very well. In the current implementation, the interpolation
        #   of each set is performed separately using the uniform grid specific for the set.
        for eline, data in eline_data.items():
            n_scans, _, _ = data.shape
            for n in range(n_scans):
                data[n, :, :], _, _ = grid_interpolate(
                    data[n, :, :], xx=positions_x_all[n, :, :], yy=positions_y_all[n, :, :]
                )
        logger.info("Interpolating XRF maps to uniform grid: success.")
    else:
        logger.info("Interpolating XRF maps to uniform grid: skipped.")

    # Align the stack of images
    if alignment_enable:
        eline_data_aligned = _align_stacks(
            eline_data=eline_data,
            eline_alignment=eline_alignment,
            alignment_starts_from=alignment_starts_from,
            normalize_alignment_stack=normalize_alignment_stack,
        )
        logger.info("Alignment of the image stack: success.")
    else:
        eline_data_aligned = eline_data
        logger.info("Alignment of the image stack: skipped.")

    if seq_generate_xanes_map:
        scan_absorption_refs = _interpolate_references(scan_energies_shifted, ref_energy, ref_data)

        skip_bl_sub = True
        if subtract_pre_edge_baseline:
            logger.info("Subtracting pre-edge baseline")
            try:
                xrf_data = subtract_xanes_pre_edge_baseline(
                    eline_data_aligned[eline_selected], scan_energies_shifted, eline_selected
                )
                skip_bl_sub = False
            except RuntimeError as ex:
                logger.error(f"XANES baseline was not subtracted: {ex}")
        if skip_bl_sub:
            xrf_data = eline_data_aligned[eline_selected]

        logger.info(f"Fitting XANES specta using '{fitting_method}' method")
        xanes_map_data, xanes_map_rfactor, _ = fit_spectrum(
            xrf_data, scan_absorption_refs, method=fitting_method, rate=fitting_descent_rate
        )

        # Scale xanes maps so that the values represent counts
        n_refs, _, _ = xanes_map_data.shape
        xanes_map_data_counts = np.zeros(shape=xanes_map_data.shape)
        for n in range(n_refs):
            xanes_map_data_counts[n, :, :] = xanes_map_data[n, :, :] * np.sum(scan_absorption_refs[:, n])

        logger.info("XANES fitting: success.")
    else:
        scan_absorption_refs = None
        xanes_map_data = None
        xanes_map_data_counts = None
        xanes_map_rfactor = None
        logger.info("XANES fitting: skipped.")

    processing_results = {
        # The processing results
        "eline_data_aligned": eline_data_aligned,
        "xanes_map_data": xanes_map_data,
        "xanes_map_data_counts": xanes_map_data_counts,
        "xanes_map_rfactor": xanes_map_rfactor,
        # Initial dataset information
        "scan_energies": scan_energies,  # Those values are used for logging only
        "scan_energies_shifted": scan_energies_shifted,  # Those values are used for processing and plotting
        "scan_ids": scan_ids,
        "files_h5": files_h5,
        # Global positions (uniform grid based on avarage values of position coordinates,
        #   may be useful for plotting data)
        "positions_x_uniform": positions_x_uniform,
        "positions_y_uniform": positions_y_uniform,
        # The values of absorption references sampled at energy values in 'scan_energies'
        "scan_absorption_refs": scan_absorption_refs,
    }

    return processing_results


def _save_xanes_processing_results(
    *, wd, eline_selected, ref_labels, output_file_formats, processing_results, output_save_all
):
    r"""
    Implements one of the final steps of the processing sequence: saving processing results.
    Currently only TIFF files are saved: TIFF with the aligned stack of XRF maps and TIFF with
    XANES maps. In addition, a .txt file is saved that contains of the list of images included in
    each TIFF.

    Parameters
    ----------

    wd : str
        working directory where the output data files are saved

    eline_selected : str
        the name of the selected emission line ("Ca_K", "Fe_K", etc.). The emission line
        of interest.

    ref_labels : list(str)
        list of labels for the references

    output_file_formats : list(str)
        list of output file formats

    processing_results : dict
        Results of processing returned by the function '_compute_xanes_maps'.

    output_save_all : bool
        indicates if processing results for every emission line, which is selected for XRF
        fitting, are saved. If set to ``True``, the (aligned) stacks of XRF maps are saved
        for each element. If set to False, then only the stack for the emission line
        selected for XANES fitting is saved (argument ``emission_line``).
    """
    positions_x_uniform = processing_results["positions_x_uniform"]
    positions_y_uniform = processing_results["positions_y_uniform"]

    eline_data_aligned = processing_results["eline_data_aligned"]
    xanes_map_data_counts = processing_results["xanes_map_data_counts"]
    xanes_map_rfactor = processing_results["xanes_map_rfactor"]

    scan_energies = processing_results["scan_energies"]  # Only for logging !!!
    scan_energies_shifted = processing_results["scan_energies_shifted"]
    scan_ids = processing_results["scan_ids"]
    files_h5 = processing_results["files_h5"]

    if "tiff" in output_file_formats:
        pos_x, pos_y = positions_x_uniform[0, :], positions_y_uniform[:, 0]
        _save_xanes_maps_to_tiff(
            wd=wd,
            eline_data_aligned=eline_data_aligned,
            eline_selected=eline_selected,
            xanes_map_data=xanes_map_data_counts,
            xanes_map_rfactor=xanes_map_rfactor,
            xanes_map_labels=ref_labels,
            scan_energies=scan_energies,
            scan_energies_shifted=scan_energies_shifted,
            scan_ids=scan_ids,
            files_h5=files_h5,
            positions_x=pos_x,
            positions_y=pos_y,
            output_save_all=output_save_all,
        )


def _plot_processing_results(
    *,
    ref_energy,
    ref_data,
    ref_labels,
    plot_position_axes_units,
    plot_use_position_coordinates,
    eline_selected,
    processing_results,
    fitting_method,
    fitting_descent_rate,
    incident_energy_shift_keV,
    subtract_pre_edge_baseline,
    wd,
):
    r"""
    Implements one of the final steps of the processing sequence: plotting processing results.
    The data is displayed on a set of Matplotlib figures:

    -- interactive plot of a stack of aligned XRF maps. Interactive options include switching
       between emission lines, browsing images of the stack and displaying of XANES spectrum
       for a selected pixel. If the emission line ``eline_selected`` is activated in the window
       and XANES maps were computed, then the results of fitting to references is displayed along
       with the XANES spectrum.

    -- plot of absorption references (if available)

    -- XANES map for each reference (if available)

    Parameters
    ----------

    ref_energy : ndarray(float), 1D
        values of incident energy for XANES reference data specified in ``ref_data``. If
        ``ref_data`` has shape (N, M), then ``ref_energy`` must have N elements.

    ref_data : ndarray(float), 2D
        reference data for the element states, used for XANES fitting. The array of shape (N, M)
        contains reference data for M element states specified at N energy points.

    ref_labels : list(str)
        list of labels for the references

    plot_position_axes_units : str
        units for position coordinates along X and Y axes. The units are used while
        plotting the results vs. position coordinates. The string specifying units
        may contain LaTeX expressions: for example ``"$\mu $m"`` will print units
        of ``micron`` as part of X and Y axes labels.

    plot_use_position_coordinates : bool
        results (image stack and XANES maps) are plotted vs. position coordinates if
        the parameter is set to True, otherwise images are plotted vs. pixel number

    eline_selected : str
        the name of the selected emission line ("Ca_K", "Fe_K", etc.). The emission line
        of interest.

    processing_results : dict
        Results of processing returned by the function '_compute_xanes_maps'.

    fitting_method : str
        method used for fitting, the currently supported methods are 'nnls' and 'admm'

    fitting_descent_rate : float
        descent rate for fitting algorithm, currently used only for ADMM fitting

    incident_energy_shift_keV : float
        the value of the shift (already) applied to energy points of the observed data

    subtract_pre_edge_baseline : bool
        indicates if pre-edge baseline is subtracted from XANES spectra before fitting.
        Currently the subtracted baseline is constant, computed as a median value of spectrum
        points in the pre-edge region.

    wd : str
        working directory (for saving data files)
    """

    positions_x_uniform = processing_results["positions_x_uniform"]
    positions_y_uniform = processing_results["positions_y_uniform"]

    eline_data_aligned = processing_results["eline_data_aligned"]
    xanes_map_data = processing_results["xanes_map_data"]
    xanes_map_data_counts = processing_results["xanes_map_data_counts"]
    xanes_map_rfactor = processing_results["xanes_map_rfactor"]

    # Only 'shifted' values of scan energies are used for plotting, since those are
    #   the values used for processing.
    scan_energies_shifted = processing_results["scan_energies_shifted"]

    scan_absorption_refs = processing_results["scan_absorption_refs"]

    axes_units = plot_position_axes_units
    # If positions are none, then axes units are pixels
    pos_x, pos_y = (
        (positions_x_uniform[0, :], positions_y_uniform[:, 0]) if plot_use_position_coordinates else (None, None)
    )

    # The following arrays must be different from None if XANES maps were generated
    if (
        (scan_absorption_refs is not None)
        and (xanes_map_data is not None)
        and (xanes_map_data_counts is not None)
        and (xanes_map_rfactor is not None)
    ):

        plot_absorption_references(
            ref_energy=ref_energy,
            ref_data=ref_data,
            scan_energies=scan_energies_shifted,
            scan_absorption_refs=scan_absorption_refs,
            ref_labels=ref_labels,
            block=False,
        )

        figures = []
        for n, map_data in enumerate(xanes_map_data_counts):
            fig = plot_xanes_map(
                map_data,
                label=ref_labels[n],
                block=False,
                positions_x=pos_x,
                positions_y=pos_y,
                axes_units=axes_units,
            )
            figures.append(fig)

        plot_xanes_map(
            xanes_map_rfactor,
            label="R-factor",
            block=False,
            positions_x=pos_x,
            positions_y=pos_y,
            axes_units=axes_units,
            map_margin=10,
        )

    # Show image stacks for the selected elements
    show_image_stack(
        eline_data=eline_data_aligned,
        energies=scan_energies_shifted,
        eline_selected=eline_selected,
        positions_x=pos_x,
        positions_y=pos_y,
        axes_units=axes_units,
        xanes_map_data=xanes_map_data,
        scan_absorption_refs=scan_absorption_refs,
        ref_labels=ref_labels,
        ref_energy=ref_energy,
        ref_data=ref_data,
        fitting_method=fitting_method,
        fitting_descent_rate=fitting_descent_rate,
        energy_shift_keV=incident_energy_shift_keV,
        subtract_pre_edge_baseline=subtract_pre_edge_baseline,
        wd=wd,
    )


def _load_dataset_from_hdf5(*, start_id, end_id, wd_xrf, load_fit_results=True):
    r"""
    Load dataset from processed HDF5 files

    Parameters
    ----------

    wd_xrf : str
        full (absolute) path name to the directory that contains processed HDF5 files
    load_fit_results : bool
        indicates if fit results should be loaded. If set to False, then only metadata
        is loaded and output dictionary ``scan_img_dict`` is empty.
    """

    # The list of file names
    files_h5 = [fl.name for fl in os.scandir(path=wd_xrf) if fl.name.lower().endswith(".h5")]
    # Sorting file names will make loading a little more orderly, but generally file names
    #   can be arbitrary (scan ID is not extracted from file names)
    files_h5.sort()

    scan_ids = []
    scan_energies = []
    scan_img_dict = []
    for fln in files_h5:
        img_dict, _, mdata = read_hdf_APS(
            working_directory=wd_xrf,
            file_name=fln,
            load_summed_data=load_fit_results,
            load_each_channel=False,
            load_processed_each_channel=False,
            load_raw_data=False,
            load_fit_results=load_fit_results,
            load_roi_results=False,
        )

        if "scan_id" not in mdata:
            logger.error(f"Metadata value 'scan_id' is missing in data file '{fln}': the file was not loaded.")
            continue

        # Make sure that the scan ID is in the specified range (if the range is
        #   not specified, then process all data files
        if (
            (start_id is not None)
            and (mdata["scan_id"] < start_id)
            or (end_id is not None)
            and (mdata["scan_id"] > end_id)
        ):
            continue

        if "instrument_mono_incident_energy" not in mdata:
            logger.error(
                "Metadata value 'instrument_mono_incident_energy' is missing "
                f"in data file '{fln}': the file was not loaded."
            )

        if load_fit_results:
            scan_img_dict.append(img_dict)
        scan_ids.append(mdata["scan_id"])
        scan_energies.append(mdata["instrument_mono_incident_energy"])

    return scan_ids, scan_energies, scan_img_dict, files_h5


def _check_dataset_consistency(*, scan_ids, scan_img_dict, files_h5, scaler_name, eline_selected, eline_alignment):
    r"""
    Perform some checks for consistency of input data and parameters
    before starting XANES mapping steps. The following processing steps
    assume that the data is consistent.

    Parameters
    ----------

    scan_ids : list(int)
        list of scan IDs for the XRF map stack

    scan_img_dict : list(dict)
        list of dictionaries, each dictionary contains dataset for the respective scan ID
        from the ``scan_id`` list.

    files_h5 : list(str)
        list of file names

    scaler_name : str
        name of the scaler to be used for data normalization. The function checks if the
        scaler is present in each dataset

    eline_selected : str
        name of the emission line selected for XANES. The function checks if the emission
        line is represented in each dataset

    eline_alignment : str
        name of the emission line selected for stack alignment. The function checks if the emission
        line is represented in each dataset. The emission line may be the same as the one
        selected for XANES.

    The function raises an exception if the dataset is inconsistent, parameters have invalid values
    or important data is missing.
    """

    if not scan_img_dict:
        raise RuntimeError("Loaded dataset is empty. No data to process.")

    # First check if processing dataset exists. The following call will raise the exception
    #   if there is not dataset with processed data.
    _get_dataset_name(scan_img_dict[0])

    # Create a list of 'img_dict' keys
    img_data_keys = []
    img_data_keys_union = set()
    for img in scan_img_dict:
        ks = _get_img_keys(img)
        img_data_keys.append(ks)
        img_data_keys_union = img_data_keys_union.union(ks)
    img_data_keys_union = list(img_data_keys_union)

    def _raise_error_exception(slist, data_tuples, msg_phrase):
        """
        Report error by raising the RuntimeError exception. Print the list
        of scan_ids and file names for the files that have the problem

        Parameters
        ----------

        slist : list(int)
            the list of indices of the scans

        scan_ids : list(int)
            the list of scan IDs

        files_h5 : list(str)
            the list of file names

        msg_phrase : str
            the string, representing varying part of the message
        """
        msg = f"Some scans in the dataset {msg_phrase}:\n"
        for n in slist:
            for dt in data_tuples:
                msg += f"    {dt[1]}:  {dt[0][n]}"
            msg += "\n"
        raise RuntimeError(msg)

    def _check_for_specific_elines(img_data_keys, elines, files_h5, scan_ids, msg_phrase):
        r"""
        Check if all emission lines from the list are present in all scan data
        """
        slist = []
        for n, ks in enumerate(img_data_keys):
            all_lines_present = True
            for e in elines:
                if e not in ks:
                    all_lines_present = False
                    break
            if not all_lines_present:
                slist.append(n)

        if slist:
            _raise_error_exception(
                slist=slist, data_tuples=[(scan_ids, "scan ID"), (files_h5, "file")], msg_phrase=msg_phrase
            )

    def _check_for_identical_eline_key_set(img_data_keys, img_data_keys_union):
        r"""
        Check if all processed datasets contain data for the same emission lines.
        If not, then processing has to be run again on the datasets
        """
        success = True
        all_eline_keys = _get_eline_keys(img_data_keys_union)
        all_eline_keys.sort()
        for n, ks in enumerate(img_data_keys):
            eks = _get_eline_keys(ks)
            eks.sort()
            if eks != all_eline_keys:
                success = False
                break

        if not success:
            msg = (
                "Files in the dataset were processed for different sets of emission lines:\n"
                "    may be fixed by rerunning the processing of the dataset "
                "with the same parameter file."
            )
            raise RuntimeError(msg)

    def _check_for_positional_data(scan_img_dict):
        slist = []
        for n, img in enumerate(scan_img_dict):
            if ("positions" not in img) or ("x_pos" not in img["positions"]) or ("y_pos" not in img["positions"]):
                slist.append(n)

        if slist:
            _raise_error_exception(
                slist=slist,
                data_tuples=[(scan_ids, "scan ID"), (files_h5, "file")],
                msg_phrase="have no positional data ('x_pos' or 'y_pos')",
            )

    def _check_for_identical_image_size(scan_img_dict, files_h5, scan_ids):
        # Create the list of image sizes for all files
        xy_list = []
        for img in scan_img_dict:
            xy_list.append(img["positions"]["x_pos"].shape)

        # Determine if all sizes are identical
        if any([_ != xy_list[0] for _ in xy_list]):
            _raise_error_exception(
                slist=list(range(len(xy_list))),
                data_tuples=[(scan_ids, "scan_ID"), (files_h5, "file"), (xy_list, "image size")],
                msg_phrase="contain XRF maps of different size",
            )

    # Check existence of the scaler only if scaler name is specified
    if scaler_name:
        _check_for_specific_elines(
            img_data_keys=img_data_keys,
            elines=[scaler_name],
            files_h5=files_h5,
            scan_ids=scan_ids,
            msg_phrase=f"have no scaler data ('{scaler_name}')",
        )

    _check_for_identical_eline_key_set(img_data_keys=img_data_keys, img_data_keys_union=img_data_keys_union)

    _check_for_specific_elines(
        img_data_keys=img_data_keys,
        elines=[eline_selected, eline_alignment],
        files_h5=files_h5,
        scan_ids=scan_ids,
        msg_phrase=f"have no emission line data ('{eline_selected}' or '{eline_alignment}')",
    )

    _check_for_positional_data(scan_img_dict=scan_img_dict)

    _check_for_identical_image_size(scan_img_dict=scan_img_dict, files_h5=files_h5, scan_ids=scan_ids)


# ============================================================================================
#   Functions for parameter manipulation


def check_elines_activation_status(scan_energies, emission_line):
    r"""
    Check if ``emission_line`` is activated at each scan energy in the list ``scan_energies``

    Parameters
    ----------

    scan_energies : list(float)
        the list (or an array) of incident beam energy values, keV. The energy values
        don't need to be sorted.

    emission_line : str
        valid name of the emission line (supported by ``scikit-beam``) in the
        form of K_K or Fe_K

    Returns
    -------
        the list of ``bool`` values, indicating if the emission line is activated at
        the respective energy in the list ``scan_energies``
    """

    is_activated = [check_if_eline_is_activated(emission_line, _) for _ in scan_energies]
    return is_activated


def adjust_incident_beam_energies(scan_energies, emission_line):
    r"""
    Adjust the values of incident beam energy in the list ``scan_energies`` so that
    fitting for the ``emission_line`` could be done at each energy value. Adjustment
    is done by finding the lowest incident energy value in the list that still activates
    the ``emission_line`` and setting every smaller energy value to the lowest value.

    Parameters
    ----------

    scan_energies : list(float)
        the list (or ndarray) of incident beam energy values, keV

    emission_line : str
        valid name of the emission line (supported by ``scikit-beam``) in the
        form of K_K or Fe_K

    Returns
    -------
        the list of adjusted values of the incident energy. The returned list has the
        same dimensions as ``scan_energies``.
    """

    activation_status = check_elines_activation_status(scan_energies, emission_line)
    e_activation = np.asarray(scan_energies)[activation_status]

    if not e_activation.size:
        raise RuntimeError(
            f"The emission line '{emission_line}' is not activated\n"
            f"    in the range of energies {min(scan_energies)} - {max(scan_energies)} keV.\n"
            f"    Check if the emission line is specified correctly."
        )

    min_activation_energy = np.min(e_activation)
    return [max(_, min_activation_energy) for _ in scan_energies]


def subtract_xanes_pre_edge_baseline(
    xrf_map_stack, scan_energies, eline_selected, *, pre_edge_upper_keV=-0.01, non_negative=True
):
    r"""
    Subtract baseline from XANES spectrum of an XRF map stack. The stack is represented
    as multidimensional array ``xrf_map_stack``: the spectral points are arranged along ``axis=0``.
    If ``xrf_map_stack`` is 1D array, then it is assumed that it represents a single spectrum

    Subtraction of the pre-edge baseline. The baseline is assumed constant throughout
    the spectrum and estimated separately for each pixel. Pre-edge is found as a set of
    spectral points with energies smaller than the lowest energy that activates
    ``eline_selected`` by the value of ``- pre_edge_upper_keV``. The baseline value is estimated
    as a median of all pre-edge points.

    Parameters
    ----------

    xrf_map_stack : ndarray
        single- or multidimensional array with XANES map spectra. The spectral points are
        along ``axis=0``.

    scan_energies : list or ndarray
        incident energies for spectral points (same number of points as axis 0 of ``xrf_map_stack``)

    eline_selected : str
        emission line, i.e. "Fe_K"

    pre_edge_upper_keV : float
        the upper boundary of the pre-edge region relative to the lowest activation energy.
        The value is added to the lowest emission line activation energy, so it should be
        negative.

    non_negative : bool
        True - set all negative values to zero, False - return the results of baseline subtraction
        without change

    Returns
    -------

        array of the same dimensions as ``xrf_map_stack`` that contains XANES spectra with
        subtracted baseline.

    Raises
    ------

        RuntimeError if no pre-edge points are detected. Typically this means that all
        the spectral point in ``scan_energies`` activate the emission line.
    """

    scan_energies = np.asarray(scan_energies)  # Make sure that 'scan_energies' is an array
    assert (
        scan_energies.ndim == 1
    ), f"Parameter 'scan_energies' must be 1D array (number of dimensions {scan_energies.ndim})"

    assert xrf_map_stack.shape[0] == scan_energies.shape[0], (
        f"The shapes of 'xrf_map_stack' {xrf_map_stack.shape} and 'scan_energies'"
        f" {scan_energies.shape} do not match"
    )

    # If data is 1D (contains only one spectrum), then add one extra dimension for consistency
    is_data_1d = xrf_map_stack.ndim == 1
    if is_data_1d:
        xrf_map_stack = np.expand_dims(xrf_map_stack, axis=1)

    # Deterimine if 'eline_selected' is activated for each point
    e_status = check_elines_activation_status(scan_energies, eline_selected)
    # Find the incident energy of the edge
    edge_energy = np.min(scan_energies[np.asarray(e_status)])
    # Mark the points in the pre-edge region
    pre_edge_pts = scan_energies < (edge_energy + pre_edge_upper_keV)
    # Find pre-edge points (points for which the emission line is not activated
    if not np.sum(pre_edge_pts):
        raise RuntimeError("No pre-edge points were found. Baseline can not be estimated.")
    # Separate pre-edge points (typically consecutive points at the lower values of the index,
    #   but the function will work with randomly permuted energy array)
    xrf_map_pre_edge = xrf_map_stack[np.asarray(pre_edge_pts)]
    # Find constant baseline values: baseline is different for each pixel
    baseline_const = np.median(xrf_map_pre_edge, axis=0)
    # Subtract the baseline
    xrf_map_out = xrf_map_stack - baseline_const
    if non_negative:
        xrf_map_out = np.clip(xrf_map_out, a_min=0, a_max=None)

    # Remove extra dimension if input data is 1D
    if is_data_1d:
        xrf_map_out = np.squeeze(xrf_map_out, axis=1)

    return xrf_map_out


# ============================================================================================
#   Functions used for processing


def _get_uniform_grid(positions_x_all, positions_y_all):
    r"""
    Compute common uniform grid based on position coordinates for a stack of maps.
    The grid is computed in two steps: the median of X and Y coordinates is computed
    for each point of the map over the stack of maps (for X and Y positions the 3D array
    containing a stack of point positions is converted to 2D array containing a single map);
    uniform grid is generated based on median X and Y values in the map.

    Parameters
    ----------

    positions_x_all : ndarray, 3D
        values of X coordinates for each point of the map. Shape (K, M, N): the stack
        of K maps, each map is MxN pixels.

    positions_y_all : ndarray, 3D
        values of Y coordinates.

    Returns
    -------

    positions_x_uniform : ndarray, 3D
        uniform grid with X coordinates, shape (K, M, N)

    positions_y_uniform : ndarray, 3D
        uniform grid with Y coordinates, shape (K, M, N)
    """

    # Median positions seem to be the best for generating common uniform grid
    positions_x_median = np.median(positions_x_all, axis=0)
    positions_y_median = np.median(positions_y_all, axis=0)
    # Generate uniform grid (interpolation function simply generates the grid if
    #   if only position data is provided).
    _, positions_x_uniform, positions_y_uniform = grid_interpolate(None, positions_x_median, positions_y_median)
    return positions_x_uniform, positions_y_uniform


def _get_eline_data(scan_img_dict, scaler_name, eline_selected=None, eline_alignment=None):
    r"""
    Rearrange data for more convenient processing: 1. Create the list of available emission lines.
    2. Create the dictionary of stacks of XRF maps: key - emission line name, value - 3D ndarray of
    the shape (K, M, N), where K is the number of maps in the stack for the emission line, each
    map is MxN pixels.

    The function is also normalizing the data by the scaler if the scaler is specified (not None).

    Parameters
    ----------

    scan_img_dict : list(dict)
        list of dictionaries, each dictionary contains dataset from one XRF scan.

    scaler_name : str
        name of the scaler. The name should be present in the dataset.
        Data should be checked for consistency at the beginning of the process.
        ``scaler_name`` may be None. In this case normalization is skipped.

    eline_selected : str
        the name of the selected emission line ("Ca_K", "Fe_K", etc.). The emission line
        of interest.

    eline_alignment : str
        the name of the emission line ("Ca_K", "Fe_K", etc.) used for alignment of image stack.
        The emission line may be the same as ``eline_selected``.

    Returns
    -------

    eline_list : list(str)
        list of available emission lines

    eline_data : dict(ndarray)
        dictionary of stacks of XRF maps: key - emission line name, value - 3D ndarray of
        the shape (K, M, N), where K is the number of maps in the stack for the emission line,
        each map is MxN pixels.
    """
    eline_list = _get_eline_keys(_get_img_keys(scan_img_dict[0]))
    # If 'eline_selected' and 'eline_alignment' are specified and not present in the list,
    #   then add them to the end of the list
    for eline in (eline_selected, eline_alignment):
        if eline and (eline not in eline_list):
            eline_list.append(eline)
    eline_data = {}
    for eline in eline_list:
        data = []
        for img_dict in scan_img_dict:
            d = _get_img_data(img_dict=img_dict, key=eline)
            if scaler_name:  # Normalization
                d = normalize_data_by_scaler(d, _get_img_data(img_dict=img_dict, key=scaler_name))
            data.append(d)
        eline_data[eline] = np.asarray(data)
    return eline_list, eline_data


def _align_stacks(eline_data, eline_alignment, alignment_starts_from="top", normalize_alignment_stack=True):
    r"""
    Align stacks of maps from the dictionary ``eline_data`` based on the stack for
    emission line specified by ``eline_alignment``. Alignment may be performed
    starting from the ``"top"`` (default) or the ``"bottom"`` of the stack.

    Parameters
    ----------

    eline_data : dict(ndarray)
        dictionary of stacks of XRF maps: key - emission line name, value - 3D ndarray of
        the shape (K, M, N), where K is the number of maps in the stack for the emission line,
        each map is MxN pixels.

    eline_alignment : str
        name of the emission line. The stack of maps for this emission line will be
        used to align the rest of the stacks.

    alignment_starts_from : str
        order of the alignment. The allowed values are ``"top"`` (alignment starts from
        the map acquired with the highest incident energy) and ``"bottom"`` (lowest energy).

    normalize_alignment_stack : bool
        enable normalization of the image stack used for alignment
        Default: ``True``

    Returns
    -------

    eline_data_aligned : dict(ndarray)
        dictionary of stacks of aligned XRF maps. All dimensions match the dimensions of
        ``eline_data``.
    """
    alignment_starts_from = alignment_starts_from.lower()
    assert alignment_starts_from in [
        "top",
        "bottom",
    ], f"Parameter 'alignment_starts_from' has invalid value: '{alignment_starts_from}'"

    def _flip_stack(img_stack, alignment_starts_from):
        """Flip image stack if alignment should be started from the top image"""
        if alignment_starts_from == "top":
            return np.flip(img_stack, 0)
        else:
            return img_stack

    """Align stack of XRF maps for each element"""
    sr = StackReg(StackReg.TRANSLATION)

    # Normalize data used to compute matrix
    logger.info(f"Stacks are aligned using the map '{eline_alignment}'")
    data = np.array(eline_data[eline_alignment])  # Create a copy
    if normalize_alignment_stack:
        logger.info("Normalizing the alignment stack ...")
        for n in range(data.shape[0]):
            data[n, :, :] = data[n, :, :] / np.sum(data[n, :, :])
    # Add the data selected for stack alignment in the original dictionary of stacks
    #   It will also be present in the aligned stack
    eline_data["ALIGN"] = data

    sr.register_stack(_flip_stack(data, alignment_starts_from), reference="previous")

    eline_data_aligned = {}
    for eline, data in eline_data.items():
        data_prepared = _flip_stack(data, alignment_starts_from)
        data_transformed = sr.transform_stack(data_prepared)
        data_transformed = _flip_stack(data_transformed, alignment_starts_from)
        eline_data_aligned[eline] = data_transformed.clip(min=0)

    return eline_data_aligned


def _interpolate_references(energy, energy_refs, absorption_refs):
    r"""
    Interpolate XANES references

    Parameters
    ----------

    energy : ndarray(float), 1D
        array that represents energy values for the interpolated reference data (P points)

    energy_refs : ndarray(float), 1D
        array that represents energy axis for XANES references (N points)

    absorption_refs : ndarray(float), 2D
        array of XANES references, shape (N, K), where K is the number of references
        and N is the number of points (same number of points as in ``energy_refs``).

    Returns
    -------

    interpolated_refs : ndarray(float), 2D
        array of interpolated XANES references, shape (P, K), where K is the number of references
        and P is the number of points (same number of points as in ``energy``).
    """
    _, n_states = absorption_refs.shape
    interpolated_refs = np.zeros(shape=[len(energy), n_states], dtype=float)
    for n in range(n_states):
        a_ref = absorption_refs[:, n]
        interpolated_refs[:, n] = np.interp(energy, energy_refs, a_ref)
    return interpolated_refs


# ==============================================================================================
#     Functions for plotting the results


def show_image_stack(
    *,
    eline_data,
    energies,
    eline_selected,
    positions_x=None,
    positions_y=None,
    axes_units=None,
    xanes_map_data=None,
    scan_absorption_refs=None,
    ref_labels=None,
    ref_energy=None,
    ref_data=None,  # Those are original sets of reference data
    fitting_method="nnls",
    fitting_descent_rate=0.2,
    energy_shift_keV=0.0,
    subtract_pre_edge_baseline=True,
    wd=None,
):
    r"""
    Display XRF Map stack


    Parameters
    ----------

    positions_x : ndarray
        values of coordinate X of the image. Must match the size of dimension 1 of the image

    positions_y : ndarray
        values of coordinate Y of the image. Must match the size of dimension 0 of the image
    """

    if not eline_data:
        logger.warning("Emission line data dictionary is empty. There is nothing to plot.")
        return

    class EnergyMapPlot:
        def __init__(
            self,
            *,
            energy,
            stack_all_data,
            label_default,
            positions_x=None,
            positions_y=None,
            axes_units=None,
            xanes_map_data=None,
            scan_absorption_refs=None,
            ref_labels=None,
            ref_energy=None,
            ref_data=None,
            fitting_method="nnls",
            fitting_descent_rate=0.2,
            energy_shift_keV=0.0,
            subtract_pre_edge_baseline=True,
            wd=None,
        ):
            r"""
            Parameters
            ----------

            energy : ndarray, 1-D
                values of energy, the number of elements must match the number of images in the stacks

            stack_all_data : dict of 3-D ndarrays
                dictionary that contains stacks of images: key - the label of the stack,
                value - 3D ndarray with dimension 0 representing the image number and dimensions
                1 and 2 representing positions along Y and X axes.

            label_default : str
                default label which is used to select the image stack which is selected at the start.
                If the value does not match any keys of ``stack_all_data``, then it is ignored

            positions_x : ndarray
                values of coordinate X of the image. Only the first and the last value of the list
                is used to set the axis range, so [xmin, xmax] is sufficient.

            positions_y : ndarray
                values of coordinate Y of the image. Only the first and the last value of the list
                is used to set the axis range, so [ymin, ymax] is sufficient.

            axes_units : str
                units for X and Y axes that are used if ``positions_x`` and ``positions_y`` are specified
                Units may include latex expressions, for example "$\mu $m" will print units of microns.
                If ``axes_units`` is None, then printed label will not include unit information.

            xanes_map_data : 3D ndarray
                The stack of XRF maps: shape = (K, N, M) - the stack of K maps, the size of each map is NxM

            scan_absorption_refs : 2D ndarray
                Absorption references, used to display fitting results and for real-time fitting.
                Shape = (K, Q) - K data points, Q references

            ref_labels : list(str)
                The list of Q reference labels, used for data plotting

            ref_energy : array (1D)
                The array of energy values for original (not resampled) loaded reference data (B points)

            ref_data : array (2D)
                The array of reference data, shape (B,C) for C references. The array needs to be resampled
                before using for processing

            fitting_method : str
                Fitting method used for real-time fitting (used when displaying fitting for selections
                of pixels)

            fitting_descent_rate : float
                descent rate for fitting algorithm, currently used only for ADMM fitting

            energy_shift_keV : float
                shift (in keV) applied to incident energy for observed data. The energy values
                supplied to this function are already shifted by this amount. The user will
                have opportunity to experiment with different values of energy shift, so the
                online processing routine will resample the references for new shifted
                energy points. This online processing will not influence data in the rest of the
                program.

            subtract_pre_edge_baseline : bool
                indicates if pre-edge baseline is subtracted from XANES spectra before fitting.
                Currently the subtracted baseline is constant, computed as a median value of spectrum
                points in the pre-edge region.

            wd : str
                working directory
            """

            self.wd = wd  # Working directory (for saving files)

            # The following are the fitting parameters that are sent to the fitting algorithm
            self.fitting_method = fitting_method
            self.fitting_descent_rate = fitting_descent_rate
            # This is the value of already applied energy shift
            self.incident_energy_shift_keV = energy_shift_keV
            # This value will changed and the difference of the updated and the original value
            #   will be applied to the energy points before processing
            self.incident_energy_shift_keV_updated = energy_shift_keV

            self.subtract_pre_edge_baseline = subtract_pre_edge_baseline

            self.label_fontsize = 15
            self.coord_fontsize = 10

            self.stack_all_data = stack_all_data
            self.energy_original = energy  # The energy values shifted by 'self.incident_energy_shift_keV'
            self.energy = energy  # The contents of this array will change if additional shift is applied
            self.label_default = label_default

            # XANES spectrum based on currently selected area in the plot. Once it is set,
            # it can be used along with 'self.energy' (e.g. saved to file)
            self.xanes_spectrum_selected = None

            self.labels = list(eline_data.keys())

            self.textbox_nlabel = None
            self.busy = False

            self.xanes_map_data = xanes_map_data
            self.absorption_refs = scan_absorption_refs
            self.ref_labels = ref_labels

            # Original reference data (used for resampling when fitting with different energy shift)
            self.ref_energy = ref_energy
            self.ref_data = ref_data

            # References to Arrow and Rectangle patches, used to mark selection on the stack image
            self.img_arrow = None
            self.img_rect = None

            # The state of the left mouse button (True - button is in pressed state)
            self.button_pressed = False

            if self.label_default not in self.labels:
                logger.warning(
                    f"XRF Energy Plot: the default label {self.label_default} "
                    "is not in the list and will be ignored"
                )
                self.label_default = self.labels[0]  # Set the first emission line as default
            self.label_selected = self.label_default
            self.select_stack()

            n_images, ny, nx = self.stack_selected.shape
            self.n_energy_selected = int(n_images / 2)  # Select image in the middle of the stack
            self.img_selected = self.stack_selected[self.n_energy_selected, :, :]

            # Select point in the middle of the plot, the coordinates of the point are in pixels
            self.pts_selected = [int(nx / 2), int(ny / 2), int(nx / 2), int(ny / 2)]

            # Check existence or the size of 'positions_x' and 'positions_y' arrays
            ny, nx = self.img_selected.shape
            if (positions_x is None) or (positions_y is None):
                self.pos_x_min, self.pos_x_max = 0, nx - 1
                self.pos_y_min, self.pos_y_max = 0, ny - 1
                self.pos_dx, self.pos_dy = 1, 1
                self.axes_units = "pixels"
            else:
                self.pos_x_min, self.pos_x_max = positions_x[0], positions_x[-1]
                self.pos_y_min, self.pos_y_max = positions_y[0], positions_y[-1]
                self.pos_dx = (self.pos_x_max - self.pos_x_min) / max(nx - 1, 1)
                self.pos_dy = (self.pos_y_max - self.pos_y_min) / max(ny - 1, 1)
                self.axes_units = axes_units if axes_units else ""

        def select_stack(self):
            self.stack_selected = self.stack_all_data[self.label_selected]
            if not isinstance(self.stack_selected, np.ndarray) or not self.stack_selected.ndim == 3:
                raise ValueError("Image stack must be 3-D numpy array.")

        def set_cbar_range(self):
            v_min = np.min(self.stack_selected)
            v_max = np.max(self.stack_selected)
            self.img_plot.set_clim(v_min, v_max)

        def draw_selection_mark(self):
            # Remove existing marked points and regions
            if self.img_arrow and self.img_arrow.figure:
                self.img_arrow.remove()
            if self.img_rect and self.img_rect.figure:
                self.img_rect.remove()

            x_min, y_min, x_max, y_max = self.pts_selected
            pt_x_min = x_min * self.pos_dx + self.pos_x_min
            pt_y_min = y_min * self.pos_dy + self.pos_y_min

            if (x_min == x_max) and (y_min == y_max):
                # Single point is marked with an arrow
                arr_param = {"length_includes_head": True, "color": "r", "width": self.arr_width}
                self.img_arrow = FancyArrow(
                    pt_x_min - self.arr_length,
                    pt_y_min - self.arr_length,
                    self.arr_length,
                    self.arr_length,
                    **arr_param,
                )
                self.ax_img_stack.add_patch(self.img_arrow)
            else:
                # Region is marked with rectangle
                pt_x_max = x_max * self.pos_dx + self.pos_x_min
                pt_y_max = y_max * self.pos_dy + self.pos_y_min
                rect_param = {"linewidth": 1, "edgecolor": "r", "facecolor": "none"}
                self.img_rect = Rectangle(
                    (pt_x_min, pt_y_min), pt_x_max - pt_x_min, pt_y_max - pt_y_min, **rect_param
                )
                self.ax_img_stack.add_patch(self.img_rect)

        def show(self, block=True):

            self.textbox_nlabel = None

            self.fig = plt.figure(figsize=(11, 6), num="XRF MAPS")
            self.ax_img_stack = plt.axes([0.07, 0.25, 0.35, 0.65])
            self.ax_img_cbar = plt.axes([0.425, 0.26, 0.013, 0.63])
            self.ax_fluor_plot = plt.axes([0.55, 0.25, 0.4, 0.65])
            self.fig.subplots_adjust(left=0.07, right=0.95, bottom=0.25)

            self.t_bar = plt.get_current_fig_manager().toolbar

            x_label = f"X, {axes_units}" if axes_units else "X"
            y_label = f"Y, {axes_units}" if axes_units else "Y"
            self.ax_img_stack.set_xlabel(x_label, fontsize=self.label_fontsize)
            self.ax_img_stack.set_ylabel(y_label, fontsize=self.label_fontsize)

            # Compute parameters of the arrow (width and length), used to mark
            #   selected point on the image plot
            max_dim = max(abs(self.pos_x_max - self.pos_x_min), abs(self.pos_y_max - self.pos_y_min))
            self.arr_width = max_dim / 200
            self.arr_length = max_dim / 20

            # display image
            extent = [self.pos_x_min, self.pos_x_max, self.pos_y_max, self.pos_y_min]
            self.img_plot = self.ax_img_stack.imshow(self.img_selected, origin="upper", extent=extent)
            self.img_label_text = self.ax_img_stack.text(
                0.5,
                1.01,
                self.label_selected,
                ha="left",
                va="bottom",
                fontsize=self.label_fontsize,
                transform=self.ax_img_stack.axes.transAxes,
            )
            self.cbar = self.fig.colorbar(self.img_plot, cax=self.ax_img_cbar, orientation="vertical")
            self.ax_img_cbar.ticklabel_format(style="sci", scilimits=(-3, 4), axis="both")
            self.set_cbar_range()

            self.draw_selection_mark()

            self.redraw_fluorescence_plot()

            # define slider
            axcolor = "lightgoldenrodyellow"
            self.ax_slider_energy = plt.axes([0.2, 0.1, 0.6, 0.03], facecolor=axcolor)
            self.set_slider_energy_title()

            self.slider = Slider(
                self.ax_slider_energy,
                "Energy",
                0,
                len(self.energy) - 1,
                valinit=self.n_energy_selected,
                valfmt="%i",
            )

            self.ax_btn_save_spectrum = plt.axes([0.85, 0.1, 0.13, 0.05])
            self.btn_save_spectrum = Button(
                self.ax_btn_save_spectrum, "Save Spectrum", color="#00ff00", hovercolor="#ff0000"
            )
            self.btn_save_spectrum.on_clicked(self.btn_save_spectrum_clicked)

            self.ax_tb_energy_shift = plt.axes([0.85, 0.935, 0.1, 0.03])
            self.tb_energy_shift = TextBox(self.ax_tb_energy_shift, "Energy shift (keV):", label_pad=0.07)
            self.tb_energy_shift.set_val(self.incident_energy_shift_keV)
            self.tb_energy_shift.color = "#00ff00"
            self.tb_energy_shift.hovercolor = "#00ff00"
            # The following line actually updates the color, not just sets the attribute
            self.tb_energy_shift.ax.set_facecolor(self.tb_energy_shift.color)
            self.tb_energy_shift.on_submit(self.tb_energy_shift_onsubmit)

            if len(self.labels) <= 10:
                # Individual button per label (emission line). Only 10 buttons will fit windows
                self.create_buttons_each_label()
            else:
                # Alternative way to organize switching between labels
                self.create_buttons_prev_next()

            self.slider.on_changed(self.slider_update)

            self.fig.canvas.mpl_connect("button_press_event", self.canvas_onpress)
            self.fig.canvas.mpl_connect("button_release_event", self.canvas_onrelease)

            self.fig.canvas.draw_idle()

            plt.show(block=block)

        def show_fluor_point_coordinates(self):

            x_min, y_min, x_max, y_max = self.pts_selected

            pt_x_min = x_min * self.pos_dx + self.pos_x_min
            pt_y_min = y_min * self.pos_dy + self.pos_y_min

            if x_min == x_max and y_min == y_max:
                pt_x_str, pt_y_str = f"{pt_x_min:.5g}", f"{pt_y_min:.5g}"
            else:
                pt_x_max = x_max * self.pos_dx + self.pos_x_min
                pt_y_max = y_max * self.pos_dy + self.pos_y_min
                pt_x_str = f"{pt_x_min:.5g} .. {pt_x_max:.5g}"
                pt_y_str = f"{pt_y_min:.5g} .. {pt_y_max:.5g}"

            self.fluor_label_text = self.ax_fluor_plot.text(
                0.99,
                0.99,
                f"({pt_x_str}, {pt_y_str})",
                ha="right",
                va="top",
                fontsize=self.coord_fontsize,
                transform=self.ax_fluor_plot.axes.transAxes,
            )

        def create_buttons_each_label(self):
            r"""
            Create separate button for each label. The maximum number of labels
            that can be assigned unique buttons is 10. More buttons may
            not fit the screen
            """
            n_labels = len(self.labels)

            max_labels = 10
            assert n_labels <= max_labels, (
                f"The stack contains too many labels ({len(self.labels)}). "
                f"Maximum allowed number of labels is {max_labels}"
            )

            bwidth = 0.07  # Width of a button
            bgap = 0.01  # Gap between buttons

            pos_left_start = 0.5 - n_labels / 2 * bwidth - (n_labels - 1) / 2 * bgap

            # Create buttons
            self.btn_eline, self.ax_btn_eline = [], []
            for n in range(n_labels):
                p_left = pos_left_start + (bwidth + bgap) * n
                self.ax_btn_eline.append(plt.axes([p_left, 0.03, bwidth, 0.04]))
                c = "#ffff00" if self.labels[n] == self.label_default else "#00ff00"
                self.btn_eline.append(Button(self.ax_btn_eline[-1], self.labels[n], color=c, hovercolor="#ff0000"))

            # Set events to each button
            for b in self.btn_eline:
                b.on_clicked(self.btn_stack_clicked)

        def create_buttons_prev_next(self):

            bwidth = 0.07  # Width of prev. and next button button
            lwidth = 0.08  # Width of the text label between prev and next buttons
            bgap = 0.01  # Gap between buttons

            self.ax_btn_prev = plt.axes([0.5 - lwidth / 2 - bwidth - bgap, 0.03, bwidth, 0.04])
            self.btn_prev = Button(self.ax_btn_prev, "Previous", color="#00ff00", hovercolor="#ff0000")
            self.btn_prev.on_clicked(self.btn_stack_prev_clicked)

            self.ax_text_nlabel = plt.axes([0.5 - lwidth / 2, 0.03, lwidth, 0.04])
            self.textbox_nlabel = Button(self.ax_text_nlabel, "", color="lightgray", hovercolor="lightgray")
            self.display_btn_stack_label()

            self.ax_btn_next = plt.axes([0.5 + lwidth / 2 + bgap, 0.03, bwidth, 0.04])
            self.btn_next = Button(self.ax_btn_next, "Next", color="#00ff00", hovercolor="#ff0000")
            self.btn_next.on_clicked(self.btn_stack_next_clicked)

        def display_btn_stack_label(self):
            if self.textbox_nlabel:
                self.textbox_nlabel.label.set_text(
                    f"{self.labels.index(self.label_selected) + 1} ({len(self.labels)})"
                )

        def set_slider_energy_title(self):
            n = self.n_energy_selected
            n_total = len(self.energy)
            e = self.energy[n]
            self.ax_slider_energy.set_title(f"{e:.5f} keV (frame #{n + 1} of {n_total})")

        def btn_stack_clicked(self, event):
            """Callback"""
            for n, ab in enumerate(self.ax_btn_eline):
                if event.inaxes is ab:
                    self.switch_to_different_stack(self.labels[n])
                    break

        def btn_stack_next_clicked(self, event):
            """Callback"""
            n_labels = len(self.labels)
            n_current = self.labels.index(self.label_selected)
            if n_current < n_labels - 1:
                self.switch_to_different_stack(self.labels[n_current + 1])

        def btn_stack_prev_clicked(self, event):
            """Callback"""
            n_current = self.labels.index(self.label_selected)
            if n_current > 0:
                self.switch_to_different_stack(self.labels[n_current - 1])

        def btn_save_spectrum_clicked(self, event):
            # Create file name for the CSV file
            x_min, y_min, x_max, y_max = self.pts_selected
            x_str = f"_x_{x_min}" if (x_min == x_max) else f"_x_{x_min}_{x_max}"
            y_str = f"_y_{y_min}" if (y_min == y_max) else f"_y_{y_min}_{y_max}"
            fln = f"spectrum_{self.label_selected}{x_str}{y_str}.csv"

            # Compute the positional coordinates of the selected area
            pt_x_min = x_min * self.pos_dx + self.pos_x_min
            pt_y_min = y_min * self.pos_dy + self.pos_y_min
            pt_x_max = x_max * self.pos_dx + self.pos_x_min
            pt_y_max = y_max * self.pos_dy + self.pos_y_min
            # This message will the placed in the first line of the csv file
            msg_info = (
                f"Selection - x: {x_min} .. {x_max} ({pt_x_min:.5g} .. {pt_x_max:.5g})   "
                f"y: {y_min} .. {y_max} ({pt_y_min:.5g} .. {pt_y_max:.5g})"
            )

            _save_spectrum_as_csv(
                fln=fln, wd=wd, msg_info=msg_info, energy=self.energy, spectrum=self.xanes_spectrum_selected
            )

        def switch_to_different_stack(self, label):
            self.label_selected = label
            self.select_stack()
            self.redraw_image()
            self.set_cbar_range()
            self.display_btn_stack_label()
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

        def redraw_fluorescence_plot(self):
            self.ax_fluor_plot.clear()

            xd_px_min, yd_px_min, xd_px_max, yd_px_max = self.pts_selected

            if xd_px_min == xd_px_max and yd_px_min == yd_px_max:
                plot_single_point = True
            else:
                plot_single_point = False

            data_stack_selected = self.stack_selected[:, yd_px_min : yd_px_max + 1, xd_px_min : xd_px_max + 1]
            data_selected = np.sum(np.sum(data_stack_selected, axis=2), axis=1)

            # Apply energy shift (it influences all plots, so it should be done regardless of the rest
            #   of the options
            if self.incident_energy_shift_keV == self.incident_energy_shift_keV_updated:
                self.energy = self.energy_original
            else:
                # Find difference between the original value of the shift (that was already
                #   applied and the new value)
                de = self.incident_energy_shift_keV_updated - self.incident_energy_shift_keV
                # Apply the shift to energy points for the observed data
                self.energy = [_ + de for _ in self.energy_original]

            # Subtract baseline if needed
            if (self.label_selected == self.label_default) and self.subtract_pre_edge_baseline:
                try:
                    data_selected = subtract_xanes_pre_edge_baseline(
                        data_selected, self.energy, self.label_default, non_negative=False
                    )
                except RuntimeError as ex:
                    logger.error(f"Background was not subtracted: {ex}")

            # Fluorescence plot will display the new energy values
            self.ax_fluor_plot.plot(
                self.energy, data_selected, marker=".", linestyle="solid", label="XANES spectrum"
            )

            # Save currently displayed spectrum data, so that it could be reused (e.g. saved to file)
            self.xanes_spectrum_selected = data_selected

            # Plot the results of fitting (if the fitting was performed and the value for the
            #    shift of incident energy was not changed)
            if (
                (self.label_selected == self.label_default)
                and (self.xanes_map_data is not None)
                and (self.absorption_refs is not None)
                and (self.ref_data is not None)
                and (self.ref_energy is not None)
            ):

                # The number of averaged points
                n_averaged_pts = (xd_px_max - xd_px_min + 1) * (yd_px_max - yd_px_min + 1)

                plot_comments = ""
                if n_averaged_pts > 1:
                    plot_comments = f"Area: {n_averaged_pts} px."

                fit_real_time = True
                if plot_single_point and (
                    self.incident_energy_shift_keV == self.incident_energy_shift_keV_updated
                ):
                    # fit_real_time = False
                    fit_real_time = True  # For now let's always do fitting on the fly
                if fit_real_time:
                    # Interpolate references (same function as in the main processing routine)
                    refs = _interpolate_references(self.energy, self.ref_energy, self.ref_data)
                    data_nn = np.clip(data_selected, a_min=0, a_max=None)
                    xanes_fit_pt, xanes_fit_rfactor, _ = fit_spectrum(
                        data_nn, refs, method=self.fitting_method, rate=self.fitting_descent_rate
                    )
                else:
                    # Use precomputed data
                    xanes_fit_pt = self.xanes_map_data[:, yd_px_min, xd_px_min]
                    # We still compute R-factor
                    xanes_fit_rfactor = rfactor_compute(data_selected, xanes_fit_pt, self.absorption_refs)

                # Compute fit results represented in counts (for output)
                xanes_fit_pt_counts = xanes_fit_pt.copy()
                for n, v in enumerate(xanes_fit_pt):
                    xanes_fit_pt_counts[n] = v * np.sum(self.absorption_refs[:, n])

                for n, v in enumerate(xanes_fit_pt_counts):
                    plot_comments += f"\n{self.ref_labels[n]}: {xanes_fit_pt_counts[n]:.2f} c."
                plot_comments += f"\nR-factor: {xanes_fit_rfactor:.5g}"

                # Labels should always be supplied when calling the function.
                #   This is not user controlled option, therefore exception should be raised.
                assert self.labels, "No labels are provided. Fitting results can not be displayed properly"

                refs_scaled = self.absorption_refs.copy()
                _, n_refs = refs_scaled.shape
                for n in range(n_refs):
                    refs_scaled[:, n] = refs_scaled[:, n] * xanes_fit_pt[n]
                refs_scaled_sum = np.sum(refs_scaled, axis=1)

                self.ax_fluor_plot.plot(self.energy, refs_scaled_sum, label="XANES fit")
                for n in range(n_refs):
                    self.ax_fluor_plot.plot(
                        self.energy, refs_scaled[:, n], label=self.ref_labels[n], linestyle="dashed"
                    )

                # Plot notes, containing the number of averaged points, counts per reference
                #  and R-factor (only for the emission line used for XANES analysis)
                if plot_comments:
                    self.fluor_plot_comments = self.ax_fluor_plot.text(
                        0.99,
                        0.05,
                        plot_comments,
                        ha="right",
                        va="bottom",
                        fontsize=self.coord_fontsize,
                        transform=self.ax_fluor_plot.axes.transAxes,
                    )

            # Always display the legend
            self.ax_fluor_plot.legend(loc="upper left")

            self.ax_fluor_plot.grid()
            self.ax_fluor_plot.set_xlabel("Energy, keV", fontsize=self.label_fontsize)
            self.ax_fluor_plot.set_ylabel("Fluorescence", fontsize=self.label_fontsize)
            self.ax_fluor_plot.ticklabel_format(style="sci", scilimits=(-3, 4), axis="y")
            self.show_fluor_point_coordinates()

        def redraw_image(self):
            self.img_selected = self.stack_selected[self.n_energy_selected, :, :]
            self.img_plot.set_data(self.img_selected)
            self.img_label_text.set_text(self.label_selected)
            self.redraw_fluorescence_plot()

        def slider_update(self, val):
            if not self.busy:
                self.busy = True
                n_slider = int(round(val))
                if n_slider != self.n_energy_selected:
                    self.n_energy_selected = n_slider
                    self.set_slider_energy_title()
                    self.redraw_image()
                    self.fig.canvas.draw()
                    self.fig.canvas.flush_events()
                self.busy = False

        def tb_energy_shift_onsubmit(self, event):
            # 'event' is just a string that contains the number
            new_val_valid = False
            try:
                new_val = float(event)
                self.incident_energy_shift_keV_updated = new_val
                new_val_valid = True
            except Exception:
                pass
            self.tb_energy_shift.set_val(f"{self.incident_energy_shift_keV_updated}")
            if new_val_valid:
                if self.incident_energy_shift_keV == self.incident_energy_shift_keV_updated:
                    self.tb_energy_shift.color = "#00ff00"
                    self.tb_energy_shift.hovercolor = "#00ff00"
                else:
                    self.tb_energy_shift.color = "#ffff00"
                    self.tb_energy_shift.hovercolor = "#ffff00"
                # The following line actually updates the color, not just sets the attribute
                self.tb_energy_shift.ax.set_facecolor(self.tb_energy_shift.color)
                self.redraw_fluorescence_plot()
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()

        def canvas_onpress(self, event):
            """Callback, mouse button pressed"""
            self.button_pressed = False  # May be pressed outside the region
            if (self.t_bar.mode == "") and (event.inaxes == self.ax_img_stack) and (event.button == 1):
                self.button_pressed = True  # Left mouse button is in pressed state
                #     and it was pressed when the cursor was inside the plot area
                xd, yd = event.xdata, event.ydata
                # Compute pixel coordinates
                xd_px = round((xd - self.pos_x_min) / self.pos_dx)
                yd_px = round((yd - self.pos_y_min) / self.pos_dy)
                self.start_sel = (int(xd_px), int(yd_px))

        def canvas_onrelease(self, event):
            """Callback, mouse button released"""
            # self.t_bar.mode == "" is True if no toolbar items are activated (zoom, pan etc.)
            if (
                self.button_pressed
                and (self.t_bar.mode == "")
                and (event.inaxes == self.ax_img_stack)
                and (event.button == 1)
            ):
                xd, yd = event.xdata, event.ydata
                # Pixel coordinates (button pressed)
                xd_px_min = self.start_sel[0]
                yd_px_min = self.start_sel[1]
                # Compute pixel coordinates (button release)
                xd_px_max = int(round((xd - self.pos_x_min) / self.pos_dx))
                yd_px_max = int(round((yd - self.pos_y_min) / self.pos_dy))
                # Sort coordinates (top-left goes first)
                xd_px_min, xd_px_max = min(xd_px_min, xd_px_max), max(xd_px_min, xd_px_max)
                yd_px_min, yd_px_max = min(yd_px_min, yd_px_max), max(yd_px_min, yd_px_max)
                self.pts_selected = [xd_px_min, yd_px_min, xd_px_max, yd_px_max]
                self.draw_selection_mark()
                self.redraw_fluorescence_plot()
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()

            self.button_pressed = False  # Left mouse button is released
            #                               (it doesn't matter where the cursor is)

    map_plot = EnergyMapPlot(
        energy=energies,
        stack_all_data=eline_data,
        label_default=eline_selected,
        positions_x=positions_x,
        positions_y=positions_y,
        axes_units=axes_units,
        xanes_map_data=xanes_map_data,
        scan_absorption_refs=scan_absorption_refs,
        ref_labels=ref_labels,
        ref_energy=ref_energy,
        ref_data=ref_data,
        fitting_method=fitting_method,
        fitting_descent_rate=fitting_descent_rate,
        energy_shift_keV=energy_shift_keV,
        subtract_pre_edge_baseline=subtract_pre_edge_baseline,
        wd=wd,
    )
    map_plot.show(block=True)


def plot_xanes_map(
    map_data, *, label=None, block=True, positions_x=None, positions_y=None, axes_units=None, map_margin=0
):
    r"""
    Plot XANES map

    Parameters
    ----------

    map_data : ndarray, 2D
        map to be plotted, dimensions: (Ny, Nx)

    label : str
        reference label that is included in the image title and window title

    block : bool
        the parameter is passed to ``plt.show()``. Indicates if the execution of the
        program should be paused until the figure is closed.

    positions_x : ndarray
        values of coordinate X of the image. Only the first and the last value of the list
        is used to set the axis range, so [xmin, xmax] is sufficient.

    positions_y : ndarray
        values of coordinate Y of the image. Only the first and the last value of the list
        is used to set the axis range, so [ymin, ymax] is sufficient.

    axes_units : str
        units for X and Y axes that are used if ``positions_x`` and ``positions_y`` are specified
        Units may include latex expressions, for example "$\mu $m" will print units of microns.
        If ``axes_units`` is None, then axes labels will be printed without unit information.

    map_margin : float
        width of the map margin in percent. The pixels that fall within the margin are not used
        to compute the range of the colorbar. Setting margin different from zero may be important
        when displaying residual data (R-factor) for XANES maps, because error values for pixels
        close to edges of the map may be very large. Typically the value 10% should be fine.

    Returns
    -------
        Reference to the figure containing the plot

    """

    # Check existence or the size of 'positions_x' and 'positions_y' arrays
    ny, nx = map_data.shape
    if (positions_x is None) or (positions_y is None):
        positions_x = range(nx)
        positions_y = range(ny)
        axes_units = "pixels"
    else:
        axes_units = axes_units if axes_units else ""

    x_label = f"X, {axes_units}" if axes_units else "X"
    y_label = f"Y, {axes_units}" if axes_units else "Y"

    # Find max and min values. The margins are likely to contain strong artifacts that distort images.
    c = max(map_margin / 100.0, 0)  # Make sure it is positive
    x_margin, y_margin = int(nx * c), int(ny * c)
    vmin = np.min(map_data[y_margin : ny - y_margin, x_margin : nx - x_margin])
    vmax = np.max(map_data[y_margin : ny - y_margin, x_margin : nx - x_margin])

    # Element label may be LaTex expression. Remove '$' and '_' from it before using it
    #   in the figure title, since LaTeX it is not rendered in the figure title.
    label_fig_title = label.replace("$", "")
    label_fig_title = label_fig_title.replace("_", "")

    if label:
        fig_title = f"XANES map: {label_fig_title}"
        img_title = f"XANES map: {label}"

    fig = plt.figure(figsize=(6, 6), num=fig_title)
    ax = fig.subplots()

    # display image
    extent = [positions_x[0], positions_x[-1], positions_y[-1], positions_y[0]]
    img_plot = ax.imshow(map_data, vmin=vmin, vmax=vmax, origin="upper", extent=extent)
    plt.colorbar(img_plot, orientation="vertical")
    ax.get_xaxis().set_label_text(x_label, fontsize=15)
    ax.get_yaxis().set_label_text(y_label, fontsize=15)
    fig.suptitle(img_title, fontsize=20)
    plt.show(block=block)
    return fig


def plot_absorption_references(
    *, ref_energy, ref_data, scan_energies, scan_absorption_refs, ref_labels=None, block=True
):
    r"""
    Plots absorption references

    Parameters
    ----------
    ref_energy : ndarray, 1D
        N elements, energy values for the original references (loaded from file)
    ref_data : ndarray, 2D
        NxK elements, K - the number of references
    scan_energies : ndarray, 1D
        M elements, energy values for resampled references, M - the number of scanned images
    scan_absorption_refs : ndarray, 2D
        MxK elements
    ref_labels : list(str)
        K elements, the labels for references. If None or empty, then no labels will be
        printed
    block : bool
        the parameter is passed to ``plt.show()``. Indicates if the execution of the
        program should be paused until the figure is closed.

    Returns
    -------
        Reference to the figure containing the plot
    """

    # Check if the number of original and sampled references match
    _, n_refs = ref_data.shape
    _, n2 = scan_absorption_refs.shape
    assert n_refs == n2, "The number of original and sample references does not match"
    if ref_labels:
        assert n_refs == len(ref_labels), "The number of references and labels does not match"

    if ref_labels:
        labels = ref_labels
    else:
        labels = [""] * n_refs

    fig = plt.figure(figsize=(6, 6), num="Element References")
    ax = fig.subplots()

    for n in range(n_refs):
        ax.plot(ref_energy, ref_data[:, n], label=labels[n])

    for n in range(n_refs):
        ax.plot(scan_energies, scan_absorption_refs[:, n], "o", label=labels[n])

    ax.set_xlabel("Energy, keV", fontsize=15)
    ax.set_ylabel("Absorption", fontsize=15)
    plt.grid(True)
    fig.suptitle("Element References", fontsize=20)
    if ref_labels:
        plt.legend(loc="upper right")
    plt.show(block=block)
    return fig


def _get_img_data(img_dict, key, detector=None):
    r"""
    Retrieves entry of ``img_dict``. The difficulty is that the dataset key (first key)
    contains file name so it is variable, so at the first step the appropriate key has to
    be found. Then the data for the element line or scaler (specified by ``key``)
    may be retrieved.

    Parameters
    ----------

    img_dict : dict
        The dictionary of data extracted from HDF5 file. Processed data for element lines
        may be accessed as ``img_dict[dataset_name][key]

    key : str
        The string that specifies element line or a scaler

    detector : int
        detector channel number: 1, 2, 3 etc. Set to None if the sum of all channels
        data should be used.
    """
    return img_dict[_get_dataset_name(img_dict, detector=detector)][key]


def _get_img_keys(img_dict, detector=None):
    """
    Returns list of keys in the dataset. See docstring for ``_get_img_data`` for more information.
    """
    return list(img_dict[_get_dataset_name(img_dict, detector=detector)].keys())


def _get_eline_keys(key_list):
    r"""
    Returns list of emission line names that are present in the list of keys.
    The function checks if key matches the pattern for the emission line name,
    it does not check if the name is valid.

    The following names will match pattern: P_K, Ca_K, Fe_L, Fe_M etc.

    Parameters
    ----------

    key_list : list(str)
        list of keys (strings), some of the keys may be emission line names

    Returns
    -------
        list of emission line names extracted from the list of keys
    """

    # The following pattern is based on assumption that elines names
    #        are of the form Si_K, La_L, P_K etc.
    re_pattern = "^[A-Z][a-z]?_[KLM]$"
    return [key for key in key_list if re.search(re_pattern, key)]


def _get_dataset_name(img_dict, detector=None):
    r"""
    Finds the name of dataset with fitted data in the ``img_dict``.
    Dataset name contains file name in it, so it is changing from file to file.
    The dataset ends with suffix '_fit'. Datasets for individual detector
    channels end with 'det1_fit', 'det2_fit' etc.

    Parameters
    ----------

    img_dict : dict
        dictionary of xrf image data loaded from .h5 file
    detector : int
        detector channel number: 1, 2, 3 etc. Set to None if the sum of all channels
        data should be used.

    Returns
    -------
    dataset name, raises RuntimeError exception if dataset is not found
    """
    for name in img_dict.keys():

        if detector is None:
            # Dataset name for the sum should have no 'det1', 'det2' etc. preceding '_fit'
            if re.search("fit$", name) and not re.search(r"det\d+_fit", name):  # noqa: W605
                return name
        else:
            if re.search(f"det{detector}_fit$", name):
                return name

    raise RuntimeError(f"No dataset name was found for the detector {detector} ('get_dataset_name').")


def read_ref_data(ref_file_name):
    r"""
    Read reference data from CSV file (for XANES fitting). The first column of the file must
    contain incident energy values. The first line of the file must contain labels for each column.
    The labels must be comma separated. If labels contain separator ',', they must be enclosed in
    double quotes '"'. Note, that spreadsheet programs tend to use other characters as opening
    and closing quotes. Those characters are not treated correctly by the CSV reading software.

    The example of the CSV file with LaTeX formatted labels
    Energy,$Fe_3P$,$LiFePO_4$,$Fe_2O_3$
    7061.9933,0.0365,0.0235,0.0153
    7063.9925,0.0201,0.0121,0.00378
    7065.9994,0.0181,0.0111,0.00327
    7067.9994,0.0161,0.0101,0.00238
    7070.0038,0.0144,0.00949,0.00233
    ... etc. ...

    Parameters
    ----------

    ref_file_name : str
        path to CSV file that contain element references


    Returns
    -------

    ref_energy : ndarray, 1D
         array of N energy values

    ref_data : ndarray, 2D
        array of (N, M) values, which specify references for M elements at N energy values

    ref_labels : list
        list of M reference names
    """

    if not os.path.isfile(ref_file_name):
        raise ValueError(
            f"The parameter file '{ref_file_name}' does not exist. Check the value of"
            f" the parameer 'ref_file_name' ('_build_xanes_map_api')."
        )
    # Read the header (first line) using 'csv' (delimiter ',' and quotechar '"')
    with open(ref_file_name, "r") as csv_file:
        for row in csv.reader(csv_file):
            ref_labels = [_ for _ in row]
            break

        ref_labels.pop(0)

    # The rest of the file may be loaded as ndarray
    data_ref_file = np.genfromtxt(ref_file_name, skip_header=1, delimiter=",")
    ref_energy = data_ref_file[:, 0]
    # The references are columns 'ref_data'
    ref_data = data_ref_file[:, 1:]

    _, n_ref_data_columns = ref_data.shape

    if n_ref_data_columns != len(ref_labels):
        raise Exception(f"Reference data file '{ref_file_name}' has unequal number of labels and data columns")

    # Scale the energy. Scaling is based on assumption that if energy is > 100, it is given in eV,
    #   otherwise it is given in keV.
    if sum(ref_energy > 100.0):
        ref_energy /= 1000.0
    # Data may contain some small negative values, so clip them to 0
    ref_data = ref_data.clip(min=0)

    return ref_energy, ref_data, ref_labels


def _generate_output_dir_name(*, wd, dir_suffix=None, base_dir_name="nanoXANES_Analysis_"):
    """
    Generate output directory name. If ``dir_suffix`` is None, then unique directory name
    is generated.
    """
    res_dir_base = os.path.join(wd, base_dir_name)
    if dir_suffix is not None:
        dir_suffix = dir_suffix.strip()  # Remove leading and trailing spaces
    if dir_suffix:
        # Simply append the suffix. Data in the directory will be overwritten
        res_dir = f"{res_dir_base}{dir_suffix}"
    else:
        # Automatically generate the suffix for the output directory name.
        # The autogenerated directory name will always be unique.
        for n in range(1, sys.maxsize):
            res_dir = f"{res_dir_base}v{n}"
            if not os.path.exists(res_dir):
                break
    return res_dir


def _save_xanes_maps_to_tiff(
    *,
    wd,
    eline_data_aligned,
    eline_selected,
    xanes_map_data,
    xanes_map_rfactor,
    xanes_map_labels,
    scan_energies,
    scan_energies_shifted,
    scan_ids,
    files_h5,
    positions_x,
    positions_y,
    output_save_all,
):

    r"""
    Saves the results of processing in stacked .tiff files and creates .txt log file
    with the list of contents of .tiff files.
    It is assumed that the input data is consistent.

    If ``xanes_map_data`` or ``xanes_map_labels`` are None, then XANES maps are not
    saved. XANES maps are not generated if file with references is not available.
    This is one of the legitimate modes of operation.

    Parameters
    ----------

    wd : str
        working directory where the output data files are saved

    eline_data_aligned : dict(ndarray)
        The dictionary that contains aligned datasets. Key: emission line,
        value: ndarray [K, Ny, Nx], K - the number of scans, Ny and Nx - the
        number of pixels along y- and x- axes.

    eline_selected : str
        The name of the selected emission line. If the emission line is not present
        in the dictionary ``eline_data_aligned``, then the image stack is not saved.

    xanes_map_data : ndarray [M, Ny, Nx]
        XANES maps. M - the number of maps

    xanes_map_rfactor : ndarray [Ny, Nx]
        R-factor for XANES maps.

    xanes_map_labels : list(str)
        Labels for XANES maps. The number of labels must be equal to M.

    scan_energies : list(float)
        Beam energy values for the scans. The number of values must be K.

    scan_energies_shifted : list(float)
        Beam energy values with applied correction shift. The number of values must be K.

    scan_ids : list(int)
        Scan IDs of the scans. There must be K scan IDs.

    files_h5 : list(str)
        list of names of the files that contain XRF scan data. The number of values must be K.

    positions_x : 1D ndarray
        vector of coordinates along X-axis, used to determine range and the number
        of scan points

    positions_y : 1D ndarray
        vector of coordinates along Y-axis, used to determine range and the number
        of scan points

    output_save_all : bool
        indicates if processing results for every emission line, which is selected for XRF
        fitting, are saved. If set to ``True``, the (aligned) stacks of XRF maps are saved
        for each element. If set to False, then only the stack for the emission line
        selected for XANES fitting is saved (argument ``emission_line``).
    """

    if eline_selected is None:
        eline_selected = ""

    # A .txt file is created along with saving the rest of the data.
    fln_log = "maps_log_tiff.txt"
    fln_log = os.path.join(wd, fln_log)
    with open(fln_log, "w") as f_log:

        print(f"Processing completed at {convert_time_to_nexus_string(ttime.localtime())}", file=f_log)

        if positions_x is not None and positions_y is not None:
            n_x_pixels = len(positions_x)
            n_y_pixels = len(positions_y)
            x_min, x_max = positions_x[0], positions_x[-1]
            y_min, y_max = positions_y[0], positions_y[-1]
            print("\nXANES scan parameters:", file=f_log)
            print(f"    image size (Ny, Nx): ({n_y_pixels}, {n_x_pixels})", file=f_log)
            print(
                f"    Y-axis scan range [Y_min, Y_max, abs(Y_max-Y_min)]: "
                f"[{y_min:.5g}, {y_max:.5g}, {abs(y_max-y_min):.5g}]",
                file=f_log,
            )
            print(
                f"    X-axis scan range [X_min, X_max, abs(X_max-X_min)]: "
                f"[{x_min:.5g}, {x_max:.5g}, {abs(x_max-x_min):.5g}]",
                file=f_log,
            )

        if eline_data_aligned and eline_selected and (eline_selected in eline_data_aligned):
            # Save the stack of XRF maps for the selected emission line
            fln_stack = f"maps_XRF_{eline_selected}.tiff"
            fln_stack = os.path.join(wd, fln_stack)
            tifffile.imsave(fln_stack, eline_data_aligned[eline_selected].astype(np.float32), imagej=True)
            fln_stack_sum = f"SUM_maps_XRF_{eline_selected}.tiff"
            fln_stack_sum = os.path.join(wd, fln_stack_sum)
            # Find sum of the stack
            stack_sum = sum(eline_data_aligned[eline_selected], 0)
            tifffile.imsave(fln_stack_sum, stack_sum.astype(np.float32), imagej=True)

            logger.info(
                f"The stack of XRF maps for the emission line {eline_selected} is saved to file '{fln_stack}'"
            )

            # Save the contents of the .tiff file to .txt file
            print(f"\nThe stack of XRF maps is saved to file '{fln_stack}'.", file=f_log)
            print("Included maps:", file=f_log)
            if scan_energies and scan_energies_shifted and scan_ids and files_h5:
                for n, energy, energy_shifted, scan_id, fln in zip(
                    range(len(scan_energies)), scan_energies, scan_energies_shifted, scan_ids, files_h5
                ):
                    print(
                        f"   Frame {n + 1}:  scan ID = {scan_id}   "
                        f"incident energy = {energy:.4f} keV (corrected to {energy_shifted:.4f} keV) "
                        f"file name = '{fln}'",
                        file=f_log,
                    )
            print(f"\nThe stack sum is saved to file '{fln_stack_sum}'.", file=f_log)

        if output_save_all:
            for eline in eline_data_aligned.keys():
                # The data for the emission line used for xanes fitting is already saved
                if eline == eline_selected:
                    continue
                fln_stack = f"maps_XRF_{eline}.tiff"
                fln_stack = os.path.join(wd, fln_stack)
                logger.info(f"Saving XRF map stack for the emission line {eline}")
                print(f"XRF map stack for the emission line {eline} is saved to file '{fln_stack}'", file=f_log)
                tifffile.imsave(fln_stack, eline_data_aligned[eline].astype(np.float32), imagej=True)
                fln_stack_sum = f"SUM_maps_XRF_{eline}.tiff"
                fln_stack_sum = os.path.join(wd, fln_stack_sum)
                # Find sum of the stack
                stack_sum = sum(eline_data_aligned[eline], 0)
                tifffile.imsave(fln_stack_sum, stack_sum.astype(np.float32), imagej=True)

        if (xanes_map_data is not None) and xanes_map_labels and eline_selected:
            # Save XANES maps for references
            fln_xanes = f"maps_XANES_{eline_selected}.tiff"
            fln_xanes = os.path.join(wd, fln_xanes)
            tifffile.imsave(fln_xanes, xanes_map_data.astype(np.float32), imagej=True)
            logger.info(f"XANES maps for the emission line {eline_selected} are saved to file '{fln_xanes}'")

            # Save the contents of the .tiff file to .txt file
            print(f"\nXANES maps are saved to file '{fln_xanes}'.", file=f_log)
            print("Included maps:", file=f_log)
            if xanes_map_labels:
                for n, label in enumerate(xanes_map_labels):
                    print(f"   Frame {n + 1}:  reference = '{xanes_map_labels[n]}'", file=f_log)

        if xanes_map_rfactor is not None:
            # Save XANES maps for references
            fln_xanes_rfactor = f"maps_XANES_{eline_selected}_rfactor.tiff"
            fln_xanes_rfactor = os.path.join(wd, fln_xanes_rfactor)
            tifffile.imsave(fln_xanes_rfactor, xanes_map_rfactor.astype(np.float32), imagej=True)
            logger.info(
                f"R-factors for XANES maps for the emission line {eline_selected} are saved "
                f"to file '{fln_xanes_rfactor}'"
            )

            # Save the contents of the .tiff file to .txt file
            print(f"\nR-factors for XANES maps are saved to file '{fln_xanes_rfactor}'.", file=f_log)


def _save_spectrum_as_csv(*, fln, wd=None, msg_info=None, energy=None, spectrum=None):
    """
    Saves spectrum as CSV file.

    Parameters
    ----------

    fln: str
        name of the file to save spectrum

    wd: str
        working directory where the file ``file`` is to be placed

    msg_info: str
        A string that will be placed at the beginning of the CSV file before the file header.

    energy: ndarray(float)
        The array of energy values in keV

    spectrum: ndarray(float)
        The array of spectrum values. Must have the same size as 'energy'.
    """

    try:
        if wd:
            # Full file path
            wd = os.path.expanduser(wd)
            file_path = os.path.join(wd, fln) if wd else fln
            # Create directory (just in case it doesn't exist)
            os.makedirs(wd, exist_ok=True)
        else:
            # 'wd' is not set, so write to current directory
            file_path = fln

        # Absolute path is more informative in the user input
        file_path = os.path.abspath(file_path)

        # Verify that the arrays are valid and provide user-friendly message
        if energy is None:
            raise ValueError("The array 'energy' is None")
        if spectrum is None:
            raise ValueError("The array 'spectrum' is None")
        if len(energy) != len(spectrum):
            raise ValueError(
                f"Arrays 'energy' and 'spectrum' have different size: "
                f"len(energy)={len(energy)} len(spectrum)={len(spectrum)}"
            )
        # Create 2D array with 'energy' and 'spectrum' as columns
        energy = np.asarray(energy)
        spectrum = np.asarray(spectrum)
        data = np.transpose(np.vstack((energy, spectrum)))
        data = pd.DataFrame(data, columns=["Incident Energy, keV", "XANES spectrum"])
        # Format as 'CSV'
        str = data.to_csv(index=False)
        # Attach the comment to the beginning of the formatted string
        #   If 'msg_info' is '' or None, then don't add the comment
        if msg_info:
            # Add '# ' at the beginning of each line
            msg_info = "\n".join([f"# {_}" for _ in msg_info.split("\n")])
            str = f"{msg_info}\n{str}"
        # Save to file
        with open(file_path, "w") as f:
            f.write(str)
    except Exception as ex:
        logger.error(f"Selected spectrum can not be saved: {ex}")
    else:
        logger.info(f"Selected spectrum was saved to file '{file_path}'")


if __name__ == "__main__":

    logger = logging.getLogger(__name__)

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(fmt="%(asctime)s : %(levelname)s : %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)

    build_xanes_map(
        start_id=92276,
        end_id=92335,
        xrf_fitting_param_fln="param_335",
        scaler_name="sclr1_ch4",
        wd=None,
        # sequence="process",
        sequence="build_xanes_map",
        alignment_starts_from="top",
        ref_file_name="refs_Fe_P23.csv",
        fitting_method="nnls",
        output_save_all=True,
        results_dir_suffix="v2",
        # emission_line="Fe_K", emission_line_alignment="P_K",
        emission_line="Fe_K",
        emission_line_alignment="total_cnt",
        incident_energy_shift_keV=-0.0013,
        interpolation_enable=True,
        alignment_enable=True,
        normalize_alignment_stack=True,
        plot_use_position_coordinates=True,
        plot_results=True,
        allow_exceptions=True,
    )

    """
    build_xanes_map(parameter_file_path="xanes_parameters.yaml", output_save_all=True)
    """
