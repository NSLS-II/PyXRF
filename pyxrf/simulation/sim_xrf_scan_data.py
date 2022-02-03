import re
import numpy as np
import math
import os

from skbeam.core.fitting.xrf_model import K_LINE, L_LINE, M_LINE
from skbeam.core.constants.xrf import XrfElement
from skbeam.core.fitting.lineshapes import gaussian

from ..model.load_data_from_db import save_data_to_hdf5
from ..core.quant_analysis import ParamQuantEstimation

from ..core.xrf_utils import generate_eline_list

import logging

logger = logging.getLogger(__name__)


def _get_elemental_line_parameters(*, elemental_line, incident_energy):
    r"""
    Retrieve information on all emission lines for the given element and the group (K, L or M)
    at given incident energy. For each emission line, the information includes emission line name,
    energy and ratio. The data is used for simulation of emission spectra of elements.

    Parameters
    ----------

    elemental_line: str
        Elemental line name in the format ``Fe_K``, ``Ca_L``, etc.

    incident_energy: float
        Incident energy in keV

    Returns
    -------

    List of dictionaries. Keys: ``name`` - emission line name, ``energy`` - energy of the
    emission line, "ratio" - ratio of the emission line area to the area of ``a1`` line.

    Raises
    ------

    RuntimeError
        Elemental line is not in the list of supported lines or the emission line is
        incorrectly formatted.
    """

    # Check format (Fe_K, Y_L, W_M etc.)
    if not re.search(r"^[A-Z][a-z]?_[KLM]$", elemental_line):
        raise RuntimeError(f"Elemental line {elemental_line} has incorrect format")

    element, line = elemental_line.split("_")
    line_name = elemental_line

    ALL_LINES = K_LINE + L_LINE + M_LINE
    if line_name not in ALL_LINES:
        raise RuntimeError(f"Elemental line {line_name} is not supported")

    elemental_lines = []

    # XrfElement class provides convenient access to xraylib library functions
    e = XrfElement(element)

    # Check if the emission line is activated (check if 'a1' line is active)
    em_line_a1 = f"{line.lower()}a1"
    i_line_a1 = e.cs(incident_energy)[em_line_a1]
    if i_line_a1 > 0:
        for num, item in enumerate(e.emission_line.all):
            l_name = item[0]  # Line name (ka1, kb2 etc.)
            energy_v = item[1]  # Energy (in kEv)
            if line.lower() not in l_name:
                continue

            i_line = e.cs(incident_energy)[l_name]
            ratio_v = i_line / i_line_a1

            if energy_v == 0 or ratio_v == 0:
                continue

            elemental_lines.append({"name": l_name, "energy": energy_v, "ratio": ratio_v})

    return elemental_lines


def gen_xrf_spectrum(
    element_line_groups=None,
    *,
    incident_energy=12.0,
    n_spectrum_points=4096,
    e_offset=0.0,
    e_linear=0.01,
    e_quadratic=0.0,
    fwhm_offset=0.102333594,
    fwhm_fanoprime=0.000113169,
    epsilon=3.85,
):
    r"""
    Computes simulated XRF spectrum for the set of element line groups. Returns the spectrum
    as ndarray with 'n_spectrum_points' elements.

    Parameters
    ----------

    element_line_groups: dict(dict)
        Dictionary of element line groups that need to be included in the spectrum: key - element
        line group (``K``, ``L`` or ``M`` group suppoted by scikit beam, e.g. ``Si_K``, ``Ba_L``,
        ``Pt_M`` etc.); value - dictionary that contains spectrum parameters for the group.
        Currently only the parameter ``area`` is supported, which defines the area under the
        spectrum composed of all emission lines that belong to the group expressed in counts
        (must be positive floating point number).
        Example: ``{"Si_K": {'area' : 800}, "Ba_L": {'area' : 900}, "Pt_M": {'area' : 1000}}``

    incident_energy: float
        incident energy of the beam (used in simulation)

    n_spectrum_points: int
        the number of spectrum points. Currently PyXRF is working 4096-point spectra

    e_offset, e_linear, e_quadratic: float
        parameters used to compute energy values for the energy axis.
        The energy value #``nn`` is computed as ``e_offset + e_linear * nn + e_quadratic * np.square(nn)``,
        where ``nn = 0 .. n_spectrum_points - 1``. The default values should be typically used.

    fwhm_offset, fwhm_fanoprime, epsilon: float
        parameters theat determine the shape of the emission line peaks. The default values
        should be typically used.

    Returns
    -------

    spectrum_total: ndarray(float)
        The spectrum that contains active emission lines of the specified group.
        Size: 'n_spectrum_points'.

    xx_energy: ndarray(float)
        The values for the energy axis. Size: 'n_spectrum_points'.

    Raises
    ------

    RuntimeError
        Raised if the list of emission line groups contains incorrectly formatted or not supported
        emission lines. Also raised if ``n_spectrum_points`` is zero or negative.
    """

    if n_spectrum_points < 1:
        raise RuntimeError(f"Spectrum must contain at least one point (n_spectrum_points={n_spectrum_points})")

    if (element_line_groups is not None) and (not isinstance(element_line_groups, dict)):
        raise RuntimeError(
            f"Parameter 'element_line_groups' has invalid type {type(element_line_groups)} "
            f"(must be None or dict)"
        )

    spectrum_total = np.zeros((n_spectrum_points,), dtype="float")

    # Energy axis
    nn = np.asarray(range(n_spectrum_points))
    xx_energy = e_offset + e_linear * nn + e_quadratic * np.square(nn)

    if element_line_groups is not None:

        for element_line_group, parameters in element_line_groups.items():

            element_area = parameters["area"]

            spectrum = np.zeros((n_spectrum_points,), dtype="float")

            elemental_lines = _get_elemental_line_parameters(
                elemental_line=element_line_group, incident_energy=incident_energy
            )

            for line in elemental_lines:
                sigma = fwhm_offset / 2.0 / math.sqrt(2 * math.log(2))
                sigma = math.sqrt(sigma**2 + line["energy"] * epsilon * fwhm_fanoprime)
                spectrum += gaussian(x=xx_energy, area=line["ratio"], center=line["energy"], sigma=sigma)

            # Normalize the spectrum, make the area equal to 'element_area'
            spectrum *= element_area / spectrum.sum()

            spectrum_total += spectrum

    return spectrum_total, xx_energy


def gen_xrf_map_const(
    element_line_groups=None,
    *,
    nx=10,
    ny=5,
    incident_energy=12.0,
    n_spectrum_points=4096,
    background_area=0,
    spectrum_parameters=None,
):

    r"""
    Generate ny (vertical) by nx (horizontal) XRF map with identical spectrum for each pixel.

    element_line_groups: dict(dict)
        Dictionary of element line groups that need to be included in the spectrum: key - element
        line group (``K``, ``L`` or ``M`` group suppoted by scikit beam, e.g. ``Si_K``, ``Ba_L``,
        ``Pt_M`` etc.); value - dictionary that contains spectrum parameters for the group.
        Currently only the parameter ``area`` is supported, which defines the area under the
        spectrum composed of all emission lines that belong to the group expressed in counts
        (must be positive floating point number).
        Example: ``{"Si_K": {'area' : 800}, "Ba_L": {'area' : 900}, "Pt_M": {'area' : 1000}}``

    nx: int
        Horizontal dimension (axis 1) of the XRF map

    ny: int
        Vertical dimension (axis 0) of the XRF map

    incident_energy: float
        incident energy of the beam (used in simulation)

    n_spectrum_points: int
        the number of spectrum points. Currently PyXRF is working 4096-point spectra

    spectrum_parameters: dict
        dict of optional spectrum parameters, which is passed to ``gen_xrf_spectrum``.
        May be None.

    Returns
    -------

    xrf_map: ndarray(np.float32)
        XRF map with the shape ``(ny, nx, n_spectrum_points)``.
        Raw XRF spectra are represented with 32-bit precision.

    xx_energy: ndarray(float)
        The values for the energy axis. Size: 'n_spectrum_points'.

    Raises
    ------

    RuntimeError
        Raised if the list of emission line groups contains incorrectly formatted or not supported
        emission lines. Also raised if ``n_spectrum_points`` is zero or negative, or map with zero
        points is generated.
    """

    if spectrum_parameters is None:
        spectrum_parameters = {}

    if nx < 1 or ny < 1:
        raise RuntimeError(f"XRF map has zero pixels: nx={nx}, ny={ny}")

    spectrum, xx_energy = gen_xrf_spectrum(
        element_line_groups,
        incident_energy=incident_energy,
        n_spectrum_points=n_spectrum_points,
        **spectrum_parameters,
    )

    background = background_area / spectrum.size
    spectrum += background

    # One spectrum is computed. Now change precision to 32 bit before using it to create a map
    spectrum = np.float32(spectrum)

    xrf_map = np.broadcast_to(spectrum, shape=[ny, nx, len(spectrum)])

    return xrf_map, xx_energy


def create_xrf_map_data(
    *,
    scan_id,
    element_line_groups=None,
    num_det_channels=3,
    nx=10,
    ny=5,
    incident_energy=12.0,
    n_spectrum_points=4096,
    background_area=0,
    spectrum_parameters=None,
):
    r"""
    Generates a complete simulated XRF dataset based on set of element lines, XRF map size,
    incident energy etc. The dataset may be used for testing of XRF map processing functions.

    Parameters
    ----------

    scan_id: str or int

        Scan ID that is included in metadata of the generated dataset.

    element_line_groups: dict(dict)

        Dictionary of element line groups that need to be included in the spectrum: key - element
        line group (``K``, ``L`` or ``M`` group suppoted by scikit beam, e.g. ``Si_K``, ``Ba_L``,
        ``Pt_M`` etc.); value - dictionary that contains spectrum parameters for the group.
        Currently only the parameter ``area`` is supported, which defines the area under the
        spectrum composed of all emission lines that belong to the group expressed in counts
        (must be positive floating point number).
        Example: ``{"Si_K": {'area' : 800}, "Ba_L": {'area' : 900}, "Pt_M": {'area' : 1000}}``

    num_det_channels: int

        The number of detector channels to simulate. Must be integer greater than 1.

    nx: int

        Horizontal dimension (axis 1) of the XRF map

    ny: int

        Vertical dimension (axis 0) of the XRF map

    incident_energy: float

        incident energy of the beam (used in simulation)

    n_spectrum_points: int

        the number of spectrum points. Currently PyXRF is working 4096-point spectra

    background_area: float

        The area of the background. The background represents a rectangle, which occupies
        all ``n_spectrum_points``. If the generated spectrum is truncated later, the value
        of the area will change proportionally.

    spectrum_parameters: dict
        dict of optional spectrum parameters, which is passed to ``gen_xrf_map_const``.
        May be None.

    Returns
    -------

    data_xrf: dict(ndarray)
        The dictionary of the datasets. The dictionary keys are ``det_sum``, ``det1``, ``det2`` etc.
        The values are 3D arrays with ``shape = (ny, nx, n_spectrum_points)``.

    data_scalers: dict
        The dictionary with scaler information. The dictionary has two entries:
        ``data_scalers["scaler_names"]`` contains the list of scaler names (currently
        ``["i0", "time", "time_diff"]``). ``data_scalers["scaler_data"]`` contains 3D array
        with scaler data (``shape = (ny, nx, N_SCALERS)``, ``N_SCALERS`` is equal to the number
        of scaler names).

    data_pos: dict
        The dictionary of positional data: ``data_pos["pos_names"]=["x_pos", "y_pos"]``,
        ``data_pos["pos_data"]`` is 3D array with ``shape=(N_POS, ny, nx)``, where ``N_POS``
        is equal to the number of position names (currently 2). Note, that the ``y_pos``
        is measured along vertical dimension (axis 0 of the array) and ``x_pos`` is measured
        along horizontal dimension (axis 1 of the array).

    metadata: dict
        dictionary of metadata values, which include ``scan_id`` (passed to the function,
        ``scan_uid`` (randomly generated), and ``instrument_mono_incident_energy``
        (incident energy passed to the function).

    Raises
    ------

    RuntimeError
        Raised if the list of emission line groups contains incorrectly formatted or not supported
        emission lines. Also raised if ``n_spectrum_points`` is zero or negative, or map with zero
        points is generated (``nx`` or ``ny`` is 0).
    """

    if spectrum_parameters is None:
        spectrum_parameters = {}

    if num_det_channels < 1:
        num_det_channels = 1

    # Generate XRF map (sum of all channels)
    xrf_map, _ = gen_xrf_map_const(
        element_line_groups,
        nx=nx,
        ny=ny,
        incident_energy=incident_energy,
        n_spectrum_points=n_spectrum_points,
        background_area=background_area,
        **spectrum_parameters,
    )

    # Distribute total fluorescence into 'num_det_channels'
    channel_coef = np.arange(num_det_channels) + 10.0
    channel_coef /= np.sum(channel_coef)

    # Create datasets
    data_xrf = {}
    data_xrf["det_sum"] = xrf_map  # 'xrf_map' is np.float32
    for n in range(num_det_channels):
        data_xrf[f"det{n + 1}"] = xrf_map * channel_coef[n]  # The arrays remain np.float32

    data_scalers = {}
    scaler_names = ["i0", "time", "time_diff"]
    data_scalers["scaler_names"] = scaler_names

    # Scaler 'i0'
    scaler_data = np.zeros(shape=(ny, nx, len(scaler_names)), dtype=float)
    scaler_data[:, :, 0] = np.ones(shape=(ny, nx), dtype=float) * 0.1
    # Time
    time = np.arange(nx, dtype=float) * 2.0
    time = np.broadcast_to(time, shape=(ny, nx))
    scaler_data[:, :, 1] = time
    # Time difference
    scaler_data[:, :, 2] = np.ones(shape=(ny, nx), dtype=float) * 2.0
    data_scalers["scaler_data"] = scaler_data

    # Generate positions
    data_pos = {}
    data_pos["pos_names"] = ["x_pos", "y_pos"]
    x_pos_line = np.arange(nx) * 0.01 + 2.0
    x_pos = np.broadcast_to(x_pos_line, shape=(ny, nx))

    y_pos_column = np.arange(ny) * 0.02 + 1
    y_pos = np.broadcast_to(y_pos_column, shape=(nx, ny))
    y_pos = np.transpose(y_pos)

    data_pos["pos_data"] = np.zeros(shape=(2, ny, nx), dtype=float)
    data_pos["pos_data"][0, :, :] = x_pos
    data_pos["pos_data"][1, :, :] = y_pos

    # Generate metadata
    metadata = {}
    metadata["scan_id"] = scan_id

    def _gen_rs(n):
        uid_characters = list("abcdef0123456789")
        s = ""
        for i in range(n):
            s += np.random.choice(uid_characters)
        return s

    metadata["scan_uid"] = _gen_rs(8) + "-" + _gen_rs(4) + "-" + _gen_rs(4) + "-" + _gen_rs(4) + "-" + _gen_rs(12)
    metadata["instrument_mono_incident_energy"] = incident_energy

    return data_xrf, data_scalers, data_pos, metadata


def create_hdf5_xrf_map_const(
    *,
    scan_id,
    wd=None,
    fln_suffix=None,
    element_line_groups=None,
    save_det_sum=True,
    save_det_channels=True,
    num_det_channels=3,
    nx=10,
    ny=5,
    incident_energy=12.0,
    n_spectrum_points=4096,
    background_area=0,
    spectrum_parameters=None,
):
    r"""
    Generates and saves the simulated XRF map data to file. The file may be loaded in PyXRF
    and used for testing processing functions. The function overwrites existing files.

    Parameters
    ----------

    scan_id: str or int

        Scan ID (positive integer) that is included in the file metadata and used in file name.

    wd: str or None

        The directory where data is to be saved. If None, then current directory is used

    fln_suffix: str or None

        File name suffix, which is attached to file name (before extension). May be used
        to specify some additional information, useful for visual interpretation of file name.

    element_line_groups: dict(dict)

        Dictionary of element lines, see docstring for ``create_xrf_map_data`` for detailed
        information.

    save_det_sum: bool

        Indicates if the sum of detector channels should be saved (currently the sum is always
        saved, so False value is ignored)

    save_det_channels: bool

        Indicates if the individual detector channels are saved.

    num_det_channels: int

        The number of the detector channels. The area specified in ``element_line_groups`` is
        distributed between the detector channels.

    nx, ny: int

        The dimensions along vertical (``ny``, axis 0) and horizontal (``nx``, axis 1) axes.

    incident_energy: float

        Incident beam energy, used for dataset generation and also saved in metadata.

    n_spectrum_points: int

        The number of points in the spectrum

    background_area: float

        The area of the simulated background, see docstring for ``create_xrf_map_data``
        for detailed discussion.

    spectrum_parameters: dict
        dict of optional spectrum parameters, which is passed to ``create_xrf_map_data``.
        May be None.

    Returns
    -------

    fpath: str

        The path to the saved file.

    Raises
    ------

    RuntimeError
        Raised if the list of emission line groups contains incorrectly formatted or not supported
        emission lines. Also raised if ``n_spectrum_points`` is zero or negative, or map with zero
        points is generated (``nx`` or ``ny`` is 0).

    IOError may be raised in case of IO errors.
    """

    if not save_det_sum:
        logger.warning("The sum of the detector channels is always saved. ")

    # Prepare file name
    fln = f"scan2D_{scan_id}_sim"
    if fln_suffix:
        fln += f"_{fln_suffix}"
    fln += ".h5"

    if wd:
        wd = os.path.expanduser(wd)
        os.makedirs(wd, exist_ok=True)
        fpath = os.path.join(wd, fln)
    else:
        fpath = fln

    data_xrf, data_scalers, data_pos, metadata = create_xrf_map_data(
        scan_id=scan_id,
        element_line_groups=element_line_groups,
        num_det_channels=num_det_channels,
        nx=nx,
        ny=ny,
        incident_energy=incident_energy,
        n_spectrum_points=n_spectrum_points,
        background_area=background_area,
        spectrum_parameters=spectrum_parameters,
    )

    data = {}
    data.update(data_xrf)
    data.update(data_scalers)
    data.update(data_pos)

    save_data_to_hdf5(
        fpath, data, metadata=metadata, file_overwrite_existing=True, create_each_det=save_det_channels
    )

    return fpath


def gen_hdf5_qa_dataset(*, wd=None, standards_serials=None, test_elements=None):
    r"""
    Create a set of data files for testing quantitative analysis features.
    The following files are created:

    -- one simulated raw .h5 is created for each reference standard in the list
    ``standards_serials``. The file name is scan2D_<scanID>_sim_<serial>.h5,
    where ``scanID`` is integer that starts from 1000 and increments for each
    saved file and ``serial`` is the respective serial number of reference standard
    from the list. In order for the function to work, the descriptions of all used
    standards must exist in the built-in file ``xrf_quant_standards.yaml`` or
    in the file ``quantitative_standards.yaml`` in ``~/.pyxrf`` directory.

    -- one simulated raw .h5 file for the set of elements specified in ``test_elements``.
    Any elements may be included (both present and not present in calibration references).
    ``test_elements`` is a dictionary, with the key representing element name (such as Fe, Ca, K),
    and the value is the dictionary of element spectrum parameters. Currently the only
    required parameter is ``density``. The following code will create the dictionary with
    three elements (density is typically expressed in ug/cm^2):

    test_elements = {}
    test_elements["Fe"] = {"density": 50}
    test_elements["W"] = {"density": 70}
    test_elements["Au"] = {"density": 80}

    -- log file ``qa_files_log.txt`` with information on each saved file

    Parameters
    ----------

    wd: str (optional)

        Working directory, where the files are saved. Files are saved in local
        directory if ``wd`` is not specified.

    standards_serials: list(str)

        The list of serial numbers of standards. One simulated reference data file will be
        generated for each serial number. See the description above

    test_elements: dict(dict)

        The dictionary with parameters of element spectra for generation of the test files.
        See the description above.

    Returns
    -------

        The list of saved files.
    """

    if not standards_serials:
        raise RuntimeError(
            "There must be at least one standard loaded. Pass the list "
            "of standards as value of the parameter 'standard_list'"
        )

    nx, ny = 30, 20

    incident_energy = 13.0

    # For simplicity use the same emission intensity for all lines
    #   This should be the same value for reference and test files
    counts_per_unit = 10.0

    # If there are no test elements, then the test file is not generated
    if test_elements is None:
        test_elements = {}

    # Load standards
    param_quant_estimation = ParamQuantEstimation()
    param_quant_estimation.load_standards()

    element_lines = []
    lines_for_testing = {}

    scan_id = 1000  # Starting scan ID for reference scans
    # Go through the list of reference scans and save the data on present element lines
    #   in 'element_lines' list. If an exception is raised (one of the serials is not found)
    #   then no files are saved.
    for serial in standards_serials:
        standard = param_quant_estimation.find_standard(serial, key="serial")
        # ALL standards must exist
        if not standard:
            raise RuntimeError(f"Standard with serial #{serial} is not found.")

        param_quant_estimation.set_selected_standard(standard)
        param_quant_estimation.gen_fluorescence_data_dict(incident_energy=incident_energy)
        element_lines.append(param_quant_estimation.fluorescence_data_dict["element_lines"].copy())

    files_saved = []

    # Log file that contains brief data on every saved file
    fln_log = "qa_files_log.txt"
    if wd:
        wd = os.path.expanduser(wd)
        os.makedirs(wd, exist_ok=True)
        fln_log = os.path.join(wd, fln_log)

    with open(fln_log, "wt") as f_log:

        for serial, elines in zip(standards_serials, element_lines):
            el_grp = {}
            for line, info in elines.items():
                el_grp[line] = {"area": info["density"] * counts_per_unit}
                el = line.split("_")[0]  # Element: e.g. Fe_K -> Fe
                if line not in lines_for_testing:
                    if el in test_elements:
                        density = test_elements[el]["density"]
                        lines_for_testing[line] = {
                            "area": density * counts_per_unit,
                            "counts_per_unit": counts_per_unit,
                            "density": density,
                            "in_reference": True,
                        }

            fln = create_hdf5_xrf_map_const(
                scan_id=scan_id,
                wd=wd,
                fln_suffix=f"{serial}",
                element_line_groups=el_grp,
                nx=nx,
                ny=ny,
                incident_energy=incident_energy,
            )
            s = f"Reference standard file: '{fln}'\n    Standard serial: {serial}\n    Emission lines:\n"
            for line, info in elines.items():
                s += f"        {line}: density = {info['density']}\n"
            f_log.write(f"{s}\n")

            files_saved.append(fln)
            scan_id += 1

        test_elines = generate_eline_list(list(test_elements.keys()), incident_energy=incident_energy)
        for line in test_elines:
            if line not in lines_for_testing:
                el = line.split("_")[0]
                density = test_elements[el]["density"]
                lines_for_testing[line] = {
                    "area": density * counts_per_unit,
                    "counts_per_unit": counts_per_unit,
                    "density": density,
                    "in_reference": False,
                }

        fln_suffix = "test" + "_" + "_".join(test_elements.keys())
        fln = create_hdf5_xrf_map_const(
            scan_id=2000,
            wd=wd,
            fln_suffix=fln_suffix,
            element_line_groups=lines_for_testing,
            nx=nx,
            ny=ny,
            incident_energy=incident_energy,
        )
        s = f"Test file '{fln}'\n    Emission lines:\n"
        for line, info in lines_for_testing.items():
            s += (
                f"        {line}: density = {info['density']}, "
                f"counts_per_unit = {info['counts_per_unit']}, "
                f"area = {info['area']}, "
                f"in_reference = {info['in_reference']}\n"
            )
        f_log.write(f"{s}\n")

        files_saved.append(fln)
        files_saved.append(fln_log)

    return files_saved


def gen_hdf5_qa_dataset_preset_1(*, wd=None):
    r"""
    Generate a set of HDF5 files for testing of quantitative analysis procedures.
    The following files are created:
        calibration (reference) files:
            ``scan2D_1000_sim_41151.h5`` - based on standard with serial 41151
            ``scan2D_1001_sim_41163.h5`` - based on standard with serial 41163
        test file with elements Fe, W and Au:
            ``scan2D_2000_sim_test_Fe_W_Au.h5
    It is required that standards 41151 and 41163 are present in the list of
    quantitative standards (``xrf_quant_standards.yaml``).
    The test file contains the elements with the following densities:
        ``Fe``: 50 um/cm^2
        ``W``:  70 um/cm^2
        ``Au``: 80 um/cm^2
    The files contain scaler ``i0`` which could be used for normalization.
    Additionally, the log file ``qa_files_log.txt`` with information on the dataset is saved.

    Parameters
    ----------

    wd: str (optional)

        Working directory, where the files are saved. Files are saved in local
        directory if ``wd`` is not specified.
    """

    standards_serials = ["41151", "41163"]
    test_elements = {}
    test_elements["Fe"] = {"density": 50}  # Density in ug/cm^2 (for simulated test scan)
    test_elements["W"] = {"density": 70}
    test_elements["Au"] = {"density": 80}
    files_saved = gen_hdf5_qa_dataset(wd=wd, standards_serials=standards_serials, test_elements=test_elements)

    # Print saved file names
    f_names = [f"    '{_}'" for _ in files_saved]
    f_names = "\n".join(f_names)
    s = "Success. The following files were created:\n" + f_names
    logger.info(s)
