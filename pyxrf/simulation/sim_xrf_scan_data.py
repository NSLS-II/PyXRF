import re
import numpy as np
import math
import os

from skbeam.core.fitting.xrf_model import get_line_energy
from skbeam.core.fitting.xrf_model import (K_LINE, L_LINE, M_LINE, K_TRANSITIONS,
                                           L_TRANSITIONS, M_TRANSITIONS, TRANSITIONS_LOOKUP)
from skbeam.core.constants.xrf import XrfElement
from skbeam.core.fitting.lineshapes import gaussian

from ..model.load_data_from_db import write_db_to_hdf_base
from ..core.quant_analysis import ParamQuantEstimation

import logging
logger = logging.getLogger()


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

    element, line = elemental_line.split('_')
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
            l_name = item[0] # Line name (ka1, kb2 etc.)
            energy_v = item[1]  # Energy (in kEv)
            if line.lower() not in l_name:
                continue

            i_line = e.cs(incident_energy)[l_name]
            ratio_v = i_line/i_line_a1

            if energy_v == 0 or ratio_v == 0:
                continue

            elemental_lines.append({"name": l_name, "energy": energy_v, "ratio": ratio_v})

    return elemental_lines


def gen_xrf_spectrum(element_line_groups=None, *,
                     incident_energy=12.0,
                     n_spectrum_points=4096,
                     e_offset=0.0,
                     e_linear=0.01,
                     e_quadratic=0.0,
                     fwhm_offset=0.102333594,
                     fwhm_fanoprime=0.000113169,
                     epsilon=3.85):
    r"""
    Computes simulated XRF spectrum for the set of element line groups. Returns the spectrum
    as ndarray with 'n_spectrum_points' elements.

    Parameters
    ----------

    element_line_groups: list(tuple)
        List of element line groups that need to be included in the spectrum. Each element line
        group must be one of ``K``, ``L`` or ``M`` groups supported by ``scikit-beam``.
        Each group represented as a tuple that contains the group name (e.g. ``Fe_K``)
        and the dictionary of the group parameters (currently only parameter ``area`` is
        supported). The area is expressed in counts (must be positive floating point number).
        Example: ``[("Si_K", {'area' : 800}), ("Ba_L", {'area' : 900}), ("Pt_M", {'area' : 1000})]``

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
        raise RuntimeError(f"Spectrum must contain at least one point "
                           f"(n_spectrum_points={n_spectrum_points})")

    spectrum_total = np.zeros((n_spectrum_points,), dtype="float")

    # Energy axis
    nn = np.asarray(range(n_spectrum_points))
    xx_energy = e_offset + e_linear * nn + e_quadratic * np.square(nn)

    if element_line_groups is not None:

        for element_line_group, parameters in element_line_groups:

            element_area = parameters['area']

            spectrum = np.zeros((n_spectrum_points,), dtype="float")

            elemental_lines = _get_elemental_line_parameters(elemental_line=element_line_group,
                                                             incident_energy=incident_energy)

            for line in elemental_lines:
                sigma = fwhm_offset / 2.0 / math.sqrt(2 * math.log(2))
                sigma = math.sqrt(sigma ** 2 + line['energy'] * epsilon * fwhm_fanoprime)
                spectrum += gaussian(x=xx_energy, area=line['ratio'], center=line['energy'], sigma=sigma)

            # Normalize the spectrum, make the area equal to 'element_area'
            spectrum *= element_area / spectrum.sum()

            spectrum_total += spectrum

    return spectrum_total, xx_energy


def gen_xrf_map_const(element_line_groups=None, *,
                      nx=10, ny=5,
                      incident_energy=12.0,
                      n_spectrum_points=4096,
                      background_area=0,
                      spectrum_parameters=None):

    r"""
    Generate ny (vertical) by nx (horizontal) XRF map with identical spectrum for each pixel.

    Parameters
    ----------
    element_line_groups: list(tuple)
        List of element line groups that need to be included in the spectrum. Each element line
        group must be one of ``K``, ``L`` or ``M`` groups supported by ``scikit-beam``.
        Each group represented as a tuple that contains the group name (e.g. ``Fe_K``)
        and the dictionary of the group parameters (currently only parameter ``area`` is
        supported). The area is expressed in counts (must be positive floating point number).
        Example: ``[("Si_K", {'area' : 800}), ("Ba_L", {'area' : 900}), ("Pt_M", {'area' : 1000})]``

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

    xrf_map: ndarray(float)
        XRF map with the shape ``(ny, nx, n_spectrum_points)``.

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

    spectrum, xx_energy = gen_xrf_spectrum(element_line_groups,
                                           incident_energy=incident_energy,
                                           n_spectrum_points=n_spectrum_points,
                                           **spectrum_parameters)

    background = background_area / spectrum.size
    spectrum += background

    xrf_map = np.broadcast_to(spectrum, shape=[ny, nx, len(spectrum)])

    return xrf_map, xx_energy


def create_xrf_map_data(*, scan_id,
                        element_line_groups=None,
                        num_det_channels=3,
                        nx=10, ny=5,
                        incident_energy=12.0,
                        n_spectrum_points=4096,
                        background_area=0,
                        spectrum_parameters=None):

    if spectrum_parameters is None:
        spectrum_parameters = {}

    if num_det_channels < 1:
        num_det_channels = 1

    # Generate XRF map (sum of all channels)
    xrf_map, _ = gen_xrf_map_const(element_line_groups,
                                   nx=nx, ny=ny,
                                   incident_energy=incident_energy,
                                   n_spectrum_points=n_spectrum_points,
                                   background_area=background_area,
                                   **spectrum_parameters)

    # Distribute total fluorescence into 'num_det_channels'
    channel_coef = np.arange(num_det_channels) + 10.0
    channel_coef /= np.sum(channel_coef)

    # Create datasets
    data_xrf = {}
    data_xrf["det_sum"] = np.copy(xrf_map)
    for n in range(num_det_channels):
        data_xrf[f"det{n + 1}"] = xrf_map * channel_coef[n]

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
    metadata["scan_uid"] = _gen_rs(8) + "-" + _gen_rs(4) + "-" + _gen_rs(4) + "-" + \
        _gen_rs(4) + "-" + _gen_rs(12)
    metadata["instrument_mono_incident_energy"] = incident_energy

    return data_xrf, data_scalers, data_pos, metadata


def create_hdf5_xrf_map_const(*, scan_id, wd=None,
                              element_line_groups=None,
                              save_det_sum=True,
                              save_det_channels=True,
                              num_det_channels=3,
                              nx=10, ny=5,
                              incident_energy=12.0,
                              n_spectrum_points=4096,
                              background_area=0,
                              spectrum_parameters=None):



    if not save_det_sum:
        logger.warning("The sum of the detector channels is always saved. ")

    # Prepare file name
    fln = f"scan2D_{scan_id}_sim.h5"
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
        nx=nx, ny=ny,
        incident_energy=incident_energy,
        n_spectrum_points=n_spectrum_points,
        background_area=background_area,
        spectrum_parameters=spectrum_parameters
    )

    data = {}
    data.update(data_xrf)
    data.update(data_scalers)
    data.update(data_pos)

    write_db_to_hdf_base(fpath, data, metadata=metadata,
                         file_overwrite_existing=True,
                         create_each_det=save_det_channels)


def gen_hdf5_quantitative_analysis_dataset(*, wd=None, standards_serial_list = None,
                                           test_element_concentrations = None):

    if not standards_serial_list:
        raise RuntimeError("There must be at least one standard loaded. Pass the list "
                           "of standards as value of the parameter 'standard_list'")

    nx, ny = 30, 20

    incident_energy = 13.0

    # For simplicity use the same emission intensity for all lines
    intensity_counts_per_ug = 10.0
    # Density for elements that are not part of the standard
    density_not_in_standard = 20

    if test_element_concentrations is None:
        test_element_concentrations = []
    test_element_lines = []

    # Load standards
    param_quant_estimation = ParamQuantEstimation()
    param_quant_estimation.load_standards()

    lines_for_testing = []
    scan_id = 1000
    for serial in standards_serial_list:
        standard = param_quant_estimation.find_standard(serial, key="serial")
        # ALL standards must exist
        if not standard:
            raise RuntimeError(f"Standard with serial #{serial} is not found.")

        param_quant_estimation.set_selected_standard(standard)
        param_quant_estimation.gen_fluorescence_data_dict(incident_energy=incident_energy)
        element_lines = param_quant_estimation.fluorescence_data_dict["element_lines"].copy()
        line_group = []
        for key, value in element_lines.items():
            area = value["density"] * intensity_counts_per_ug
            line_group.append((key, {"area": value["density"] * intensity_counts_per_ug},))
            el = key.split('_')[0]
            if (el in test_element_concentrations) and (key not in test_element_lines):
                lines_for_testing.append((key, {"area": area},))
                test_element_lines.append(key)
            #else:
            #    lines_for_testing.append((key, {"area": density_not_in_standard * intensity_counts_per_ug},))

        create_hdf5_xrf_map_const(scan_id=scan_id, wd=wd, element_line_groups=line_group,
                                  nx=nx, ny=ny, incident_energy=incident_energy)
        scan_id += 1

    create_hdf5_xrf_map_const(scan_id=2000, wd=wd, element_line_groups=lines_for_testing,
                              nx=nx, ny=ny, incident_energy=incident_energy)



"""

#element_line_groups = [("Si_K", {'area' : 10})]
#element_line_groups = [("Ba_L", {'area' : 10})]
#element_line_groups = [("Pt_M", {'area' : 10})]
element_line_groups = [("Si_K", {'area' : 10}), ("Ba_L", {'area' : 10}), ("Pt_M", {'area' : 10})]

spectrum, xx_energy = gen_spectrum(element_line_groups)

# Add constant background
#background = np.ones_like(spectrum, dtype=float) / spectrum.size * 10.0
#spectrum += background

#plt.semilogy(xx_energy, spectrum)
plt.plot(xx_energy, spectrum)
plt.show()

nx, ny = 5, 10
#nx, ny = 1, 1
#nx, ny = 1, 200
#nx, ny = 200, 1
#nx, ny = 2, 200
#nx, ny = 200, 2
n_spectrum_samples = 4096

hdf_file_name = "test.h5"
print(f"Creating new hdf5 file: '{hdf_file_name}'")
with h5py.File(hdf_file_name, "w") as f:  # Erase the old one "w"

    grp_xrfmap = f.create_group("xrfmap")

    grp_xrfmap_detsum = grp_xrfmap.create_group("detsum")
    dset_xrfmap_detsum_counts = grp_xrfmap_detsum.create_dataset("counts", (ny, nx, n_spectrum_samples), dtype = 'f')

    counts = np.ndarray(shape=(ny, nx, n_spectrum_samples), dtype=float)
    for n1 in range(ny):
        for n2 in range(nx):
            # counts[n1, n2, :] = n1 * ny + n2
            # counts[n1, n2, :] = 1
            counts[n1, n2, :] = spectrum.copy()
    dset_xrfmap_detsum_counts[:, :, :] = counts

    grp_xrfmap_positions = grp_xrfmap.create_group("positions")
    dset_xrfmap_positions_name = grp_xrfmap_positions.create_dataset("name", (2,), dtype="S10")
    dset_xrfmap_positions_pos = grp_xrfmap_positions.create_dataset("pos", (2, ny, nx), dtype = 'f')

    dset_xrfmap_positions_name[:] = np.array([b"x_pos", b"y_pos"])
    x_offset, x_step, y_offset, y_step = 0.05, 0.001, 0.1, 0.002
    xy = np.ndarray(shape=(2, ny, nx), dtype=float)
    # Change along x
    for n in range(nx):
        xy[0, :, n] = x_step * n + x_offset
    # Change along y
    for n in range(ny):
        xy[1, n, :] = y_step * n + y_offset
    dset_xrfmap_positions_pos[:, :, :] = xy

    grp_xrfmap_scalers = grp_xrfmap.create_group("scalers")
    dset_xrfmap_scalers_name = grp_xrfmap_scalers.create_dataset("name", (4,), dtype="S10")
    dset_xrfmap_scalers_pos = grp_xrfmap_scalers.create_dataset("val", (ny, nx, 4), dtype = 'f')

    dset_xrfmap_scalers_name[:] = np.array([b"i0", b"time", b"i0_time", b"time_diff"])
    scalers = np.ndarray(shape=(ny, nx, 4), dtype=float)
    # Scaler 'i0'
    scalers[:, :, 0] = 10
    #scalers[0, 0, 0] = 0  # The scaler has one zero (for testing of normalization)
    #scalers[:, :, 0] = 0   # Zero scaler (for testing of normalization)
    # Scaler 'time' (scanning along x axis)
    t0, dt = 2.5, 5
    for n in range(nx):
        scalers[:, n, 1] = dt * n + t0
    # Scaler 'i0_time'
    dt_i0 = 1000
    scalers[:, :, 2] = dt_i0
    # Scaler 'time_diff'
    scalers[:, :, 3] = dt

    dset_xrfmap_scalers_pos[:, :, :] = scalers

"""