import os

import numpy as np
import numpy.testing as npt
import pytest

from pyxrf.simulation.sim_xrf_scan_data import (
    _get_elemental_line_parameters,
    create_hdf5_xrf_map_const,
    create_xrf_map_data,
    gen_hdf5_qa_dataset,
    gen_hdf5_qa_dataset_preset_1,
    gen_xrf_map_const,
    gen_xrf_spectrum,
)


# fmt: off
@pytest.mark.parametrize("params", [
    {"eline": "Fe_K", "incident_energy": 12.0, "n_lines": 4},
    {"eline": "Fe_K", "incident_energy": 1.0, "n_lines": 0},
    {"eline": "Se_L", "incident_energy": 12.0, "n_lines": 9},
    {"eline": "Se_L", "incident_energy": 1.0, "n_lines": 0},
    {"eline": "W_M", "incident_energy": 12.0, "n_lines": 4},
    {"eline": "W_M", "incident_energy": 1, "n_lines": 0},
])
# fmt: on
def test_get_elemental_line_parameters_1(params):
    eline_list = _get_elemental_line_parameters(
        elemental_line=params["eline"], incident_energy=params["incident_energy"]
    )

    assert isinstance(eline_list, list), "Returned value is not a list"
    assert len(eline_list) == params["n_lines"], (
        f"The number of emission lines {len(eline_list)} does not match "
        f"the expected number {params['n_lines']}"
    )

    expected_keys = set(["name", "energy", "ratio"])
    for eline in eline_list:
        assert expected_keys == set(
            eline.keys()
        ), "Dictionary with emission line parameters contains incorrect set of keys"


@pytest.mark.parametrize("eline", ["Fe_L", "Fe_M", "Ab_K"])
def test_get_elemental_line_parameters_2(eline):
    r"""Failing cases (emission line is correctly formatted, but not supported)"""
    with pytest.raises(RuntimeError, match=f"Elemental line {eline} is not supported"):
        _get_elemental_line_parameters(elemental_line=eline, incident_energy=12)


@pytest.mark.parametrize("eline", ["FeK", "Fe_Ka", "Fe_B", "fe_K", "Fe_k"])
def test_get_elemental_line_parameters_3(eline):
    r"""Failing cases (incorrectly formatted elemental line)"""
    with pytest.raises(RuntimeError, match=f"Elemental line {eline} has incorrect format"):
        _get_elemental_line_parameters(elemental_line=eline, incident_energy=12)


def test_gen_xrf_spectrum_1():
    r"""Successful operation of ``gen_xrf_spectrum``. Some tests are empirical and don't guarantee
    accurate operation of the function."""

    element_groups = {"Fe_K": {"area": 800}, "Se_L": {"area": 900}, "W_M": {"area": 1000}}
    total_area = sum([_["area"] for _ in element_groups.values()])
    # Expected (known) number of peaks due to emission lines in the spectrum
    #   The number of peak is not equal to the number of emission lines (some peaks overlap)
    n_peaks_expected = 5

    spectrum, xx = gen_xrf_spectrum(element_groups, incident_energy=12.0)

    npt.assert_almost_equal(sum(spectrum), total_area, err_msg="Total spectrum area is incorrect")

    # Compute the number of peaks (the number of lines) using primitive algorithm
    #   (there is no noise in simulated data)
    n_peaks = 0
    for n in range(1, len(spectrum) - 1):
        if (spectrum[n] > 1e-20) and (spectrum[n] == max(spectrum[n - 1 : n + 2])):
            n_peaks += 1

    assert n_peaks == n_peaks_expected, "The number of peaks in simulated data is incorrect"

    # Test if the energy axis has correct values (for default parameters)
    xx_expected = 0.01 * np.arange(4096)
    npt.assert_array_almost_equal(xx, xx_expected, err_msg="The returned energy axis values are incorrect")


def test_gen_xrf_spectrum_2():
    r"""Failing test of ``gen_xrf_spectrum``: invalid types of parameter ``element_line_groups``"""

    # 'element_groups' == None is valid
    element_groups = None
    spectrum_expected = np.zeros(shape=(4096,), dtype=float)
    spectrum, xx = gen_xrf_spectrum(element_groups, incident_energy=12.0)
    npt.assert_array_equal(spectrum, spectrum_expected, err_msg="Values in the spectrum array are not all zeros")

    # Try submit a list instead of dict
    with pytest.raises(RuntimeError, match="Parameter 'element_line_groups' has invalid type"):
        gen_xrf_spectrum([1, 2, 3], incident_energy=12.0)


def test_gen_xrf_spectrum_3():
    r"""Failing test of ``gen_xrf_spectrum``: unsupported emission line in the list"""

    # 'Fe_L' is not supported
    element_groups = {"Fe_L": {"area": 800}, "Se_L": {"area": 900}, "W_M": {"area": 1000}}

    with pytest.raises(RuntimeError):
        gen_xrf_spectrum(element_groups, incident_energy=12.0)


def test_gen_xrf_map_const_1():
    r"""Successful generation of XRF map"""

    element_groups = {"Fe_K": {"area": 800}, "Se_L": {"area": 900}, "W_M": {"area": 1000}}
    background_area = 1000
    nx, ny = 50, 100

    total_area = sum([_["area"] for _ in element_groups.values()])
    total_area += background_area

    xrf_map, xx = gen_xrf_map_const(
        element_groups, nx=nx, ny=ny, incident_energy=12.0, background_area=background_area
    )
    assert xrf_map.shape == (ny, nx, 4096), "Incorrect shape of generated XRF maps"
    assert xx.shape == (4096,), "Incorrect shape of energy axis"
    npt.assert_array_equal(xrf_map[0, 0, :], xrf_map[1, 1, :], err_msg="Elements of XRF map are not equal")
    npt.assert_almost_equal(
        np.sum(xrf_map[0, 0, :]),
        total_area,
        err_msg="Area of the generated spectrum does not match the expected value",
    )

    # Test generation of XRF map with different size spectrum
    xrf_map, xx = gen_xrf_map_const(
        element_groups, nx=nx, ny=ny, n_spectrum_points=1000, incident_energy=12.0, background_area=background_area
    )
    assert xrf_map.shape == (ny, nx, 1000), "Incorrect shape of generated XRF maps"
    assert xx.shape == (1000,), "Incorrect shape of energy axis"


def test_gen_xrf_map_const_2():
    r"""Failing test of ``gen_xrf_map_const``"""

    # 'Fe_L' is not supported
    element_groups = {"Fe_L": {"area": 800}, "Se_L": {"area": 900}, "W_M": {"area": 1000}}

    background_area = 1000
    nx, ny = 50, 100

    with pytest.raises(RuntimeError, match="Elemental line Fe_L is not supported"):
        gen_xrf_map_const(element_groups, nx=nx, ny=ny, incident_energy=12.0, background_area=background_area)

    # Spectrum with ZERO points
    with pytest.raises(RuntimeError, match="Spectrum must contain at least one point"):
        gen_xrf_map_const(
            element_groups,
            nx=nx,
            ny=ny,
            n_spectrum_points=0,
            incident_energy=12.0,
            background_area=background_area,
        )

    # XRF map has ZERO points
    with pytest.raises(RuntimeError, match="XRF map has zero pixels"):
        gen_xrf_map_const(
            element_groups,
            nx=0,
            ny=ny,
            n_spectrum_points=1000,
            incident_energy=12.0,
            background_area=background_area,
        )


def test_create_xrf_map_data():
    r"""Test creating xrf map dataset. Only tests if the datasets are returned."""

    element_groups = {"Fe_K": {"area": 800}, "Se_L": {"area": 900}, "W_M": {"area": 1000}}
    scan_id = 1000

    data_xrf, data_scalers, data_pos, metadata = create_xrf_map_data(
        scan_id=scan_id, element_line_groups=element_groups, nx=100, ny=50, background_area=0
    )

    assert data_xrf and data_scalers and data_pos and metadata, "Failed to create dataset"


def test_create_hdf5_xrf_map_const(tmp_path):
    r"""Test creating file. Only the existence of the created file is verified."""

    wd = os.path.join(tmp_path, "file_dir")

    element_groups = {"Fe_K": {"area": 800}, "Se_L": {"area": 900}, "W_M": {"area": 1000}}

    scan_id = 1000

    fln_path = create_hdf5_xrf_map_const(
        scan_id=scan_id, wd=wd, element_line_groups=element_groups, nx=100, ny=50, background_area=0
    )
    file_name = f"scan2D_{scan_id}_sim.h5"
    fln_path_expected = os.path.join(wd, file_name)
    assert fln_path == fln_path_expected, "File path is different from expected"
    assert os.path.isfile(fln_path_expected), f"File '{file_name}' was not created"

    # Add suffix at the end of file name
    fln_suffix = "hello123"
    fln_path = create_hdf5_xrf_map_const(
        scan_id=scan_id,
        wd=wd,
        fln_suffix=fln_suffix,
        element_line_groups=element_groups,
        nx=100,
        ny=50,
        background_area=0,
    )
    file_name = f"scan2D_{scan_id}_sim_{fln_suffix}.h5"
    fln_path_expected = os.path.join(wd, file_name)
    assert fln_path == fln_path_expected, "File path is different from expected"
    assert os.path.isfile(fln_path_expected), f"File '{file_name}' was not created"


def test_gen_hdf5_qa_dataset(tmp_path):
    r"""Test 'gen_hdf5_quantitative_analysis_dataset'.
    Simple test, which checks if all the files are created"""

    wd = os.path.join(tmp_path, "file_dir")

    standards_serials = ["41151", "41163"]
    test_elements = {}
    test_elements["Fe"] = {"density": 50}  # Density in ug/cm^2 (for simulated test scan)
    test_elements["W"] = {"density": 70}
    test_elements["Au"] = {"density": 80}
    files_saved = gen_hdf5_qa_dataset(wd=wd, standards_serials=standards_serials, test_elements=test_elements)
    assert len(files_saved) == 4, f"Incorrect number of saved files: expected 4, saved {len(files_saved)}"
    for fln in files_saved:
        assert os.path.isfile(fln), f"File '{fln}' was not created"


def test_gen_hdf5_qa_dataset_preset_1(tmp_path):
    r"""Test for 'gen_hdf5_qa_dataset_preset_1'. Test that the function runs."""
    gen_hdf5_qa_dataset_preset_1()
    gen_hdf5_qa_dataset_preset_1(wd=tmp_path)
