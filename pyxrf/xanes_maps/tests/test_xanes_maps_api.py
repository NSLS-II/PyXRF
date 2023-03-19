import logging
import os
from io import StringIO

import jsonschema
import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

from pyxrf.core.yaml_param_files import (
    _parse_docstring_parameters,
    _verify_parsed_docstring,
    create_yaml_parameter_file,
    read_yaml_parameter_file,
)
from pyxrf.xanes_maps.xanes_maps_api import (
    _build_xanes_map_api,
    _build_xanes_map_param_default,
    _build_xanes_map_param_schema,
    _save_spectrum_as_csv,
    adjust_incident_beam_energies,
    build_xanes_map,
    check_elines_activation_status,
    subtract_xanes_pre_edge_baseline,
)


def _get_xanes_energy_axis():
    r"""
    Generates a reasonable range of energy values for testing of XANES processing.
    The set of energies contains values in pre- and post-edge regions.
    """
    eline = "Fe_K"
    eline_activation_energy = 7.1115  # keV, an approximate value that makes test suceed
    e_min, e_max, de = 7.05, 7.15, 0.001
    incident_energies = np.mgrid[e_min : e_max : ((e_max - e_min) / de + 1) * 1j]
    incident_energies = np.round(incident_energies, 5)  # Nicer view for debugging
    return incident_energies, eline, eline_activation_energy


def test_check_elines_activation_status():
    r"""Tests for ``check_elines_activation_status``"""

    incident_energies, eline, eline_activation_energy = _get_xanes_energy_axis()

    activation_status = [_ >= eline_activation_energy for _ in incident_energies]

    # Send incident energies as an array
    activation_status_output = check_elines_activation_status(np.asarray(incident_energies), eline)
    assert (
        activation_status == activation_status_output
    ), "Activation status of some energy values is determined incorrectly"

    # Send incident energies as a list
    activation_status_output = check_elines_activation_status(list(incident_energies), eline)
    assert (
        activation_status == activation_status_output
    ), "Activation status of some energy values is determined incorrectly"

    # Empty list of energy should yield empty list of flags
    activation_status_output = check_elines_activation_status([], eline)
    assert not activation_status_output, "Empty list of incident energies is processed incorrectly"


def test_adjust_incident_beam_energies():
    r"""Tests for ``adjust_incident_beam_energies``"""

    incident_energies, eline, eline_activation_energy = _get_xanes_energy_axis()
    incident_energies = np.random.permutation(incident_energies)

    threshold = np.min(incident_energies[incident_energies >= eline_activation_energy])
    ie_adjusted = np.clip(incident_energies, a_min=threshold, a_max=None)

    # Send incident energies as an array
    ie_adjusted_output = adjust_incident_beam_energies(np.asarray(incident_energies), eline)
    np.testing.assert_almost_equal(
        ie_adjusted_output, ie_adjusted, err_msg="Incident energies are adjusted incorrectly"
    )

    # Send incident energies as a list
    ie_adjusted_output = adjust_incident_beam_energies(list(incident_energies), eline)
    np.testing.assert_almost_equal(
        ie_adjusted_output, ie_adjusted, err_msg="Incident energies are adjusted incorrectly"
    )


def _get_sim_pre_edge_spectrum(incident_energies, eline_activation_energy, pre_edge_upper_keV, img_dims):
    r"""
    Generate spectrum for testing baseline removal function
    ``img_dims`` is the size of the image, e.g. [5, 10] is 5x10 pixel image.
    The spectrum points are always placed along axis 0.
    """

    n_pts = incident_energies.shape[0]
    n_pixels = np.prod(img_dims)
    n_pre_edge = np.sum(incident_energies < eline_activation_energy + pre_edge_upper_keV)

    spectrum = np.random.rand(n_pts, n_pixels)
    spectrum_no_base = spectrum
    for n in range(n_pixels):
        spectrum[n_pre_edge:n_pts, n] += 2
        v_bs = np.median(spectrum[0:n_pre_edge, n])
        spectrum_no_base[:, n] = spectrum[:, n] - v_bs

    if n_pixels == 1:
        spectrum = np.squeeze(spectrum, axis=1)
        spectrum_no_base = np.squeeze(spectrum_no_base, axis=1)
    else:
        spectrum = np.reshape(spectrum, np.insert(img_dims, 0, n_pts))
        spectrum_no_base = np.reshape(spectrum_no_base, np.insert(img_dims, 0, n_pts))

    return spectrum, spectrum_no_base


def test_subtract_xanes_pre_edge_baseline1():
    r"""Tests for ``subtract_xanes_pre_edge_baseline``"""

    pre_edge_upper_keV_default = -0.01  # Relative location of the pre-edge upper boundary
    pre_edge_upper_keV = pre_edge_upper_keV_default

    incident_energies, eline, eline_activation_energy = _get_xanes_energy_axis()

    # Tests with a single spectrum with default 'pre_edge_upper_keV'
    #   1. Allow negative output values in the results with subtracted baseline
    spectrum, spectrum_no_base = _get_sim_pre_edge_spectrum(
        incident_energies, eline_activation_energy, pre_edge_upper_keV, [3, 5]
    )
    spectrum_out = subtract_xanes_pre_edge_baseline(spectrum, incident_energies, eline, non_negative=False)
    np.testing.assert_almost_equal(
        spectrum_out, spectrum_no_base, err_msg="Baseline subtraction from 1D XANES spectrum failed"
    )
    #   2. Non-negative value only (default)
    spectrum_no_base = np.clip(spectrum_no_base, a_min=0, a_max=None)
    spectrum_out = subtract_xanes_pre_edge_baseline(spectrum, incident_energies, eline)
    np.testing.assert_almost_equal(
        spectrum_out, spectrum_no_base, err_msg="Baseline subtraction from 1D XANES spectrum failed"
    )

    # Test for the case when no pre-edge points are detected (RuntimeError)
    #   Add 1 keV to the incident energy, in this case all energy values activate the line
    spectrum, spectrum_no_base = _get_sim_pre_edge_spectrum(
        incident_energies, eline_activation_energy, pre_edge_upper_keV, [3, 7]
    )
    with pytest.raises(RuntimeError, match="No pre-edge points were found"):
        subtract_xanes_pre_edge_baseline(spectrum, incident_energies + 1, eline)


# fmt: off
@pytest.mark.parametrize("p_generate, p_test", [
    ({"pre_edge_upper_keV": -0.01, "img_dims": [1]}, {"pre_edge_upper_keV": -0.01}),
    ({"pre_edge_upper_keV": -0.008, "img_dims": [1]}, {"pre_edge_upper_keV": -0.008}),
    ({"pre_edge_upper_keV": -0.01, "img_dims": [7]}, {"pre_edge_upper_keV": -0.01}),
    ({"pre_edge_upper_keV": -0.01, "img_dims": [3, 7]}, {"pre_edge_upper_keV": -0.01}),
    ({"pre_edge_upper_keV": -0.01, "img_dims": [5, 3, 7]}, {"pre_edge_upper_keV": -0.01}),
    ({"pre_edge_upper_keV": -0.013, "img_dims": [5, 3, 7]}, {"pre_edge_upper_keV": -0.013}),
])
# fmt: on
def test_subtract_xanes_pre_edge_baseline2(p_generate, p_test):
    r"""
    Tests for 'subtract_xanes_pre_edge_baseline':
        Successful tests for different combination of parameters
    """

    incident_energies, eline, eline_activation_energy = _get_xanes_energy_axis()

    spectrum, spectrum_no_base = _get_sim_pre_edge_spectrum(
        incident_energies, eline_activation_energy, **p_generate
    )

    spectrum_no_base = np.clip(spectrum_no_base, a_min=0, a_max=None)
    spectrum_out = subtract_xanes_pre_edge_baseline(spectrum, incident_energies, eline, **p_test)
    np.testing.assert_almost_equal(
        spectrum_out, spectrum_no_base, err_msg="Baseline subtraction from 1D XANES spectrum failed"
    )


def test_parse_docstring_parameters__build_xanes_map_api():
    """Test that the docstring of ``build_xanes_map_api`` and ``_build_xanes_map_param_default``
    are consistent: parse the docstring and match with the dictionary"""
    parameters = _parse_docstring_parameters(_build_xanes_map_api.__doc__)
    _verify_parsed_docstring(parameters, _build_xanes_map_param_default)


def test_create_yaml_parameter_file__build_xanes_map_api(tmp_path):
    # Some directory
    yaml_dirs = ["param", "file", "directory"]
    yaml_fln = "parameter.yaml"
    file_path = os.path.join(tmp_path, *yaml_dirs, yaml_fln)

    create_yaml_parameter_file(
        file_path=file_path,
        function_docstring=_build_xanes_map_api.__doc__,
        param_value_dict=_build_xanes_map_param_default,
        dir_create=True,
    )

    param_dict_recovered = read_yaml_parameter_file(file_path=file_path)

    # Validate the schema of the recovered data
    jsonschema.validate(instance=param_dict_recovered, schema=_build_xanes_map_param_schema)

    assert (
        _build_xanes_map_param_default == param_dict_recovered
    ), "Parameter dictionary read from YAML file is different from the original parameter dictionary"


def test_build_xanes_map_1(tmp_path):
    """Basic test: creating new parameter file"""

    # Successful test
    yaml_dir = "param"
    yaml_fln = "parameter.yaml"
    yaml_fln2 = "parameter2.yaml"
    # Create the directory
    os.makedirs(os.path.join(tmp_path, yaml_dir), exist_ok=True)

    file_path = os.path.join(tmp_path, yaml_dir, yaml_fln)
    file_path2 = os.path.join(tmp_path, yaml_dir, yaml_fln2)

    # The function will raise an exception if it fails
    build_xanes_map(parameter_file_path=file_path, create_parameter_file=True, allow_exceptions=True)
    # Also check if the file is there
    assert os.path.isfile(file_path), f"File '{file_path}' was not created"

    # Try creating file with specifying 'file_path' as the first argument (not kwarg)
    build_xanes_map(file_path2, create_parameter_file=True, allow_exceptions=True)
    # Also check if the file is there
    assert os.path.isfile(file_path2), f"File '{file_path2}' was not created"

    # Try to create file that already exists
    with pytest.raises(IOError, match=r"File .* already exists"):
        build_xanes_map(parameter_file_path=file_path, create_parameter_file=True, allow_exceptions=True)

    # Try to create file that already exists with disabled exceptions
    #   (the function is expected to exit without raising the exception)
    build_xanes_map(parameter_file_path=file_path, create_parameter_file=True)

    # Try to create file in non-existing directory
    file_path3 = os.path.join(tmp_path, "some_directory", "yaml_fln")
    with pytest.raises(IOError, match=r"Directory .* does not exist"):
        build_xanes_map(parameter_file_path=file_path3, create_parameter_file=True, allow_exceptions=True)

    # Try creating the parameter file without specifying the path
    with pytest.raises(RuntimeError, match="parameter file path is not specified"):
        build_xanes_map(create_parameter_file=True, allow_exceptions=True)
    # Specify scan ID instead of the path as the first argument
    with pytest.raises(RuntimeError, match="parameter file path is not specified"):
        build_xanes_map(1000, create_parameter_file=True, allow_exceptions=True)


def test_build_xanes_map_2():
    """Try calling the function with invalid (not supported) argument"""
    with pytest.raises(RuntimeError, match=r"The function is called with invalid arguments:.*\n.*some_arg1"):
        build_xanes_map(some_arg1=65, allow_exceptions=True)
    with pytest.raises(
        RuntimeError, match=r"The function is called with invalid arguments:.*\n.*some_arg1.*\n.*some_arg2"
    ):
        build_xanes_map(some_arg1=65, some_arg2="abc", allow_exceptions=True)


def test_build_xanes_map_3():
    """Test passing arguments to ``_build_xanes_map_api``"""

    # The function should fail, because the emission line is not specified
    with pytest.raises(ValueError):
        build_xanes_map(allow_exceptions=True)

    # The function is supposed to fail, because 'xrf_subdir' is not specified
    with pytest.raises(ValueError, match="The parameter 'xrf_subdir' is None or contains an empty string"):
        build_xanes_map(emission_line="Fe_K", xrf_subdir="", allow_exceptions=True)

    # The function should succeed if exceptions are not allowed
    build_xanes_map(emission_line="Fe_K", xrf_subdir="")


def test_build_xanes_map_4(tmp_path):
    """Load parameters from YAML file"""

    # Successful test
    yaml_dir = "param"
    yaml_fln = "parameter.yaml"
    # Create the directory
    os.makedirs(os.path.join(tmp_path, yaml_dir), exist_ok=True)
    file_path = os.path.join(tmp_path, yaml_dir, yaml_fln)

    # Create YAML file
    build_xanes_map(parameter_file_path=file_path, create_parameter_file=True, allow_exceptions=True)
    # Now start the program and load the file, the call should fail, because 'xrf_subdir' is empty str
    with pytest.raises(ValueError, match="The parameter 'xrf_subdir' is None or contains an empty string"):
        build_xanes_map(emission_line="Fe_K", parameter_file_path=file_path, xrf_subdir="", allow_exceptions=True)

    # Repeat the same operation with exceptions disabled. The operation should succeed.
    build_xanes_map(emission_line="Fe_K", parameter_file_path=file_path, xrf_subdir="")


# fmt: off
@pytest.mark.parametrize("kwargs", [
    {},
    {"wd": None, "msg_info": "header line 1"},
    {"wd": ".", "msg_info": "line1\nline2"},
    {"wd": "test_dir", "msg_info": "line1\n  line2"},
    {"wd": ("test_dir1", "test_dir2"), "msg_info": "line1\n line2"}])
# fmt: on
def test_save_spectrum_as_csv_1(tmp_path, caplog, kwargs):
    """Save data file, then read it and verify that the data match"""

    fln = "output.csv"

    os.chdir(tmp_path)  # Make 'tmp_path' current directory

    fln_full = fln
    if ("wd" in kwargs) and (kwargs["wd"] is not None):
        if isinstance(kwargs["wd"], tuple):
            kwargs["wd"] = os.path.join(*kwargs["wd"])
        fln_full = os.path.join(kwargs["wd"], fln)
    fln_full = os.path.abspath(fln_full)

    n_pts = 50
    energy = np.random.rand(n_pts)
    spectrum = np.random.rand(n_pts)

    caplog.set_level(logging.INFO)

    # Save CSV file
    _save_spectrum_as_csv(fln=fln, energy=energy, spectrum=spectrum, **kwargs)

    assert f"Selected spectrum was saved to file '{fln_full}'" in str(
        caplog.text
    ), "Incorrect reporting of the event of the correctly saved file"
    caplog.clear()

    # Now read the CSV file as a string
    with open(fln_full, "r") as f:
        s = f.read()

    # Check if the comment lines were written in the file
    if "msg_info" in kwargs:
        # Find the match for each line in 'msg_info'
        for s_msg in kwargs["msg_info"].split("\n"):
            assert f"# {s_msg}" in s, "Mismatch between original and loaded comment lines"

    # Remove comments (lines that start with #, may contain spaces at the beginning of the string)
    s = "\n".join([_ for _ in s.split("\n") if not _.strip().startswith("#")])

    dframe = pd.read_csv(StringIO(s))
    assert tuple(dframe.columns) == (
        "Incident Energy, keV",
        "XANES spectrum",
    ), f"Incorrect column labels: {tuple(dframe.columns)}"

    data = dframe.values
    energy2, spectrum2 = data[:, 0], data[:, 1]
    npt.assert_array_almost_equal(energy, energy2, err_msg="Recovered energy array is different from the original")
    npt.assert_array_almost_equal(
        spectrum, spectrum2, err_msg="Recovered spectrum array is different from the original"
    )


def test_save_spectrum_as_csv_2(tmp_path, caplog):
    """Failing cases"""

    fln = "output.csv"
    os.chdir(tmp_path)  # Make 'tmp_path' current directory

    n_pts = 50
    energy = np.random.rand(n_pts)
    spectrum = np.random.rand(n_pts)

    caplog.set_level(logging.INFO)
    _save_spectrum_as_csv(fln=fln, energy=None, spectrum=spectrum)
    assert "The array 'energy' is None" in str(caplog.text)
    caplog.clear()

    _save_spectrum_as_csv(fln=fln, spectrum=spectrum)
    assert "The array 'energy' is None" in str(caplog.text)
    caplog.clear()

    _save_spectrum_as_csv(fln=fln, energy=energy, spectrum=None)
    assert "The array 'spectrum' is None" in str(caplog.text)
    caplog.clear()

    _save_spectrum_as_csv(fln=fln, energy=energy)
    assert "The array 'spectrum' is None" in str(caplog.text)
    caplog.clear()

    spectrum = spectrum[:-1]
    _save_spectrum_as_csv(fln=fln, energy=energy, spectrum=spectrum)
    assert "Arrays 'energy' and 'spectrum' have different size:" in str(caplog.text)
    caplog.clear()
