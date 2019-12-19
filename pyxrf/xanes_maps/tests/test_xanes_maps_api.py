import os
import jsonschema
import pytest
import numpy as np

from pyxrf.xanes_maps.xanes_maps_api import (
    _build_xanes_map_api, _build_xanes_map_param_default, _build_xanes_map_param_schema,
    build_xanes_map, check_elines_activation_status, adjust_incident_beam_energies)

from pyxrf.core.yaml_param_files import (
    _parse_docstring_parameters, _verify_parsed_docstring,
    create_yaml_parameter_file, read_yaml_parameter_file)


def test_check_elines_activation_status():
    r""" Tests for ``check_elines_activation_status``"""

    eline = "Fe_K"
    eline_activation_energy = 7.115  # keV, an approximate value that makes test suceed
    e_min, e_max, de = 7.05, 7.15, 0.01
    incident_energies = np.mgrid[e_min: e_max: ((e_max - e_min) / de + 1) * 1j]
    incident_energies = np.round(incident_energies, 5)  # Nicer view for debugging
    incident_energies = np.random.permutation(incident_energies)

    activation_status = [_ >= eline_activation_energy for _ in incident_energies]

    # Send incident energies as an array
    activation_status_output = check_elines_activation_status(np.asarray(incident_energies), eline)
    assert activation_status == activation_status_output, \
        "Activation status of some energy values is determined incorrectly"

    # Send incident energies as a list
    activation_status_output = check_elines_activation_status(list(incident_energies), eline)
    assert activation_status == activation_status_output, \
        "Activation status of some energy values is determined incorrectly"

    # Empty list of energy should yield empty list of flags
    activation_status_output = check_elines_activation_status([], eline)
    assert not activation_status_output, \
        "Empty list of incident energies is processed incorrectly"


def test_adjust_incident_beam_energies():
    r""" Tests for ``adjust_incident_beam_energies``"""

    eline = "Fe_K"
    eline_activation_energy = 7.115  # keV, an approximate value that makes test suceed
    e_min, e_max, de = 7.05, 7.15, 0.01
    incident_energies = np.mgrid[e_min: e_max: ((e_max - e_min) / de + 1) * 1j]
    incident_energies = np.round(incident_energies, 5)  # Nicer view for debugging
    incident_energies = np.random.permutation(incident_energies)

    threshold = np.min(incident_energies[incident_energies >= eline_activation_energy])
    ie_adjusted = np.clip(incident_energies, a_min=threshold, a_max=None)

    # Send incident energies as an array
    ie_adjusted_output = adjust_incident_beam_energies(np.asarray(incident_energies), eline)
    np.testing.assert_almost_equal(ie_adjusted_output, ie_adjusted,
                                   err_msg="Incident energies are adjusted incorrectly")

    # Send incident energies as a list
    ie_adjusted_output = adjust_incident_beam_energies(list(incident_energies), eline)
    np.testing.assert_almost_equal(ie_adjusted_output, ie_adjusted,
                                   err_msg="Incident energies are adjusted incorrectly")


def test_parse_docstring_parameters__build_xanes_map_api():
    """ Test that the docstring of ``build_xanes_map_api`` and ``_build_xanes_map_param_default``
    are consistent: parse the docstring and match with the dictionary"""
    parameters = _parse_docstring_parameters(_build_xanes_map_api.__doc__)
    _verify_parsed_docstring(parameters, _build_xanes_map_param_default)


def test_create_yaml_parameter_file__build_xanes_map_api(tmp_path):

    # Some directory
    yaml_dirs = ["param", "file", "directory"]
    yaml_fln = "parameter.yaml"
    file_path = os.path.join(tmp_path, *yaml_dirs, yaml_fln)

    create_yaml_parameter_file(file_path=file_path, function_docstring=_build_xanes_map_api.__doc__,
                               param_value_dict=_build_xanes_map_param_default, dir_create=True)

    param_dict_recovered = read_yaml_parameter_file(file_path=file_path)

    # Validate the schema of the recovered data
    jsonschema.validate(instance=param_dict_recovered, schema=_build_xanes_map_param_schema)

    assert _build_xanes_map_param_default == param_dict_recovered, \
        "Parameter dictionary read from YAML file is different from the original parameter dictionary"


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
    with pytest.raises(RuntimeError,
                       match=f"The function is called with invalid arguments: some_arg1"):
        build_xanes_map(some_arg1=65, allow_exceptions=True)
    with pytest.raises(RuntimeError,
                       match=f"The function is called with invalid arguments: some_arg1, some_arg2"):
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
        build_xanes_map(emission_line="Fe_K", parameter_file_path=file_path,
                        xrf_subdir="", allow_exceptions=True)

    # Repeat the same operation with exceptions disabled. The operation should succeed.
    build_xanes_map(emission_line="Fe_K", parameter_file_path=file_path, xrf_subdir="")
