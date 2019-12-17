import os
import jsonschema
import pytest

from pyxrf.xanes_maps.xanes_maps_api import (
    _build_xanes_map_api, _build_xanes_map_param_default, _build_xanes_map_param_schema,
    build_xanes_map)

from pyxrf.core.yaml_param_files import (
    _parse_docstring_parameters, _verify_parsed_docstring,
    create_yaml_parameter_file, read_yaml_parameter_file)


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
    with pytest.raises(RuntimeError, match="no file path is specified"):
        build_xanes_map(create_parameter_file=True, allow_exceptions=True)
    # Specify scan ID instead of the path as the first argument
    with pytest.raises(RuntimeError, match="no file path is specified"):
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
    with pytest.raises(TypeError):
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
