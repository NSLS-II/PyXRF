import os
import jsonschema

from pyxrf.xanes_maps.xanes_maps_api import (
    build_xanes_map_api, _build_xanes_map_param_default, _build_xanes_map_param_schema)

from pyxrf.core.yaml_param_files import (
    _parse_docstring_parameters, _verify_parsed_docstring,
    create_yaml_parameter_file, read_yaml_parameter_file)


def test_parse_docstring_parameters__build_xanes_map_api():
    """ Test that the docstring of ``build_xanes_map_api`` and ``_build_xanes_map_param_default``
    are consistent: parse the docstring and match with the dictionary"""
    parameters = _parse_docstring_parameters(build_xanes_map_api.__doc__)
    _verify_parsed_docstring(parameters, _build_xanes_map_param_default)


def test_create_yaml_parameter_file__build_xanes_map_api(tmp_path):

    # Some directory
    yaml_dirs = ["param", "file", "directory"]
    yaml_fln = "parameter.yaml"
    file_path = os.path.join(tmp_path, *yaml_dirs, yaml_fln)

    create_yaml_parameter_file(file_path=file_path, function_docstring=build_xanes_map_api.__doc__,
                               param_value_dict=_build_xanes_map_param_default, dir_create=True)

    param_dict_recovered = read_yaml_parameter_file(file_path=file_path)

    # Validate the schema of the recovered data
    jsonschema.validate(instance=param_dict_recovered, schema=_build_xanes_map_param_schema)

    assert _build_xanes_map_param_default == param_dict_recovered, \
        "Parameter dictionary read from YAML file is different from the original parameter dictionary"
