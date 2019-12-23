import os
import pytest
import jsonschema
import copy
from pyxrf.core.quant_analysis import (
    save_xrf_standard_yaml_file, load_xrf_standard_yaml_file, _xrf_standard_schema,
    load_included_xrf_standard_yaml_file)

# Short example of XRF standard data
_standard_data_sample = [
        {
            "name": "Micromatter 41157",
            "serial": "41157",
            "description": "InSx 22.4 (In=16.0 S=6.4) / SrF2 20.6 / "
                           "Cr 19.8 / Ni 19.2 / GaAs 19.1 (Ga=8.3 As=10.8)",
            "compounds": {
                "In": 16.0,
                "S": 6.4,
                "SrF2": 20.6,
                "Cr": 19.8,
                "Ni": 19.2,
                "Ga": 8.3,
                "As": 10.8
            },
            "density": 101.1
        },
        {
            "name": "Micromatter 41164",
            "serial": "41164",
            "description": "CeF3 21.1 / Au 20.6",
            "compounds": {
                "CeF3": 21.1,
                "Au": 20.6
            }
            # Missing optional 'density' field
        }
    ]


@pytest.mark.parametrize("standard_data", [
    _standard_data_sample,
    []  # The function should also work if the list is empty
])
def test_save_xrf_standard_yaml_file1(tmp_path, standard_data):
    r"""Basic test of function 'save_xrf_standard_yaml_file' and 'load_xrf_standard_yaml_file'"""

    yaml_path = ["yaml", "param", "file"]
    file_name = "standard.yaml"
    yaml_path = os.path.join(tmp_path, *yaml_path, file_name)

    # Sample data
    save_xrf_standard_yaml_file(yaml_path, standard_data)

    data_loaded = load_xrf_standard_yaml_file(yaml_path)

    assert data_loaded == standard_data, \
        "Loaded data is not equal to the original data"


def test_save_xrf_standard_yaml_file2(tmp_path):
    r"""Test the case of of no match between total density and sum of densities of individual compounds"""

    yaml_path = ["yaml", "param", "file"]
    file_name = "standard.yaml"
    yaml_path = os.path.join(tmp_path, *yaml_path, file_name)

    standard_data = []  # No records, but the file will still be saved

    # Sample data
    save_xrf_standard_yaml_file(yaml_path, standard_data)

    # Try to overwrite the file
    with pytest.raises(IOError, match=f"File '{yaml_path}' already exists"):
        save_xrf_standard_yaml_file(yaml_path, standard_data)

    # The following should work
    save_xrf_standard_yaml_file(yaml_path, standard_data, overwrite_existing=True)


def test_save_xrf_standard_yaml_file3(tmp_path):
    r"""Test the case of existing file"""

    yaml_path = ["yaml", "param", "file"]
    file_name = "standard.yaml"
    yaml_path = os.path.join(tmp_path, *yaml_path, file_name)

    standard_data = copy.deepcopy(_standard_data_sample)
    standard_data[0]['density'] = 50.0  # Wrong value of total density

    # Save incorrect data
    save_xrf_standard_yaml_file(yaml_path, standard_data)
    #    and now try to read it
    with pytest.raises(RuntimeError, match="Sum of areal densities does not match total density"):
        load_xrf_standard_yaml_file(yaml_path)


def test_load_xrf_standard_yaml_file1(tmp_path):
    r"""Try loading non-existent YAML file"""

    file_name = "standard.yaml"

    # Try loading from the existing directory
    yaml_path = os.path.join(tmp_path, file_name)
    with pytest.raises(IOError, match=f"File '{yaml_path}' does not exist"):
        load_xrf_standard_yaml_file(yaml_path)

    # Try loading from the non-existing directory
    yaml_path = ["yaml", "param", "file"]
    yaml_path = os.path.join(tmp_path, *yaml_path, file_name)
    with pytest.raises(IOError, match=f"File '{yaml_path}' does not exist"):
        load_xrf_standard_yaml_file(yaml_path)


def test_load_xrf_standard_yaml_file2(tmp_path):
    r"""Test for reporting schema violation"""

    yaml_path = ["yaml", "param", "file"]
    file_name = "standard.yaml"
    yaml_path = os.path.join(tmp_path, *yaml_path, file_name)

    standard_data = _standard_data_sample
    # Change name from string to number (this will not satisfy the built-in schema)
    for v in standard_data:
        v["name"] = 50.36

    # Save the changed dataset
    save_xrf_standard_yaml_file(yaml_path, standard_data)

    with pytest.raises(jsonschema.ValidationError):
        load_xrf_standard_yaml_file(yaml_path)

    # Disable schema validation by setting 'schema=None'
    load_xrf_standard_yaml_file(yaml_path, schema=None)

    # Change the schema to match data and validate with changed schema
    schema = copy.deepcopy(_xrf_standard_schema)
    schema['properties']['name'] = {"type": ["string", "number"]}
    standard_data_out = load_xrf_standard_yaml_file(yaml_path, schema=schema)
    assert standard_data_out == standard_data, \
        "Loaded data is different from the original"


def test_load_included_xrf_standard_yaml_file():

    data = load_included_xrf_standard_yaml_file()
    assert len(data) > 1, "Standard YAML file can not be read"
