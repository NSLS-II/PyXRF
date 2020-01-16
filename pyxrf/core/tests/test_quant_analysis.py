import os
import pytest
import jsonschema
import copy
import numpy as np
import numpy.testing as npt
import time as ttime
from pyxrf.core.utils import convert_time_from_nexus_string
from pyxrf.core.xrf_utils import validate_element_str, generate_eline_list, split_compound_mass
from pyxrf.core.quant_analysis import (
    save_xrf_standard_yaml_file, load_xrf_standard_yaml_file, _xrf_standard_schema,
    load_included_xrf_standard_yaml_file, compute_standard_element_densities,
    _xrf_quant_fluor_schema, save_xrf_quant_fluor_json_file, load_xrf_quant_fluor_json_file,
    get_quant_fluor_data_dict, fill_quant_fluor_data_dict, prune_quant_fluor_data_dict,
    set_quant_fluor_data_dict_optional, set_quant_fluor_data_dict_time)

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

    standard_data = copy.deepcopy(_standard_data_sample)
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


def test_compute_standard_element_densities():

    standard_data = _standard_data_sample

    for data in standard_data:

        # Find total density
        if "density" in data:
            total_density = data["density"]
        else:
            total_density = np.sum(list(data["compounds"].values()))

        element_densities = compute_standard_element_densities(data["compounds"])

        # Validate that all the keys of the returned dictionary represent elements
        assert all([validate_element_str(_) for _ in element_densities.keys()]), \
            f"Some of the elements in the list {list(element_densities.keys())} have invalid representation"

        # Check if the sum of all return densities equals total density
        npt.assert_almost_equal(np.sum(list(element_densities.values())), total_density,
                                err_msg="The sum of element densities and the total sum don't match")


# -----------------------------------------------------------------------------------------------------


# Short example of XRF standard quantitative data record
#   The data has no physical meaning, used for testing of saving/loading of JSON file
_xrf_standard_fluor_sample = {
    "name": "Hypothetical sample #41157",
    "serial": "41157",
    "description": "Name of hypothetical sample",
    "element_lines": {
        "In": {"density": 16.0, "fluorescence": 1.5453452},
        "S_K": {"density": 6.4, "fluorescence": 2.0344345},
        "Sr_L": {"density": 20.6, "fluorescence": 0.93452365},
        "Au_M": {"density": 10.4, "fluorescence": 0.734234},
        "Cr_Ka": {"density": 19.8, "fluorescence": 0.7435234},
        "Ni_Kb": {"density": 19.2, "fluorescence": 0.7435234},
        "Ga_Ka1": {"density": 8.3, "fluorescence": 0.7435234},
        "Mg_Ka2": {"density": 9.6, "fluorescence": None}
    },
    "incident_energy": 10.5,
    "detector_channel": "sum",
    "scaler_name": "i0",
    "distance_to_sample": 50.6,
    "creation_time_local": "2020-01-10T11:50:39+00:00",
    "source_scan_id": 92276,
    "source_scan_uid": "f07e3065-ab92-4b20-a702-ef61ed164dbc"

}


def _get_data_and_json_path(tmp_path):

    # Create some complicated path
    json_path = ["json", "param", "file"]
    file_name = "standard.yaml"
    json_path = os.path.join(tmp_path, *json_path, file_name)

    data = _xrf_standard_fluor_sample

    return data, json_path


def test_save_xrf_quant_fluor_json_file1(tmp_path):
    r"""Basic test of function 'save_xrf_standard_yaml_file' and 'load_xrf_standard_yaml_file'"""

    data, json_path = _get_data_and_json_path(tmp_path)

    # Sample data
    save_xrf_quant_fluor_json_file(json_path, data)

    data_loaded = load_xrf_quant_fluor_json_file(json_path)

    assert data_loaded == data, \
        "Loaded data is not equal to the original data"


def test_save_xrf_quant_fluor_json_file2(tmp_path):
    # 'save_xrf_quant_fluor_json_file' - overwrite existing file

    data, json_path = _get_data_and_json_path(tmp_path)

    # Create file
    save_xrf_quant_fluor_json_file(json_path, data)

    # Attempt to overwrite
    with pytest.raises(IOError, match=f"File '{json_path}' already exists"):
        save_xrf_quant_fluor_json_file(json_path, data)

    # Now overwrite the file by setting the flag 'overwrite_existing=True'
    save_xrf_quant_fluor_json_file(json_path, data, overwrite_existing=True)


def test_save_xrf_quant_fluor_json_file3(tmp_path):
    # 'save_xrf_quant_fluor_json_file' - invalid schema

    data, json_path = _get_data_and_json_path(tmp_path)

    data = copy.deepcopy(data)
    # Change the data so that it doesn't satisfy the schema
    data["incident_energy"] = "incident_energy"  # Supposed to be a number

    with pytest.raises(jsonschema.ValidationError):
        save_xrf_quant_fluor_json_file(json_path, data)


def test_save_xrf_quant_fluor_json_file4(tmp_path):
    r"""Schema allows some data fields to hold value of ``None``. Test if this works."""
    data, json_path = _get_data_and_json_path(tmp_path)

    # Modify some elements of the dictionary
    data = copy.deepcopy(data)
    data["detector_channel"] = None
    data["scaler_name"] = None
    data["distance_to_sample"] = None

    # Sample data
    save_xrf_quant_fluor_json_file(json_path, data)

    data_loaded = load_xrf_quant_fluor_json_file(json_path)

    assert data_loaded == data, \
        "Loaded data is not equal to the original data"


def test_load_xrf_quant_fluor_json_file1(tmp_path):
    # 'load_xrf_quant_fluor_json_file' - non-existing file

    _, json_path = _get_data_and_json_path(tmp_path)

    with pytest.raises(IOError, match=f"File '{json_path}' does not exist"):
        load_xrf_quant_fluor_json_file(json_path)


def test_load_xrf_quant_fluor_json_file2(tmp_path):
    # 'load_xrf_quant_fluor_json_file' - schema is not matched

    data, json_path = _get_data_and_json_path(tmp_path)

    # Create file
    save_xrf_quant_fluor_json_file(json_path, data)

    schema = copy.deepcopy(_xrf_quant_fluor_schema)
    # Modify schema, so that it is incorrect
    schema["properties"]["scaler_name"] = {"type": "number"}  # Supposed to be a string

    with pytest.raises(jsonschema.ValidationError):
        load_xrf_quant_fluor_json_file(json_path, schema=schema)


def test_get_quant_fluor_data_dict():
    f"""Tests for 'get_quant_fluor_data_dict': basic tests for consistensy of the returned dictionary"""

    for standard_data in _standard_data_sample:

        quant_fluor_data_dict = get_quant_fluor_data_dict(standard_data, incident_energy=12.0)
        # Will raise exception is schema is not satisfied
        jsonschema.validate(instance=quant_fluor_data_dict, schema=_xrf_quant_fluor_schema)

        assert quant_fluor_data_dict["name"] == standard_data["name"], \
            "Dictionary element 'name' is incorrect"

        assert quant_fluor_data_dict["serial"] == standard_data["serial"], \
            "Dictionary element 'serial' is incorrect"

        assert quant_fluor_data_dict["description"] == standard_data["description"], \
            "Dictionary element 'description' is incorrect"

        eline_set = set()
        # The 'mass' is not actual mass. If elements has multiple emission lines activated, then
        #   the mass (density) of the element will be counted multiple times. There is no
        #   physical meaning in the computed value: it is used to verify the sum of densities in
        #   the 'element_lines' dictionary of emission lines in 'quant_fluor_data_dict'
        mass_sum_expected = 0
        for cmpd, mass in standard_data["compounds"].items():
            em_dict = split_compound_mass(cmpd, mass)
            for el, ms in em_dict.items():
                elines = generate_eline_list([el], incident_energy=12.0)
                n_elines = len(elines)
                if n_elines:
                    mass_sum_expected += n_elines * ms
                    eline_set.update(elines)

        eline_out_list = list(quant_fluor_data_dict["element_lines"].keys())
        assert len(eline_out_list) == len(eline_set), "The number of emission lines is not as expected"
        assert set(eline_out_list) == eline_set, \
            "Generated object contains emission lines that are different from expected"

        mass_sum = sum([_["density"] for _ in quant_fluor_data_dict["element_lines"].values()])
        assert mass_sum == mass_sum_expected, \
            "The total mass (density) of the components is different from expected"


def test_fill_quant_fluor_data_dict():
    r"""Test for 'fill_quant_fluor_data_dict': testing basic functionality"""
    # Create copy, because the dictionary will be modified
    fluor_standard = copy.deepcopy(_xrf_standard_fluor_sample)

    # Create dictionary 'img' with XRF maps
    nx, ny = 10, 5
    map_S_K = np.random.rand(ny, nx)
    map_S_K_fluor = np.mean(map_S_K)

    # The line Fe_K is not present in the reference standard
    map_Fe_K = np.random.rand(ny, nx) * 3

    map_Au_M = np.random.rand(ny, nx) * 2
    map_Au_M_fluor = np.mean(map_Au_M)

    # Scaler
    v_sclr = 1.4
    map_sclr = np.ones(shape=(ny, nx), dtype=float) * 1.4

    img = {}
    img["S_K"] = map_S_K
    img["Fe_K"] = map_Fe_K
    img["Au_M"] = map_Au_M
    img["sclr"] = map_sclr

    # Fill the dictionary, use existing scaler 'sclr'
    fill_quant_fluor_data_dict(fluor_standard, xrf_map_dict=img, scaler_name="sclr")

    npt.assert_almost_equal(fluor_standard["element_lines"]["S_K"]["fluorescence"],
                            map_S_K_fluor / v_sclr,
                            err_msg=f"Fluorescence of 'S_K' is estimated incorrectly")
    npt.assert_almost_equal(fluor_standard["element_lines"]["Au_M"]["fluorescence"],
                            map_Au_M_fluor / v_sclr,
                            err_msg=f"Fluorescence of 'Au_M' is estimated incorrectly")
    for eline, param in fluor_standard["element_lines"].items():
        assert (eline in img) or (param["fluorescence"] is None), \
            f"Fluorescence line {eline} is not present in the dataset and it was not reset to None"

    # Fill the dictionary, use non-existing scaler 'abc'
    fill_quant_fluor_data_dict(fluor_standard, xrf_map_dict=img, scaler_name="abc")
    npt.assert_almost_equal(fluor_standard["element_lines"]["S_K"]["fluorescence"],
                            map_S_K_fluor,
                            err_msg=f"Fluorescence of 'S_K' is estimated incorrectly")

    # Fill the dictionary, don't use a scaler (set to None)
    fill_quant_fluor_data_dict(fluor_standard, xrf_map_dict=img, scaler_name=None)
    npt.assert_almost_equal(fluor_standard["element_lines"]["S_K"]["fluorescence"],
                            map_S_K_fluor,
                            err_msg=f"Fluorescence of 'S_K' is estimated incorrectly")


def test_prune_quant_fluor_data_dict():
    r"""Test for ``prune_quant_fluor_data_dict``: basic test of functionality"""

    # Create copy, because the dictionary will be modified
    fluor_standard = copy.deepcopy(_xrf_standard_fluor_sample)

    elines_not_none = []
    for eline, info in fluor_standard["element_lines"].items():
        set_to_none = np.random.rand() < 0.5  # 50% chance
        if set_to_none:
            info["fluorescence"] = None
        # For some emission lines, fluorescence could already be set to None
        if info["fluorescence"] is not None:
            elines_not_none.append(eline)

    fluor_standard_pruned = prune_quant_fluor_data_dict(fluor_standard)
    for eline, info in fluor_standard_pruned["element_lines"].items():
        assert eline in elines_not_none, \
            f"Emission line {eline} should have been removed from the dictionary"
        assert info["fluorescence"] is not None, \
            f"Emission line {eline} has 'fluorescence' set to None"


def test_set_quant_fluor_data_dict_optional_1():
    r"""
    Tests for 'set_quant_fluor_data_dict_optional': basic test of functionality.
    Successful tests.
    """

    # Create copy, because the dictionary will be modified
    fluor_standard = copy.deepcopy(_xrf_standard_fluor_sample)

    scan_id = 45378  # scan_id has to be int or a string representing int
    scan_uid = "abc-12345"  # Just some string, format is not checked

    set_quant_fluor_data_dict_optional(fluor_standard,
                                       scan_id=scan_id,
                                       scan_uid=scan_uid)

    # Check if scan_id and scan_uid are set
    assert fluor_standard["source_scan_id"] == scan_id, "Scan ID is set incorrectly"
    assert fluor_standard["source_scan_uid"] == scan_uid, "Scan UID is set incorrectly"

    # Check if time is set correctly
    t = fluor_standard["creation_time_local"]
    assert t is not None, "Time is not set"

    t = convert_time_from_nexus_string(t)
    t = ttime.mktime(t)
    t_current = ttime.mktime(ttime.localtime())

    # 5 seconds is more than sufficient to complete this test
    assert abs(t_current - t) < 5.0, "Time is set incorrectly"

    # Test if sending Scan ID as a string works
    scan_id2 = 45346
    scan_id2_str = f"{scan_id2}"
    set_quant_fluor_data_dict_optional(fluor_standard,
                                       scan_id=scan_id2_str)
    assert fluor_standard["source_scan_id"] == scan_id2, "Scan ID is set incorrectly"


def test_set_quant_fluor_data_dict_optional_2():
    r"""
    Tests for 'set_quant_fluor_data_dict_optional'.
    Failing tests.
    """

    # Create copy, because the dictionary will be modified
    fluor_standard = copy.deepcopy(_xrf_standard_fluor_sample)

    # Scan ID is a string, which can not be interpreted as int
    with pytest.raises(RuntimeError,
                       match="Parameter 'scan_id' must be integer or a string representing integer"):
        set_quant_fluor_data_dict_optional(fluor_standard, scan_id="abc_34g")

    # Scan ID is of wrong type
    with pytest.raises(RuntimeError,
                       match="Parameter 'scan_id' must be integer or a string representing integer"):
        set_quant_fluor_data_dict_optional(fluor_standard, scan_id=[1, 5, 14])

    # Scan UID is of wrong type
    with pytest.raises(RuntimeError,
                       match="Parameter 'scan_uid' must be a string representing scan UID"):
        set_quant_fluor_data_dict_optional(fluor_standard, scan_uid=[1, 5, 14])


def test_set_quant_fluor_data_dict_time():

    # Create copy, because the dictionary will be modified
    fluor_standard = copy.deepcopy(_xrf_standard_fluor_sample)

    set_quant_fluor_data_dict_time(fluor_standard)

    # Check if time is set correctly
    t = fluor_standard["creation_time_local"]
    assert t is not None, "Time is not set"

    t = convert_time_from_nexus_string(t)
    t = ttime.mktime(t)
    t_current = ttime.mktime(ttime.localtime())

    # 5 seconds is more than sufficient to complete this test
    assert abs(t_current - t) < 5.0, "Time is set incorrectly"
