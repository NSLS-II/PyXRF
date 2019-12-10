import pytest
import numpy as np
from pyxrf.xanes_maps.xanes_maps_api import (
    build_xanes_map_api, _parse_docstring_parameters, _verify_parsed_docstring,
    _build_xanes_map_param_default)

# ---------------------------------------------------------------------------
#   Testing _parse_docstring_parameters


def _generate_sample_docstring():
    """
    Generates sample docstring, returns docstring and the list of parameter/description pairs.
    For testing of ``_parse_docstring_parameters`` function.
    """
    parameters = [
        ("parameter_1",
         ["parameter_1 : str", "Description of parameter 1", "",
          "The end of the description."]),
        ("_parameter_2",
         ["_parameter_2 : str", "Description of parameter 2", "",
          "The end of the description of parameter 2."]),
        ("_parameter__3",
         ["_parameter__3 : str", "Description of parameter 3", "",
          "The end of the description of parameter 3."])
    ]

    n_empty_lines_before, n_empty_lines_after = 5, 5
    d_str = [""] * n_empty_lines_before

    d_str.append("    Parameters")
    d_str.append("    ----------")

    for p in parameters:
        # Indentation by 4 spaces
        st = [f"    {s}" if s else s for s in p[1]]
        s = "\n".join(st)
        d_str.append(s)

    d_str.append("    Returns")
    d_str.append("    -------")

    d_str.extend([""] * n_empty_lines_after)

    d_str = "\n".join(d_str)  # Convert the list to a single string

    return d_str, parameters


def test_parse_docstring_parameters():
    # Simple test for the successfully parsed docstring. It seems sufficient, since all error cases are trivial.

    d_str, parameters = _generate_sample_docstring()
    parameters_output = _parse_docstring_parameters(d_str)
    assert parameters == parameters_output, "Parsed parameters or parameter descriptions are invalid"


def test_verify_parsed_docstring():

    # Generate the set of parameters (we don't use docstring in this test)
    _, parameters = _generate_sample_docstring()

    param_dict = {}
    for p_name, _ in parameters:
        param_dict[p_name] = 0  # Value doesn't matter here (it is not used for verification

    # This verification should be successful
    parameters_copy = parameters.copy()
    param_dict_copy = param_dict.copy()
    _verify_parsed_docstring(parameters, param_dict)  # This may raise an exception
    assert parameters == parameters_copy, "'parameters' was unintentionally changed by the function"
    assert param_dict == param_dict_copy, "'param_dict' was unintentionally changed by the function"

    # This test should fail (2 extra parameters)
    param_dict2 = param_dict.copy()
    param_dict2["extra_parameter1"] = 0
    param_dict2["extra_parameter2"] = 0

    parameters_copy = parameters.copy()
    param_dict_copy = param_dict2.copy()
    with pytest.raises(AssertionError, match="not found in the docstring.+extra_parameter1.+extra_parameter2"):
        _verify_parsed_docstring(parameters, param_dict2)
    assert parameters == parameters_copy, "'parameters' was unintentionally changed by the function"
    assert param_dict2 == param_dict_copy, "'param_dict' was unintentionally changed by the function"

    # This test should fail (1 parameter is removed)
    param_dict2 = param_dict.copy()
    # Select random key for removal
    key_to_remove = list(param_dict2.keys())[np.random.choice(len(param_dict2))]
    del param_dict2[key_to_remove]

    parameters_copy = parameters.copy()
    param_dict_copy = param_dict2.copy()
    with pytest.raises(AssertionError, match=f"not in the dictionary.+{key_to_remove}"):
        _verify_parsed_docstring(parameters, param_dict2)
    assert parameters == parameters_copy, "'parameters' was unintentionally changed by the function"
    assert param_dict2 == param_dict_copy, "'param_dict' was unintentionally changed by the function"


def test_parse_build_xanes_map_docstring():
    """ Test that the docstring of ``build_xanes_map_api`` and ``_build_xanes_map_param_default``
    are consistent: parse the docstring and match with the dictionary"""
    parameters = _parse_docstring_parameters(build_xanes_map_api.__doc__)
    _verify_parsed_docstring(parameters, _build_xanes_map_param_default)
