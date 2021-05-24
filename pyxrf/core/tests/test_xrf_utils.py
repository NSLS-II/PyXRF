import pytest
import re
import numpy as np
import numpy.testing as npt
from pyxrf.core.xrf_utils import (
    get_element_atomic_number,
    validate_element_str,
    parse_compound_formula,
    split_compound_mass,
    get_supported_eline_list,
    check_if_eline_supported,
    check_if_eline_is_activated,
    generate_eline_list,
)


# fmt: off
@pytest.mark.parametrize("element_number", [
    ("He", 2), ("H", 1), ("Fe", 26),
    # Failing cases, 0 is returned
    ("he", 0), ("h", 0), ("Fe2O3", 0), ("50", 0), ("Fe ", 0), ("", 0)
])
# fmt: on
def test_get_element_atomic_number(element_number):

    element_str, atomic_number = element_number
    assert (
        get_element_atomic_number(element_str) == atomic_number
    ), "Atomic number returned by the function is incorrect"


# fmt: off
@pytest.mark.parametrize("element_valid", [
    ("He", True), ("H", True), ("Fe", True),
    # Failing cases
    ("he", False), ("h", False), ("Fe2O3", False), ("50", False), ("Fe ", False), ("", False)
])
# fmt: on
def test_validate_element_str(element_valid):

    element_str, is_valid = element_valid
    assert validate_element_str(element_str) == is_valid, "Element validation is not successful"


# fmt: off
@pytest.mark.parametrize("formula, elements, n_atoms", [
    ("Fe2O3", ("Fe", "O"), (2, 3)),
    ("He", ("He",), (1,)),
    ("H2SO4", ("H", "S", "O"), (2, 1, 4)),
    ("C", ("C",), (1,)),
    ("C2H5OH", ("C", "H", "O"), (2, 6, 1))
])
# fmt: on
def test_parse_compound_formula1(formula, elements, n_atoms):
    data = parse_compound_formula(formula)
    assert len(data) == len(elements), "The number of parsed elements is incorrect"
    assert set(elements) == set(data.keys()), "The set of parsed elements is incorrect"
    assert (
        tuple(data[e]["nAtoms"] for e in elements) == n_atoms
    ), "The number of atoms in parsed data is determined incorrectly"


# fmt: off
@pytest.mark.parametrize("formula, element_mass_fraction", [
    ("Fe2O3", {"Fe": 0.6994364433312461, "O": 0.30056355666875395}),
    ("C", {"C": 1.0})
])
# fmt: on
def test_parse_compound_formula2(formula, element_mass_fraction):
    # Verify that mass fraction is found correctly
    data = parse_compound_formula(formula)
    assert len(data) == len(element_mass_fraction), "The number of elements in the parsed data is incorrect"
    for e in element_mass_fraction.keys():
        assert e in data, f"Element {e} is not found in parsed data"
        npt.assert_almost_equal(
            data[e]["massFraction"],
            element_mass_fraction[e],
            err_msg=f"Mass fraction for element {e} was evaluated incorrectly",
        )


# fmt: off
@pytest.mark.parametrize("formula", [
    "FE2O3", "fe2O3", "D", "Abc", ""
])
@pytest.mark.filterwarnings("ignore:")  # Ignore the warnings from XRayLib
# fmt: on
def test_parse_compound_formula_fail(formula):
    with pytest.raises(RuntimeError, match=f"Invalid chemical formula.*{formula}"):
        parse_compound_formula(formula)


# fmt: off
@pytest.mark.parametrize("formula, elements, n_atoms", [
    ("Fe2O3", ("Fe", "O"), (2, 3)),
    ("He", ("He",), (1,)),
    ("H2SO4", ("H", "S", "O"), (2, 1, 4)),
    ("C", ("C",), (1,))
])
# fmt: on
def test_split_compound_mass(formula, elements, n_atoms):

    mass_total = 10.0

    data = split_compound_mass(formula, mass_total)
    # Basic checks for proper parsing
    assert len(data) == len(elements), "The number of parsed elements is incorrect"
    assert set(elements) == set(data.keys()), "The set of parsed elements is incorrect"
    # Make sure that the sum of mass of each element equals to total mass
    npt.assert_almost_equal(
        np.sum(list(data.values())),
        mass_total,
        err_msg="The computed mass is not distributed properly among elements",
    )


# fmt: off
@pytest.mark.parametrize("formula", [
    "FE2O3", "fe2O3", "D", "Abc", ""
])
@pytest.mark.filterwarnings("ignore:")  # Ignore the warnings from XRayLib
# fmt: on
def test_split_compound_mass_fail(formula):
    with pytest.raises(RuntimeError, match=f"Invalid chemical formula.*{formula}"):
        split_compound_mass(formula, 10.0)


# --------------------------------------------------------------------------------------


def test_get_supported_eline_list():

    list_k = get_supported_eline_list(lines=("K",))
    list_l = get_supported_eline_list(lines=("L",))
    list_m = get_supported_eline_list(lines=("M",))

    list_kl = get_supported_eline_list(lines=("K", "L"))
    list_lm = get_supported_eline_list(lines=("L", "M"))

    list_klm = get_supported_eline_list(lines=("K", "L", "M"))
    list_default = get_supported_eline_list()

    # Check eline formatting
    for v in list_klm:
        assert re.search(r"[A-Z][a-z]?_[KLM]", v), f"Emission line name {v} is not properly formatted"

    assert (
        (len(list_k) > 0) and (len(list_l) > 0) and (len(list_m) > 0)
    ), "At least one of the lists for K, L or M lines is empty"
    assert list_klm == list_default, "The complete list of emission lines is incorrectly assembled"
    assert list_klm == list_k + list_l + list_m, "The complete list does not include all for K, L and M lines"
    assert list_kl == list_k + list_l, "The list for K and L lines is not equivalent to the sum of the lists"
    assert list_lm == list_l + list_m, "The list for K and L lines is not equivalent to the sum of the lists"


# fmt: off
@pytest.mark.parametrize("eline, success", [
    ("Fe_K", True),
    ("W_L", True),
    ("Ta_M", True),
    ("Ab_K", False),
    ("Fe_A", False),
    ("", False),
    (None, False)
])
# fmt: on
def test_check_if_eline_supported(eline, success):
    assert check_if_eline_supported(eline) == success, f"Emission line {eline} is indentified incorrectly"


# fmt: off
@pytest.mark.parametrize("eline, incident_energy, success", [
    ("Fe_K", 8.0, True),
    ("Fe_K", 6.0, False),
    ("W_L", 12.0, True),
    ("W_L", 8.0, False),
    ("Ta_M", 2.0, True),
    ("Ta_M", 1.0, False),

])
# fmt: on
def test_check_if_eline_is_activated(eline, incident_energy, success):
    assert (
        check_if_eline_is_activated(eline, incident_energy) == success
    ), f"Activation status for the emission line {eline} at {incident_energy} keV is {success}"


# fmt: off
@pytest.mark.parametrize("elements, incident_energy, elines", [
    (["Fe", "W", "Ta"], 12.0, ['Fe_K', 'W_L', 'W_M', 'Ta_L', 'Ta_M']),
    (["Fe", "W", "Ta"], 6.0, ['W_M', 'Ta_M']),
])
# fmt: on
def test_generate_eline_list1(elements, incident_energy, elines):
    r"""
    ``generate_eline_list``: search all lines
    """
    assert (
        generate_eline_list(elements, incident_energy=incident_energy) == elines
    ), "Emission line list is generated incorrectly"


# fmt: off
@pytest.mark.parametrize("elements, incident_energy, lines, elines", [
    (["Fe", "W", "Ta"], 12.0, ("K", "L", "M"), ['Fe_K', 'W_L', 'W_M', 'Ta_L', 'Ta_M']),
    (["Fe", "W", "Ta"], 12.0, ("K",), ['Fe_K']),
    (["Fe", "W", "Ta"], 12.0, ("L",), ['W_L', 'Ta_L']),
    (["Fe", "W", "Ta"], 12.0, ("M",), ['W_M', 'Ta_M']),
    (["Fe", "W", "Ta"], 12.0, ("K", "L"), ['Fe_K', 'W_L', 'Ta_L']),
    (["Fe", "W", "Ta"], 12.0, ("K", "M"), ['Fe_K', 'W_M', 'Ta_M']),
    (["Fe", "W", "Ta"], 12.0, ("L", "M"), ['W_L', 'W_M', 'Ta_L', 'Ta_M']),
])
# fmt: on
def test_generate_eline_list2(elements, incident_energy, lines, elines):
    r"""
    ``generate_eline_list``: explicitely select eline categories
    """
    assert (
        generate_eline_list(elements, incident_energy=incident_energy, lines=lines) == elines
    ), "Emission line list is generated incorrectly"


def test_generate_eline_list3():
    r"""
    ``generate_eline_list``: fails if emission line category is specified incorrectly
    """
    with pytest.raises(RuntimeError, match="Some of the selected emission lines are incorrect"):
        generate_eline_list(["Fe", "W", "Ta"], incident_energy=12.0, lines=["K", "Ka"])
