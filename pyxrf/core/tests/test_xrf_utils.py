import re

import numpy as np
import numpy.testing as npt
import pytest

from pyxrf.core.xrf_utils import (
    check_if_eline_is_activated,
    check_if_eline_supported,
    compute_atomic_scaling_factor,
    compute_atomic_weight,
    compute_cs,
    compute_cs_ratio,
    generate_eline_list,
    get_element_atomic_number,
    get_supported_eline_list,
    parse_compound_formula,
    split_compound_mass,
    validate_element_str,
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


# fmt: off
@pytest.mark.parametrize("ZZ, incident_energy, line, cs_value", [
    (31, 12.0, "K", 7924.59),
    (31, 12.0, "L", 279.98),
    (82, 12.0, "M", 713.19),
])
# fmt: on
def test_compute_cs_01(ZZ, incident_energy, line, cs_value):
    """
    ``compute_cs``: basic test
    """
    cs = compute_cs(ZZ, incident_energy, line=line)
    npt.assert_equal(cs, cs_value)


def test_compute_cs_02_fail():
    """
    ``compute_cs``: basic test
    """
    ZZ, line = 31, "K"
    incident_energy = 1  # Too small
    with pytest.raises(ValueError, match="Failed to compute cross section for an element"):
        compute_cs(ZZ, incident_energy, line=line)

    line2 = "A"  # non-existing
    with pytest.raises(ValueError, match="Unrecognized emission line"):
        compute_cs(ZZ, incident_energy, line=line2)


# fmt: off
@pytest.mark.parametrize("eline1, eline2, incident_energy, cs_ratio", [
    ("Cu_K", "Ga_K", 12.0, 0.705789),
    ("Cu_K", "Ga_L", 12.0, 19.976748),
])
# fmt: on
def test_compute_cs_ratio_01(eline1, eline2, incident_energy, cs_ratio):
    csr = compute_cs_ratio(eline1, eline2, incident_energy)
    npt.assert_almost_equal(csr, cs_ratio, 5)


# fmt: off
@pytest.mark.parametrize("eline1, eline2, incident_energy", [
    ("Cu", "Ga_K", 12.0),
    ("Cu_K", "Ga", 12.0),
    ("Ab_K", "Ga_K", 12.0),
    ("Cu_K", "Ab_K", 12.0),
    ("Cu_Z", "Ga_K", 12.0),
    ("Cu_K", "Ga_Z", 12.0),
])
# fmt: on
def test_compute_cs_ratio_02_fail(eline1, eline2, incident_energy):
    with pytest.raises(ValueError, match="Invalid emission line"):
        compute_cs_ratio(eline1, eline2, incident_energy)


# fmt: off
@pytest.mark.parametrize("element, weight", [
    ("Cu_K", 63.54),
    ("Cu", 63.54),
])
# fmt: on
def test_compute_atomic_weight_01(element, weight):
    aw = compute_atomic_weight(element)
    npt.assert_almost_equal(aw, weight)


# fmt: off
@pytest.mark.parametrize("ref_eline, quant_eline, incident_energy, factor", [
    ("Cu_K", "Ga_K", 12.0, 0.77443536),
    ("Cu_K", "Ga_L", 12.0, 21.919718),
])
# fmt: on
def test_compute_atomic_scaling_factor_01(ref_eline, quant_eline, incident_energy, factor):
    sf = compute_atomic_scaling_factor(ref_eline, quant_eline, incident_energy)
    npt.assert_almost_equal(sf, factor, 5)


# fmt: off
@pytest.mark.parametrize("ref_eline, quant_eline, incident_energy, msg", [
    ("Cu_K", "Ga_M", 12.0, "Failed to compute cross section for an element"),
    ("Ga_M", "Cu_K", 12.0, "Failed to compute cross section for an element"),
    ("Ab_K", "Ga_L", 12.0, "Failed to compute atomic weight for the element"),
    ("Ga_L", "Ab_K", 12.0, "Failed to compute atomic weight for the element"),
    ("Cu_Z", "Ga_L", 12.0, "Invalid emission line"),
    ("Ga_L", "Cu_Z", 12.0, "Invalid emission line"),
])
# fmt: on
def test_compute_atomic_scaling_factor_02_fail(ref_eline, quant_eline, incident_energy, msg):
    with pytest.raises(ValueError, match=msg):
        compute_atomic_scaling_factor(ref_eline, quant_eline, incident_energy)
