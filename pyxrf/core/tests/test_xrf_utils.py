import pytest
import numpy as np
import numpy.testing as npt
from pyxrf.core.xrf_utils import parse_compound_formula, split_compound_mass


@pytest.mark.parametrize("formula, elements, n_atoms", [
    ("Fe2O3", ("Fe", "O"), (2, 3)),
    ("He", ("He",), (1,)),
    ("H2SO4", ("H", "S", "O"), (2, 1, 4)),
    ("C", ("C",), (1,)),
    ("C2H5OH", ("C", "H", "O"), (2, 6, 1))
])
def test_parse_compound_formula1(formula, elements, n_atoms):
    data = parse_compound_formula(formula)
    assert len(data) == len(elements), "The number of parsed elements is incorrect"
    assert set(elements) == set(data.keys()), "The set of parsed elements is incorrect"
    assert tuple(data[e]["nAtoms"] for e in elements) == n_atoms, \
        "The number of atoms in parsed data is determined incorrectly"


@pytest.mark.parametrize("formula, element_mass_fraction", [
    ("Fe2O3", {"Fe": 0.6994364433312461, "O": 0.30056355666875395}),
    ("C", {"C": 1.0})
])
def test_parse_compound_formula2(formula, element_mass_fraction):
    # Verify that mass fraction is found correctly
    data = parse_compound_formula(formula)
    assert len(data) == len(element_mass_fraction), \
        "The number of elements in the parsed data is incorrect"
    for e in element_mass_fraction.keys():
        assert e in data, f"Element {e} is not found in parsed data"
        npt.assert_almost_equal(data[e]["massFraction"], element_mass_fraction[e],
                                err_msg=f"Mass fraction for element {e} was evaluated incorrectly")


@pytest.mark.parametrize("formula", [
    "FE2O3", "fe2O3", "D", "Abc", ""
])
@pytest.mark.filterwarnings("ignore:")  # Ignore the warnings from XRayLib
def test_parse_compound_formula_fail(formula):
    with pytest.raises(RuntimeError, match=f"Invalid chemical formula.*{formula}"):
        parse_compound_formula(formula)


@pytest.mark.parametrize("formula, elements, n_atoms", [
    ("Fe2O3", ("Fe", "O"), (2, 3)),
    ("He", ("He",), (1,)),
    ("H2SO4", ("H", "S", "O"), (2, 1, 4)),
    ("C", ("C",), (1,))
])
def test_split_compound_mass(formula, elements, n_atoms):

    mass_total = 10.0

    data = split_compound_mass(formula, mass_total)
    # Basic checks for proper parsing
    assert len(data) == len(elements), "The number of parsed elements is incorrect"
    assert set(elements) == set(data.keys()), "The set of parsed elements is incorrect"
    # Make sure that the sum of mass of each element equals to total mass
    npt.assert_almost_equal(
        np.sum(list(data.values())), mass_total,
        err_msg="The computed mass is not distributed properly among elements")


@pytest.mark.parametrize("formula", [
    "FE2O3", "fe2O3", "D", "Abc", ""
])
@pytest.mark.filterwarnings("ignore:")  # Ignore the warnings from XRayLib
def test_split_compound_mass_fail(formula):
    with pytest.raises(RuntimeError, match=f"Invalid chemical formula.*{formula}"):
        split_compound_mass(formula, 10.0)
