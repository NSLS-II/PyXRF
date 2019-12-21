import pytest
import numpy as np
import numpy.testing as npt
from pyxrf.model.utils import parse_compound_formula, split_compound_mass


@pytest.mark.parametrize("formula, elements, n_atoms", [
    ("Fe2O3", ("Fe", "O"), (2, 3)),
    ("He", ("He",), (1,)),
    ("H2SO4", ("H", "S", "O"), (2, 1, 4)),
    ("C", ("C",), (1,))
])
def test_parse_compound_formula(formula, elements, n_atoms):
    data = parse_compound_formula(formula)
    assert len(data) == len(elements), "The number of parsed elements is incorrect"
    assert set(elements) == set(data.keys()), "The set of parsed elements is incorrect"
    assert tuple(data[e]["nAtoms"] for e in elements) == n_atoms, \
        "The number of atoms in parsed data is determined incorrectly"


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
