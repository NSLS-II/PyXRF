import re
import xraylib
from distutils.version import LooseVersion

from skbeam.core.constants import XrfElement as Element
from skbeam.core.fitting.xrf_model import K_LINE, L_LINE, M_LINE


def get_element_atomic_number(element_str):
    r"""
    A wrapper to ``SymbolToAtomicNumber`` function from ``xraylib``.
    Returns atomic number for the sybmolic element name (e.g. ``C`` or ``Fe``).

    Parameters
    ----------

    element_str: str
        sybmolic representation of an element

    Returns
    -------

    Atomic number of the element ``element_str``. If element is invalid, then
    the function returns 0.

    """
    if LooseVersion(xraylib.__version__) < LooseVersion("4.0.0"):
        xraylib.SetErrorMessages(0)  # Turn off error messages from ``xraylib``

    try:
        val = xraylib.SymbolToAtomicNumber(element_str)
    except ValueError:
        # Imitate the behavior of xraylib 3
        val = 0
    return val


def validate_element_str(element_str):
    r"""
    Checks if ``element_str`` is valid representation of an element according to
    standard notation for chemical formulas. Valid representation of elements can
    be processed by ``xraylib`` tools. This functions attempts to find the atomic
    number for the element and returns ``True`` if it succeeds and ``False`` if
    it fails.

    Parameters
    ----------

    element_str: str
        sybmolic representation of an element

    Returns
    -------

    ``True`` if ``element_str`` is valid representation of an element and ``False``
    otherwise.
    """

    if get_element_atomic_number(element_str):
        return True
    else:
        return False


def parse_compound_formula(compound_formula):
    r"""
    Parses the chemical formula of a compound and returns the dictionary,
    which contains element name, atomic number, number of atoms and mass fraction
    in the compound.

    Parameters
    ----------

    compound_formula: str
        chemical formula of the compound in the form ``FeO2``, ``CO2`` or ``Fe``.
        Element names must start with capital letter.

    Returns
    -------

        dictionary of dictionaries, data on each element in the compound: key -
        sybolic element name, value - a dictionary that contains ``AtomicNumber``,
        ``nAtoms`` and ``massFraction`` of the element. The elements are sorted
        in the order of growing atomic number.

    Raises
    ------

        RuntimeError is raised if compound formula cannot be parsed
    """

    if LooseVersion(xraylib.__version__) < LooseVersion("4.0.0"):
        xraylib.SetErrorMessages(0)  # This is supposed to stop XRayLib from printing
        #                              internal error messages, but it doesn't work

    try:
        compound_data = xraylib.CompoundParser(compound_formula)
    except (SystemError, ValueError):
        msg = f"Invalid chemical formula '{compound_formula}' is passed, parsing failed"
        raise RuntimeError(msg)

    # Now create more manageable structure
    compound_dict = {}
    for e_an, e_mf, e_na in zip(
        compound_data["Elements"], compound_data["massFractions"], compound_data["nAtoms"]
    ):
        e_name = xraylib.AtomicNumberToSymbol(e_an)
        compound_dict[e_name] = {"AtomicNumber": e_an, "nAtoms": e_na, "massFraction": e_mf}

    return compound_dict


def split_compound_mass(compound_formula, compound_mass):
    r"""
    Computes mass of each element in the compound given total mass of the compound

    Parameters
    ----------

    compound_formula: str
        chemical formula of the compound in the form ``FeO2``, ``CO2`` or ``Fe``.
        Element names must start with capital letter.

    compound_mass: float
        total mass of the compound

    Returns
    -------

        dictionary: key - symbolic element name, value - mass of the element

    Raises
    ------

        RuntimeError is raised if compound formula cannot be parsed
    """

    compound_dict = parse_compound_formula(compound_formula)

    element_dict = {}
    for el_name, el_info in compound_dict.items():
        element_dict[el_name] = el_info["massFraction"] * compound_mass

    return element_dict


def get_supported_eline_list(*, lines=None):
    """
    Returns the list of the emission lines supported by ``scikit-beam``

    Parameters
    ----------

    lines : list(str)
        tuple or list of strings, that defines, which emission lines are going to be included
        in the output list (e.g. ``("K",)`` or ``("L", "M")`` etc.) If ``None`` (default),
        then K, L and M lines are going to be included.

    Returns
    -------
        the list of supported emission line. The lines are in the format ``"Fe_K"`` or ``"Mg_M"``.
    """

    if lines is None:
        lines = ("K", "L", "M")

    eline_list = []
    if "K" in lines:
        eline_list += K_LINE
    if "L" in lines:
        eline_list += L_LINE
    if "M" in lines:
        eline_list += M_LINE

    return eline_list


def check_if_eline_supported(eline_name, *, lines=None):
    """
    Check if the emission line name is in the list of supported names.
    Emission name must be in the format: K_K, Fe_K etc. The list includes K, L and M lines.
    The function is case-sensitive.

    Parameters
    ----------
    eline_name : str
        name of the emission line (K_K, Fe_K etc. for valid emission line). In general
        the string may contain arbitrary sequence characters, may be empty or None. The
        function will return True only if the sequence represents emission line from
        the list of supported emission lines.

    lines : list(str)
        tuple or list of strings, that defines, which emission lines are going to be included
        in the output list (e.g. ``("K",)`` or ``("L", "M")`` etc.) If ``None`` (default),
        then K, L and M lines are going to be included.

    Returns
        True if ``eline_name`` is in the list of supported emission lines, False otherwise
    """
    if not eline_name or not isinstance(eline_name, str):
        return False

    if eline_name in get_supported_eline_list(lines=lines):
        return True
    else:
        return False


def check_if_eline_is_activated(elemental_line, incident_energy):
    """
    Checks if emission line is activated at given incident beam energy

    Parameters
    ----------

    elemental_line : str
        emission line in the format K_K or Fe_K
    incident_energy : float
        incident energy in keV

    Returns
    -------
        bool value, indicating if the emission line is activated
    """

    # Check if the emission line has correct format
    if not re.search(r"^[A-Z][a-z]?_[KLM]([ab]\d?)?$", elemental_line):
        raise RuntimeError(f"Elemental line {elemental_line} is improperly formatted")

    # The validation of 'elemental_line' is strict enough to do the rest of the processing
    #   without further checks.
    [element, line] = elemental_line.split("_")
    line = line.lower()
    if len(line) == 1:
        line += "a1"
    elif len(line) == 2:
        line += "1"

    e = Element(element)
    if e.cs(incident_energy)[line] == 0:
        return False
    else:
        return True


def generate_eline_list(element_list, *, incident_energy, lines=None):
    r"""
    Generate a list of emission lines based on the list of elements (``element_list``)
    and incident energy. Only the emission lines that are supported by ``scikit-beam``
    and activated by the incident energy are included in the list.

    Parameters
    ----------

    element_list: list(str)
        list of valid element names (e.g. ["S", "Al", "Fe"])

    incident_energy: float
        incident beam energy, keV

    lines: list(str)
        tuple or list of strings, that defines, which classes of emission lines to include
        in the output list (e.g. ``("K",)`` or ``("L", "M")`` etc.) If ``None`` (default),
        then K, L and M lines are going to be included.

    Returns
    -------

    list(str) - the list of emission lines

    Raises
    ------

    RuntimeError is raised if 'lines' contains incorrect specification of emission line class.
    """

    if lines is None:
        # By default lines "K", "L" and "M" are included in the output list
        lines = ("K", "L", "M")

    # Verify line selection
    invalid_lines = []
    for ln in lines:
        if not re.search(r"^[KLM]$", ln):
            invalid_lines.append(ln)
    if invalid_lines:
        msg = f"Some of the selected emission lines are incorrect: {invalid_lines}"
        raise RuntimeError(msg)

    eline_list = []

    for element in element_list:
        for ln in lines:
            eline = f"{element}_{ln}"
            is_activated = check_if_eline_is_activated(eline, incident_energy)
            is_supported = check_if_eline_supported(eline)
            if is_activated and is_supported:
                eline_list.append(eline)

    return eline_list


# TODO: the following function needs tests
def get_eline_parameters(elemental_line, incident_energy):
    """
    Returns emission line parameters

    Parameters
    ----------

    elemental_line : str
        emission line in the format K_K, Fe_K, Ca_k, Ca_ka, Ca_kb2 etc.
    incident_energy : float
        incident energy in keV

    Returns
    -------
    dict
        Computed parameters of the emission line. Keys: "energy" (central energy of the peak),
        "cs" (crossection), "ratio" (normalized crossection, e.g. cs(Ca_kb2)/ cs(Ca_ka1)
    """

    # Check if the emission line has correct format
    # TODO: verify the correnct range for lines (a-z covers all the cases, but may be too broad)
    if not re.search(r"^[A-Z][a-z]?_[KLMklm]([a-z]\d?)?$", elemental_line):
        raise RuntimeError(f"Elemental line {elemental_line} is improperly formatted")

    # The validation of 'elemental_line' is strict enough to do the rest of the processing
    #   without further checks.
    [element, line] = elemental_line.split("_")
    line = line.lower()
    if len(line) == 1:
        line += "a1"
    elif len(line) == 2:
        line += "1"

    # This is the name of line #1 (ka1, la1 etc.)
    line_1 = line[0] + "a1"

    try:
        e = Element(element)
        energy = e.emission_line[line]
        cs = e.cs(incident_energy)[line]
        cs_1 = e.cs(incident_energy)[line_1]
        ratio = cs / cs_1 if cs_1 else 0
    except Exception:
        energy, cs, ratio = 0, 0, 0

    return {"energy": energy, "cs": cs, "ratio": ratio}


# TODO: the following function needs tests
def get_eline_energy(elemental_line):
    """
    Returns emission line parameters

    Parameters
    ----------

    elemental_line : str
        emission line in the format K_K, Fe_K, Ca_k, Ca_ka, Ca_kb2 etc.

    Returns
    -------
    float
        Peak center energy for the emission line, keV.
    """

    # Check if the emission line has correct format
    # TODO: verify the correnct range for lines (a-z covers all the cases, but may be too broad)
    if not re.search(r"^[A-Z][a-z]?_[KLMklm]([a-z]\d?)?$", elemental_line):
        raise RuntimeError(f"Elemental line {elemental_line} is improperly formatted")

    # The validation of 'elemental_line' is strict enough to do the rest of the processing
    #   without further checks.
    [element, line] = elemental_line.split("_")
    line = line.lower()
    if len(line) == 1:
        line += "a1"
    elif len(line) == 2:
        line += "1"

    try:
        e = Element(element)
        energy = e.emission_line[line]
    except Exception:
        energy = 0

    return energy
