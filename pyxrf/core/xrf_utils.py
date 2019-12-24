import xraylib


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

    xraylib.SetErrorMessages(0)  # This is supposed to stop XRayLib from printing
    #                              internal error messages, but it doesn't work

    try:
        compound_data = xraylib.CompoundParser(compound_formula)
    except SystemError:
        msg = f"Invalid chemical formula '{compound_formula}' is passed, parsing failed"
        raise RuntimeError(msg)

    # Now create more manageable structure
    compound_dict = {}
    for e_an, e_mf, e_na in zip(compound_data["Elements"],
                                compound_data["massFractions"],
                                compound_data["nAtoms"]):
        e_name = xraylib.AtomicNumberToSymbol(e_an)
        compound_dict[e_name] = {"AtomicNumber": e_an,
                                 "nAtoms": e_na,
                                 "massFraction": e_mf}

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
