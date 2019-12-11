import yaml
import re
import os


def _parse_docstring_parameters(doc_string):
    r"""
    Parses Google-style docstring and returns the list of (``parameter_name``, ``parameter_description``)
    pairs. The properly composed parameter section of the docstring should start with the line
    containing the word ``    Parameters`` followed by the line containing ``    ----------`` and
    end with the line containing the word ``    Returns`` followed by the line containing ``    -------``.
    The parameter descriptions must start with the line containing parameter name and type:
    ``   parameter_name : type``. Parameter name may contain symbols '_', 'A-Z', 'a-z', '0'-'9' and
    start with a letter or '_'. Every line in parameter description section must be empty or
    start with FOUR spaces. The lines that contain description text may contain additional spaces.

    This function:

    - finds the parameter description section;

    - removes 4 spaces from the beginning of each line;

    - finds parameter names and descriptions for each parameter.

    Parameters
    ----------

    doc_string : str
        doc string as return by ``some_function.__doc__``

    Returns
    -------

        A list of tuples ``(parameter_name, parameter_description)``:

        - ``parameter_name`` is a string

        - ``parameter_description`` is a list of strings
    """

    str_list = doc_string.split('\n')

    # Remove all spaces at the end of the strings (the should be no spaces there, but still)
    str_list = [s.rstrip() for s in str_list]

    # We are interested only in the part of the docstring that contains description of parameters
    #   Google-style docstrings are expected
    n_first, n_last = None, None
    for n in range(1, len(str_list) - 1):
        if (str_list[n - 1] == "    Parameters") and re.search(r"^    -+$", str_list[n]):
            n_first = n + 1
        if (str_list[n] == "    Returns") and re.search(r"^    -+$", str_list[n + 1]):
            n_last = n - 1
            break

    assert (n_first is not None) or (n_last is not None), \
        "Incorrect docstring format: 'Parameters' or 'Return' statement was not found in the docstring"

    # The list of strings contains parameter descriptions
    str_list = str_list[n_first:n_last + 1]
    # Each line must start with 4 spaces or be empty. Verify this
    assert all([(not s) or re.search(r"^    ", s) for s in str_list]), \
        "Incorrect docstring format: parameter descriptions should be indented by at least FOUR spaces"
    # Now remove the spaces from nonempty lines
    str_list = [s[4:] if s else s for s in str_list]

    param_pos = [n for n, s in enumerate(str_list) if re.search(r"^[_A-Za-z][_A-Za-z0-9]* :", s)]
    n_parameters = len(param_pos)
    assert n_parameters, "Incorrect docstring format: no parameters were found"

    param_pos.append(len(str_list))  # Having the last index will be helpful

    param_names = [re.search("[_A-Za-z][_A-Za-z0-9]*", str_list[param_pos[n]])[0]
                   for n in range(n_parameters)]
    param_descriptions = [str_list[param_pos[n]: param_pos[n + 1]] for n in range(n_parameters)]

    # Remove empty strings from the end of each description (if any)
    for pd in param_descriptions:
        while pd and (not pd[-1]):
            pd.pop(-1)

    # Check if some of the parameter has no descriptions (the number of line must be > 1)
    #   The fist line of the description is actually the
    assert all([len(s) > 1 for s in param_descriptions]), \
        "Incomplete docstring: some parameters have not descriptions"

    params = list(zip(param_names, param_descriptions))

    return params


def _verify_parsed_docstring(parameters, param_dict):
    """
    Verification of consistency of the parameters extracted from docstring and
    dictionary of parameters (probably default parameters). There must be one-to-one
    match between the entries of the list of the parameters extracted from the docstring
    and the parameter dictionary.

    Parameters
    ----------

    parameters : list(tuple)
        The list of ``(parameter_name, parameter_description)`` tuples.
        Descriptions are not analyzed by this function.

    param_dict : dict
        The dictionary of parameters: key - parameter name, value - default value.
        Values are not analyized by this function.

    Raises exception if there is mismatch between the parameter sets.
    """

    pd = param_dict.copy()
    missing_param_list = []
    for p_name, _ in parameters:
        if p_name in pd:
            del pd[p_name]
        else:
            missing_param_list.append(p_name)

    extra_param_list = pd.keys()

    err_msg_list = []
    if missing_param_list:
        err_msg_list.append(f"parameters that are in docstring, but not in the dictionary {missing_param_list}")
    if extra_param_list:
        err_msg_list.append(f"dictionary parameters that are not found in the docstring: {extra_param_list}")

    if err_msg_list:
        err_msg = "Parsed parameter verification failed: parameter set is inconsistent:\n"
        err_msg += ", ".join(err_msg_list)
        assert False, err_msg


def create_yaml_parameter_file(*, file_path, function_docstring, param_value_dict,
                               dir_create=False, file_overwrite=False):
    """
    Creates YAML parameter file based on parameter names and descriptions from ``parameters``
    and default values from ``param_value_dict``. The file is supposed to have simple
    human-readable and editable structure.

    The function should be used to create YAML file with default parameter values that are later
    modified by users according to their needs.

    TODO: edit this docstring
    """

    # Convert to absolute path
    file_path = os.path.expanduser(file_path)
    file_path = os.path.abspath(file_path)

    # Parse docstring (returns the list of param_name/param_description pairs)
    parameters = _parse_docstring_parameters(function_docstring)

    # Check if entries in 'parameters' and 'param_value_dict' match (will raise an exception).
    _verify_parsed_docstring(parameters, param_value_dict)

    # Check if file already exists
    file_exists = False
    if not file_overwrite:
        file_exists = os.path.exists(file_path)
    else:
        # We would like to overwrite only files, not directories etc.
        file_exists = os.path.exists(file_path) and not os.path.isfile(file_path)

    if file_exists:
        raise IOError(f"File '{file_path}' already exists")

    # Create the directory if necessary
    dir_path, _ = os.path.split(file_path)
    if not dir_create and not os.path.isdir(dir_path):
        raise IOError(f"Directory '{dir_create}' does not exist")

    # Attempt to create the directory
    os.makedirs(dir_path, exist_ok=True)

    # Create the file output
    s_output = ""
    for p_name, p_desc in parameters:
        desc = [f"#  {p}" for p in p_desc]
        s = "\n".join(desc)
        s += "\n\n"

        # Print the dictionary entry itself: yaml.dump performs necessary formatting
        d = {p_name: param_value_dict[p_name]}
        s += yaml.dump(d, indent=4)
        s += "\n\n"
        s_output += s

    with open(file_path, "w") as f:
        f.write(s_output)


def read_yaml_parameter_file(*, file_path):

    # Convert to absolute path
    file_path = os.path.expanduser(file_path)
    file_path = os.path.abspath(file_path)

    if not os.path.isfile(file_path):
        raise IOError(f"File '{file_path}' does not exist")

    with open(file_path, 'r') as f:
        param_dict = yaml.load(f, Loader=yaml.FullLoader)

    return param_dict
