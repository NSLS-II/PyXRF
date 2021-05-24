import os
import yaml
import jsonschema
import numpy as np
import math
import json
import copy
import time as ttime
from .xrf_utils import split_compound_mass, generate_eline_list
from .utils import normalize_data_by_scaler, convert_time_to_nexus_string
import logging

logger = logging.getLogger(__name__)

# ==========================================================================================
#    Functions for operations with YAML files used for keeping descriptions of XRF standards

_xrf_standard_schema = {
    "type": "object",
    "additionalProperties": False,
    "required": ["name", "serial", "description", "compounds"],
    "properties": {
        "name": {"type": "string"},
        "serial": {"type": "string"},
        "description": {"type": "string"},
        "compounds": {
            "type": "object",
            # Chemical formula should always start with a captial letter (Fe2O3)
            "patternProperties": {"^[A-Z][A-Za-z0-9]*$": {"type": "number"}},
            "additionalProperties": False,
            "minProperties": 1,
        },
        "density": {"type": "number"},  # Total density is an optional parameter
    },
}

_xrf_standard_schema_instructions = """
# The file was automatically generated.
#
# Instructions for editing this file:
#
# Description of each standard starts with '- name: ...'. Every following line
#   must be indented by 4 spaces. Each description contains the following items:
#   'name' (name of the standard, arbitrary string), 'serial' (serial number, of
#   the standard, but can be arbitrary string, 'description' (string that contains
#   description of the standard). Those fields may be filled with arbitrary information,
#   best suited to distinguish the standard later. If string consists of only digits
#   (in case of serial number) it must be enclosed in quotes.
#
# The field 'compounds' lists all compounds in the standard. The compounds are
#   presented in the form <compound_formula>: <concentration>.
#   <compound_formula> has to be a valid chemical formula, representing a pure
#   element (C, Fe, Ga, etc.) or compound (Fe2O3, GaAs, etc). Element names
#   must start with a capital letter followed by a lowercase letter (if present).
#   No characters except 'A-Z', 'a-z' and '0-1' are allowed. Lines containing
#   compound specifications must be indented by extra 4 spaces.
#
# The optional field 'density' specifies total density of the sample and used
#   to check integrity of the data (the sum of densities of all compounds
#   must be equaly to 'density' value.
#
# All density values (for compounds and total density) are specified in ug/cm^2
#
# Example (the lines contain extra '#' character, which is not part of YAML file):
#
#-   name: Micromatter 41164
#    serial: '41164'
#    description: CeF3 21.1 / Au 20.6
#    compounds:
#        CeF3: 21.1
#        Au: 20.6
#    density: 41.7
#
# The easiest way to start creating the list of custom standards is to uncomment
#   and edit the following example. To create extra records, duplicate and
#   edit the example or any existing record.

#-    name: Name of the Standard
#     serial: '32654'
#     description: CeF3 21.1 / Au 20.6 (any convenient description)
#     compounds:
#         CeF3: 21.1
#         Au: 20.6

"""


def save_xrf_standard_yaml_file(file_path, standard_data, *, overwrite_existing=False):
    r"""
    Save descriptions of of XRF standards to YAML file

    Parameters
    ----------

    file_path: str
        absolute or relative path to the saved YAML file. If the path does not exist, then
        it is created.

    standard_data: list(dict)
        list of dictionaries, each dictionary is representing the description of one
        XRF standard. Sending ``[]`` will create YAML file, which contains only instructions
        for manual editing of records. Such file can be read by the function
        ``load_xrf_standard_yaml_file``, which returns ``[]``.

    overwrite_existing: bool
        indicates if existing file should be overwritten. Default is False, since
        overwriting of an existing parameter file will lead to loss of data.

    Returns
    -------

        no value is returned

    Raises
    ------

    IOError if the YAML file already exists and ``overwrite_existing`` is not enabled.
    """

    # Make sure that the directory exists
    file_path = os.path.expanduser(file_path)
    file_path = os.path.abspath(file_path)
    flp, _ = os.path.split(file_path)
    os.makedirs(flp, exist_ok=True)

    if not overwrite_existing and os.path.isfile(file_path):
        raise IOError(f"File '{file_path}' already exists")

    s_output = _xrf_standard_schema_instructions
    if standard_data:
        s_output += yaml.dump(standard_data, default_flow_style=False, sort_keys=False, indent=4)
    with open(file_path, "w") as f:
        f.write(s_output)


def load_xrf_standard_yaml_file(file_path, *, schema=_xrf_standard_schema):
    r"""
    Load the list of XRF standard descriptions from YAML file and verify the schema.

    Parameters
    ----------

    file_path: str
        absolute or relative path to YAML file. If file does not exist then IOError is raised.

    schema: dict
        reference to schema used for validation of the descriptions. If ``schema`` is ``None``,
        then validation is disabled (this is not the default behavior).

    Returns
    -------

        list of dictionaries, each dictionary is representing the description of one XRF
        standard samples. Empty dictionary is returned if the file contains no data.

    Raises
    ------

    IOError is raised if the YAML file does not exist.

    jsonschema.ValidationError is raised if schema validation fails.

    RuntimeError if the sum of areal densities of all compounds does not match the
    total density of the sample for at least one sample. The list of all sample
    records for which the data integrity is not confirmed is returned in the
    error message. For records that do not contain 'density' field the integrity
    check is not performed.
    """

    file_path = os.path.expanduser(file_path)
    file_path = os.path.abspath(file_path)

    if not os.path.isfile(file_path):
        raise IOError(f"File '{file_path}' does not exist")

    with open(file_path, "r") as f:
        standard_data = yaml.load(f, Loader=yaml.FullLoader)

    if standard_data is None:
        standard_data = []

    if schema is not None:
        for data in standard_data:
            jsonschema.validate(instance=data, schema=schema)

    # Now check if all densities of compounds sums to total density in every record
    msg = []
    for data in standard_data:
        if "density" in data:
            # The sum of all densities must be equal to total density
            sm = np.sum(list(data["compounds"].values()))
            if not math.isclose(sm, data["density"], abs_tol=1e-6):
                msg.append(f"Record #{data['serial']} ({data['name']}): computed {sm} vs total {data['density']}")
    if msg:
        msg = [f"    {_}" for _ in msg]
        msg = "\n".join(msg)
        msg = "Sum of areal densities does not match total density:\n" + msg
        raise RuntimeError(msg)

    return standard_data


def load_included_xrf_standard_yaml_file():
    r"""
    Load YAML file with descriptions of XRF standards that is part of the
    package.

    Returns
    -------

    List of dictionaries, each dictionary represents description of one XRF standard.

    Raises
    ------

    Exceptions may be raised by ``load_xrf_standard_yaml_file`` function
    """

    # Generate file name (assuming that YAML file is in the same directory)
    file_name = "xrf_quant_standards.yaml"
    file_path = os.path.realpath(__file__)
    file_path, _ = os.path.split(file_path)
    file_path = os.path.join(file_path, file_name)

    return load_xrf_standard_yaml_file(file_path)


def compute_standard_element_densities(compounds):
    r"""
    Computes areal density of each element in the mix of compounds.
    Some compounds in the mix may contain the same elements.

    Parameters
    ----------

    compounds: dict

        dictionary of compound densities: key - compound formula,
        value - density (typically ug/cm^2)

    Returns
    -------

    Dictionary of element densities: key - element name (symbolic),
    value - elmenet density.
    """

    element_densities = {}

    for key, value in compounds.items():
        el_dens = split_compound_mass(key, value)
        for el, dens in el_dens.items():
            if el in element_densities:
                element_densities[el] += dens
            else:
                element_densities[el] = dens

    return element_densities


# ==========================================================================================
#    Functions for operations with JSON files used for keeping quantitative data obtained
#      after processing of XRF standard samples. The data is saved after processing
#      XRF scan of standard samples and later used for quantitative analysis of
#      experimental samples.

_xrf_quant_fluor_schema = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "name",
        "serial",
        "description",
        "element_lines",
        "incident_energy",
        "scaler_name",
        "distance_to_sample",
        "creation_time_local",
        "source_scan_id",
        "source_scan_uid",
    ],
    "properties": {
        # 'name', 'serial' and 'description' (optional) are copied
        #   from the structure used for description of XRF standard samples
        "name": {"type": "string"},
        "serial": {"type": "string"},
        "description": {"type": "string"},
        # The list of element lines. The list is not expected to be comprehensive:
        #   it includes only the lines selected for processing of standard samples.
        "element_lines": {
            "type": "object",
            "additionalProperties": False,
            "minProperties": 1,
            # Symbolic expression representing an element line:
            # Fe - represents all lines, Fe_K - K-lines, Fe_Ka - K alpha lines,
            # Fe_Ka1 - K alpha 1 line. Currently only selections that contain
            # all K, L or M lines is supported.
            "patternProperties": {
                r"^[A-Z][a-z]?(_[KLM]([ab]\d?)?)?$": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["density", "fluorescence"],
                    "properties": {"density": {"type": "number"}, "fluorescence": {"type": ["number", "null"]}},
                }
            },
        },
        # Incident energy used in the processing experiment
        "incident_energy": {"type": "number"},
        # Selected channel, expected values are 'sum', 'det1', 'det2', 'det3' etc.
        "detector_channel": {"type": ["string", "null"]},
        # Name of the valid scaler name (specific for data recorded on the beamline
        "scaler_name": {"type": ["string", "null"]},
        # Distance to the sample (number or null)
        "distance_to_sample": {"type": ["number", "null"]},
        # Time of file creation (NEXUS format), optional, null if not set
        "creation_time_local": {"type": ["string", "null"]},
        # Scan ID of the source (scan of the standard), optional, null if not set
        "source_scan_id": {"type": ["integer", "null"]},
        # Scan UID of the source (scan of the standard), optional, null if not set
        "source_scan_uid": {"type": ["string", "null"]},
    },
}


def save_xrf_quant_fluor_json_file(file_path, fluor_data, *, overwrite_existing=False):
    r"""
    Save the results of processing of a scan data for XRF standard sample to a JSON file.
    The saved data will be used later for quantitative analysis of experimental samples.

    Parameters
    ----------

    file_path: str
        absolute or relative path to the saved JSON file. If the path does not exist, then
        it is created.

    fluor_data: dict
        dictionary, which contains the results of processing of a scan of an XRF standard.
        The dictionary should conform to ``_xrf_quantitative_fluorescence_schema``.
        The schema is verified before saving to ensure that the data can be successfully read.

    overwrite_existing: bool
        indicates if existing file should be overwritten. Default is False, since
        overwriting of an existing parameter file will lead to loss of data.

    Returns
    -------

        no value is returned

    Raises
    ------

    IOError if the JSON file already exists and ``overwrite_existing`` is not enabled.

    jsonschema.ValidationError if schema validation fails
    """

    # Note: the schema is fixed (not passed as a parameter). If data format is changed,
    #   then the built-in schema must be changed. The same schema is always used
    #   both for reading and writing of data.
    jsonschema.validate(instance=fluor_data, schema=_xrf_quant_fluor_schema)

    # Make sure that the directory exists
    file_path = os.path.expanduser(file_path)
    file_path = os.path.abspath(file_path)
    flp, _ = os.path.split(file_path)
    os.makedirs(flp, exist_ok=True)

    if not overwrite_existing and os.path.isfile(file_path):
        raise IOError(f"File '{file_path}' already exists")

    s_output = json.dumps(fluor_data, sort_keys=False, indent=4)
    with open(file_path, "w") as f:
        f.write(s_output)


def load_xrf_quant_fluor_json_file(file_path, *, schema=_xrf_quant_fluor_schema):
    r"""
    Load the quantitative data for XRF standard sample from JSON file and verify the schema.

    Parameters
    ----------

    file_path: str
        absolute or relative path to JSON file. If file does not exist then IOError is raised.

    schema: dict
        reference to schema used for validation of the descriptions. If ``schema`` is ``None``,
        then validation is disabled (this is not the default behavior).

    Returns
    -------

        dictionary containing quantitative fluorescence data on XRF sample.

    Raises
    ------

    IOError is raised if the YAML file does not exist.

    jsonschema.ValidationError is raised if schema validation fails.
    """

    file_path = os.path.expanduser(file_path)
    file_path = os.path.abspath(file_path)

    if not os.path.isfile(file_path):
        raise IOError(f"File '{file_path}' does not exist")

    with open(file_path, "r") as f:
        fluor_data = json.load(f)

    if schema is not None:
        jsonschema.validate(instance=fluor_data, schema=schema)

    return fluor_data


def get_quant_fluor_data_dict(quant_param_dict, incident_energy):
    r"""
    Create the dictionary used for storage of data on XRF reference sample. The field
    ``element_lines`` is the dictionary, which stores data on density density of the
    element (in the sample) and fluorescence of the emission line (computed later
    during processing of the reference scan.

    Parameters
    ----------

    quant_param_dict: dict
        Dictionary with the information on reference sample (loaded from YAML configuration
        file). The dictionary should satifsfy the ``_xrf_standard_schema`` schema.

    incident_energy: float
        Incident beam energy

    Returns
    -------

    quant_fluor_data_dict: dict
        Dictionary that contains data on XRF reference sample, including the field
        ``element_lines``, which relates the emission lines (active at ``incident_energy``)
        with respective fluorescence (area under the line spectra) and density of the element.
        The fluorescence is set to None and needs to be computed later. The dictionary
        should satisfy the ``_xrf_quant_fluor_schema`` schema.
    """
    if incident_energy is not None:
        # Make sure that it is 'float', not 'float64', since 'float64' is not supported by 'yaml' package
        incident_energy = float(max(incident_energy, 0))

    quant_fluor_data_dict = {}
    quant_fluor_data_dict["name"] = quant_param_dict["name"]
    quant_fluor_data_dict["serial"] = quant_param_dict["serial"]
    quant_fluor_data_dict["description"] = quant_param_dict["description"]

    # Find the density (mass) of each element in the mis of compounds.
    #   Note, that the sample may contain the same element as a component of multiple compounds.
    element_dict = {}
    for compound, mass in quant_param_dict["compounds"].items():
        # Split compound/compound_density into elements/element_density
        el_and_mass = split_compound_mass(compound, mass)
        for el, ms in el_and_mass.items():
            if el in element_dict:
                element_dict[el] += ms
            else:
                element_dict[el] = ms

    # Create the dictionary of element lines. Fluorescence is unknown at this point,
    #   so it is always None.
    element_lines = {}
    for el, ms in element_dict.items():
        lines = generate_eline_list([el], incident_energy=incident_energy)
        e = {_: {"density": ms, "fluorescence": None} for _ in lines}
        element_lines.update(e)

    quant_fluor_data_dict["element_lines"] = element_lines

    quant_fluor_data_dict["incident_energy"] = incident_energy

    quant_fluor_data_dict["detector_channel"] = None
    quant_fluor_data_dict["scaler_name"] = None
    quant_fluor_data_dict["distance_to_sample"] = None

    quant_fluor_data_dict["creation_time_local"] = None
    quant_fluor_data_dict["source_scan_id"] = None
    quant_fluor_data_dict["source_scan_uid"] = None

    return quant_fluor_data_dict


def fill_quant_fluor_data_dict(quant_fluor_data_dict, *, xrf_map_dict, scaler_name):
    r"""
    Computes average normalized fluorescence values for element lines that are part of
    the reference sample and listed in ``quant_fluor_data_dict["element_lines"]`` and
    present in the ``xrf_map_dict`` and writes the result to
    ``quant_fluor_data_dict["element_lines"][<element_line>]["fluorescence"]``.
    Fluorescence is set to None for the element lines that are not present in ``xrf_map_dict``.
    Element lines that are present in ``xrf_map_dict``, but not part of the reference
    standard, are ignored. If `scaler_name` is one of the keys of ``xrf_map_dict``, then
    fluorescence map is normalized by the scaler before average value is computed. If
    ``scaler_name`` is not one of the keys of ``xrf_map_dict`` or set to None, then
    the average fluorescence is computed without normalization.

    Pixels along the edges of the map are very likely to contain outliers, so the edges
    are not used in computation whenever sufficient data is available (if map contains
    more than 2 pixels along a dimension, the first and the last pixels are not used
    for averaging)

    Parameters
    ----------

    quant_fluor_data_dict: dict
        Dictionary with XRF reference sample data. This dictionary is modified by the function.
        The dictionary must satisfy the '_xrf_quant_fluor_schema' schema.

    xrf_map_dict: dict(array)
        The dictionary of 2D ndarrays, which contain XRF maps for element lines and scalers.
        Dictionary keys are the names of the emission lines (e.g. Fe_K, S_K, Au_M etc.)

    scaler_name: str
        Scaler name. In order for the scaler to be applied, the name must match one of the
        keys of `xrf_map_dict'. If the scaler name is not one of the list keys or set to None,
        then fluorescence is not normalized

    Returns

        No value is returned. The computed fluorescence for the element lines is saved to
        ``quant_fluor_data_dict["element_lines"][<element_line>]["fluorescence"]``

    """

    if not scaler_name:
        logger.warning("No scaler is selected for computing quantitative coefficients. Data is not normalized.")
    elif scaler_name not in xrf_map_dict:
        logger.warning("Scaler '{scaler_name}' is not in XRF map dictionary. Normalization can not be performed.")
        scaler_name = None

    # Clear ALL fluorescence values. Don't touch any other data
    for eline, info in quant_fluor_data_dict["element_lines"].items():
        info["fluorescence"] = None

    # Save the scaler name
    quant_fluor_data_dict["scaler_name"] = scaler_name

    # Compute fluorescence of the emission lines
    eline_list = tuple(quant_fluor_data_dict["element_lines"].keys())
    for eline, map in xrf_map_dict.items():
        if eline in eline_list:
            # Normalize the map if scaler is selected. (Typically scaler IS selected.)
            if scaler_name:
                norm_map = normalize_data_by_scaler(xrf_map_dict[eline], xrf_map_dict[scaler_name])
            else:
                norm_map = xrf_map_dict[eline]
            # Ignore pixels along the edges (those pixels are likely to be outliers that will visibly bias
            #   the mean value in small calibration scans). If scan is smaller that 2 pixels along any
            #   dimension, then all pixels are used, including edges.

            def _get_range(n_elements):
                if n_elements > 2:
                    n_min, n_max = 1, n_elements - 1
                else:
                    n_min, n_max = 0, n_elements
                return n_min, n_max

            ny_min, ny_max = _get_range(norm_map.shape[0])
            nx_min, nx_max = _get_range(norm_map.shape[1])
            mean_fluor = np.mean(norm_map[ny_min:ny_max, nx_min:nx_max])
            # Note: numpy 'float64' is explicitely converted to 'float'
            #     (yaml package does not seem to support 'float64')
            quant_fluor_data_dict["element_lines"][eline]["fluorescence"] = float(mean_fluor)


def prune_quant_fluor_data_dict(quant_fluor_data_dict):
    r"""
    Prunes the fluorescence data dictionary by removing the element lines that are not
    present (fluorescence is None) or have fluorescence <= 0. 'Pruning' is performed before
    saving calibration data, so that only meaningful information is saved.
    The function does not modify the original data structure. Instead it returns the
    copy of the original dictionary with some emission line removed.

    Parameters
    ----------

    quant_fluor_data_dict: dict

        Dictionary with XRF reference sample data. This dictionary is modified by the function.
        The dictionary must satisfy the '_xrf_quant_fluor_schema' schema.

    Returns
    -------

        Copy of ``quant_fluor_data_dict`` with some emission lines removed. Only the emission
        lines that have fluorescence set to valid value are left.
    """
    quant_fluor_data_dict = copy.deepcopy(quant_fluor_data_dict)
    for key, val in quant_fluor_data_dict["element_lines"].copy().items():
        if (val["fluorescence"] is None) or (val["fluorescence"] <= 0):
            del quant_fluor_data_dict["element_lines"][key]

    return quant_fluor_data_dict


def set_quant_fluor_data_dict_optional(quant_fluor_data_dict, *, scan_id=None, scan_uid=None):
    r"""
    Set optional parameters in the existing dictionary with quantitative fluorescence data.
    The parameters include: source_scan_id (if provided), source_scan_uid (if provided),
    and creation time (local time). The function modifies the dictionary ``quant_fluor_data_dict``

    Parameters
    ----------

    quant_fluor_data_dict: dict

        Dictionary with XRF reference sample data. This dictionary is modified by the function.
        The dictionary must satisfy the '_xrf_quant_fluor_schema' schema.

    scan_id: int or str
        Scan ID, must be positive int or a string representing int. If None, then the current
        value of Scan ID is not changed.

    scan_uid: str
        Scan UID. UID has defined format, but it is not checked by the function, so it may
        be any string. If None, then the current value of Scan UID is not changed.

    Returns
    -------

        No value is returned. Instead ``quant_fluor_data_dict`` is modified.
    """
    # Set scan ID (optional parameter)
    if scan_id is not None:
        try:
            # Attempt to convert to 'int' (from str, 'int64' etc)
            scan_id = int(scan_id)
        except Exception:
            raise RuntimeError("Parameter 'scan_id' must be integer or a string representing integer")

        quant_fluor_data_dict["source_scan_id"] = scan_id

    # Set scan UID (optional parameter)
    if scan_uid is not None:
        if not isinstance(scan_uid, str):
            raise RuntimeError("Parameter 'scan_uid' must be a string representing scan UID")

        quant_fluor_data_dict["source_scan_uid"] = scan_uid

    # Set time as well (it may be changed later)
    set_quant_fluor_data_dict_time(quant_fluor_data_dict)


def set_quant_fluor_data_dict_time(quant_fluor_data_dict):
    r"""
    Set creation time to current local time

    Parameters
    ----------

    quant_fluor_data_dict: dict

        Dictionary with XRF reference sample data. This dictionary is modified by the function.
        The dictionary must satisfy the '_xrf_quant_fluor_schema' schema.

    Returns
    -------

        No value is returned. Instead ``quant_fluor_data_dict`` is modified.
    """
    # TODO: Documentation and tests

    # Set creation time (current local time in NEXUS format)
    quant_fluor_data_dict["creation_time_local"] = convert_time_to_nexus_string(ttime.localtime())


# -------------------------------------------------------------------------------------------------


class ParamQuantEstimation:
    r"""
    The class is used for measurement of parameters in the process of estimation of
    quantitative calibration data.

    The methods of the class are designed to be used in the followingsequence:

    # Create object
    pqe = ParamQuantEstimation()
    # Load standards
    pqe.load_standards()
    # Find standard if needed (different options are available)
    st = pqe.find_standard(serial_number, key="serial")
    # Select standard
    pqe.set_selected_standard(st)

    # Generate data dictionary
    pqe.gen_fluorescence_data_dict(incident_energy=12.0)
    # Fill the dictionary using XRF map dictionary (e.g. ``img_dict``) and scaler name (e.g. ``i0``)
    pqe.fill_fluorescence_data_dict(xrf_map_dict=img_dict, scaler_name="i0")
    # Set different (optional but desired) parameters
    pqe.set_detector_channel_in_data_dict(detector_channel="sum")
    pqe.set_optional_parameters(scan_id=12345, scan_uid="some-uid-string")

    # Get preview (if needed) for displaying to users
    preview_str = pqe.get_fluorescence_data_dict_text_preview()

    # Get suggested file name for the parameter file
    fln = pqe.get_suggested_json_fln()
    # Generate full path based on fln
    file_path = .....
    # Save data to file
    pqe.save_fluorescence_data_dict(file_path=file_path)

    At any time, the generated/filled fluorescence data dictionary may be
    accessed as ``self.fluorescence_data_dict``.
    """

    def __init__(self, *, home_dir="~", config_dir=".pyxrf", standards_fln="quantitative_standards.yaml"):
        r"""
        Constructor of the ``ParamQuantEstiomation`` class. In addition to initalization
        of the fields, the constructor checks if the default file with user-defined
        reference standards exists in the PyXRF config directory and creates an empty file
        if it does not exist.

        Parameters
        ----------

        home_dir: str
            HOME directory for PyXRF configuration files. Typically it is ``~``,
            but may be changed to temporary directory to run unit tests

        config_dir: str
            subdirectory in the HOME directory, used to keep PyXRF config files.
            Typical value is ``.pyxrf``.

        standards_fln: str
            name of the file for storing the data on user-defined reference standards.
        """

        custom_path = (os.path.expanduser(home_dir), config_dir, standards_fln)
        self.custom_standards_file_path = os.path.join(*custom_path)

        # If file with user-defined set of reference standards does not exist, create one
        if not os.path.isfile(self.custom_standards_file_path):
            try:
                save_xrf_standard_yaml_file(self.custom_standards_file_path, [])
            except Exception as ex:
                logger.error(f"Failed to create empty file for custom set of quantitative standards: {ex}")

        # Lists of standard descriptions
        self.standards_built_in = None
        self.standards_custom = None

        # Reference to the selected standard description (in custom or built-in list)
        self.standard_selected = None

        # List of emission lines for the selected incident energy
        self.incident_energy = 0.0
        self.emission_line_list = None

        # Dictionary with fluorescence data for the selected standard. Filled as data is processed
        self.fluorescence_data_dict = None

    def load_standards(self):
        r"""
        Load reference standards data (both built-in and user-defined). Must be called before
        attempting to access the reference standards
        """
        self.clear_standards()

        try:
            self.standards_built_in = load_included_xrf_standard_yaml_file()
        except Exception as ex:
            self.standards_built_in = None
            logger.error(f"Failed to load built-in set of quantitative standards: {ex}")

        try:
            self.standards_custom = load_xrf_standard_yaml_file(self.custom_standards_file_path)
        except Exception as ex:
            self.standards_custom = None
            logger.error(f"Failed to load custom set of quantitative standards: {ex}")

    def clear_standards(self):
        r"""
        Clear the lists of reference standards and all data that was computed based on
        loaded reference standards. This function is called before reloading the standards.
        It is recommended that the data is reloaded often (each time the dialog box
        for reference standard selection is opened), because the reference files may be
        edited by the user and the most recent version of the data should be displayed
        in the dialog box
        """
        self.standards_built_in = None
        self.standards_custom = None
        self.standard_selected = None
        self.emission_line_list = None
        self.fluorescence_data_dict = None

    def _find_standard_custom(self, standard, key=None):
        r"""
        Search for standard in user-defined list. For more detailed description
        see docstring for ``find_standard`` function
        """
        standard_ref = None
        if self.standards_custom:
            for st in self.standards_custom:
                if (st == standard) if (key is None) else (st[key] == standard):
                    standard_ref = st
                    break
        return standard_ref

    def _find_standard_built_in(self, standard, key=None):
        r"""
        Search for standard in built-in list. For more detailed description
        see docstring for ``find_standard`` function
        """

        standard_ref = None
        if self.standards_built_in:
            for st in self.standards_built_in:
                if (st == standard) if (key is None) else (st[key] == standard):
                    standard_ref = st
                    break
        return standard_ref

    def find_standard(self, standard, key=None):
        r"""
        Search for the standard in the lists of user-defined or built-in standards.
        Search may be performed by using the complete standard information (dictionary)
        or by one of the keys. The function returns reference to the first matching
        entry in the list or None.

            Examples:
            ``find_standard(st)`` - search for dictionary ``st``
            ``find_standard(st["name"], key="name")`` - search by key ``name``.

        Parameters
        ----------

        standard: dict or other
            The dictionary with standard information (the complete dictionary
            must match one of the list elements) or the value of one of the
            dictionary element (e.g. serial, or name). In the latter case,
            the ``key`` must be specified

        key: str or None
            The name of the dictionary key used for searching. The first
            matching entry is returned.

        Returns
        -------

            Reference to one of the entries of user-defined or built-in lists
            if search is successful or None if the standard was not found.
        """
        if standard is None:
            return None

        standard_ref = self._find_standard_custom(standard, key=key)
        if not standard_ref:
            standard_ref = self._find_standard_built_in(standard, key=key)

        return standard_ref

    def set_selected_standard(self, standard=None):
        r"""
        Set standard ``standard`` as currently selected. If ``standard`` does not
        exist in user-defined or built-in list, then the first entry of the
        user-defined list is set as current. If user-defined list is empty or
        None, then the first entry of the built-in list is set as current.
        If both lists are empty or None, then then nothing is selected.
        If ``standard`` is None, then the first available standard is selected.

        Parameters
        ----------

        standard: dict
            The dictionary holding standard information. The complete dictionary
            is compared with entries of internally stored lists to find the match.

        Returns
        -------
            Returns reference to the selected standard (None if no entry is selected)
        """

        standard_ref = self.find_standard(standard)

        if not standard_ref:
            # Set reference pointing to the first available standard description
            if self.standards_custom:
                self.standard_selected = self.standards_custom[0]
            elif self.standards_built_in:
                self.standard_selected = self.standards_built_in[0]
            else:
                self.standard_selected = None

        else:
            # The reference was found in one of the arrays
            self.standard_selected = standard_ref

        return self.standard_selected

    def is_standard_custom(self, standard):
        r"""
        Returns True if standard ``standard`` is user-defined and False otherwise

        Parameters
        ----------

        standard: dict
            The dictionary holding standard information. The complete dictionary
            is compared with entries of internally stored lists to find the match.
        """
        return bool(self._find_standard_custom(standard))

    def gen_fluorescence_data_dict(self, incident_energy):
        r"""
        Generate fluorescence data dictionary based on the description of the selected
        reference standard (reference standard must be selected using ``set_selected_standard``).

        Parameters
        ----------

        incident_energy: float

            Incident beam energy, keV
        """

        if incident_energy:
            self.incident_energy = incident_energy
        self.incident_energy = float(max(self.incident_energy, 0.0))
        # 'incident_energy' must be 'float', not numpy 'float64' in order to be correctly displayed
        #   in 'yaml' formatted preview ('yaml' package does not support 'float64')

        if incident_energy == 0.0:
            logger.warning("Attempting to compute the list of emission lines with incident energy set to 0")

        self.fluorescence_data_dict = get_quant_fluor_data_dict(self.standard_selected, incident_energy)

    def fill_fluorescence_data_dict(self, *, xrf_map_dict, scaler_name):
        r"""
        Fills the generated fluorescence data dictionary based on XRF map dictionary and scaler name.
        The function resets fluorescence for all emission lines in the dictionary to None and then
        iterates through the list of the emission lines. If the map for the emission line is found
        in the ``xrf_map_dict``, then the fluorescence value is computed and filled. Fluorescence
        is computed as mean value of the respective map (over all pixels). If the scaler with name
        ``scaler_name`` is found in ``xrf_map_dict`` then XRF map is normalized by the scaler before
        computing fluorescence, otherwise no normalization is performed.

        Before this function is called, the standards must be loaded (``load_standards``),
        a standard selected (``set_selected_standard``) and fluorescence data dictionary generated
        (``gen_fluorescence_data_dict``).

        Parameters
        ----------

        xrf_map_dict: dict(array)
            The dictionary of 2D ndarrays, which contain XRF maps for element lines and scalers.
            Dictionary keys are the names of the emission lines (e.g. Fe_K, S_K, Au_M etc.)

        scaler_name: str
            Scaler name. In order for the scaler to be applied, the name must match one of the
            keys of `xrf_map_dict'. If the scaler name is not one of the list keys or set to None,
            then fluorescence is not normalized
        """

        fill_quant_fluor_data_dict(self.fluorescence_data_dict, xrf_map_dict=xrf_map_dict, scaler_name=scaler_name)

    def set_detector_channel_in_data_dict(self, *, detector_channel=None):
        r"""
        Set detector channel (``sum``, ``det1``, ``det2`` etc.) in the fluorescence data dictionary.
        Information on the selected detector channel will be used to ensure that correct
        detector channel is selected when processing data (the channels MUST match).

        Parameters
        ----------

        detector_channel: str or None

            The name of the detector channel (``sum``, ``det1``, ``det2`` etc.)
        """
        self.fluorescence_data_dict["detector_channel"] = detector_channel

    def set_distance_to_sample_in_data_dict(self, *, distance_to_sample=None):
        r"""
        Set distance-to-sample in the fluorescence data dictionary. The value must be
        postive float or ZERO. Distance-to-sample is the distance between the sample
        and the detector that is used to adjust calibration in case the distance
        changes between calibration and experimental scans. If distance-to-sample is
        zero, then no adjustment will be performed. If both calibration and experimental
        scans are performed without moving the detector, the value may be kept zero (or None)

        Parameters
        ----------

        distance_to_sample: float

            Distance-to-sample, may be positive float or 0.0.
        """
        if distance_to_sample is not None:
            distance_to_sample = float(max(distance_to_sample, 0.0))
        self.fluorescence_data_dict["distance_to_sample"] = distance_to_sample

    def set_optional_parameters(self, *, scan_id=None, scan_uid=None):
        r"""
        Set optional parameters in the existing dictionary with quantitative fluorescence data.
        The parameters include: source_scan_id (if provided), source_scan_uid (if provided),
        and creation time (local time). The function modifies the dictionary ``quant_fluor_data_dict``

        Parameters
        ----------

        quant_fluor_data_dict: dict

            Dictionary with XRF reference sample data. This dictionary is modified by the function.
            The dictionary must satisfy the '_xrf_quant_fluor_schema' schema.

        scan_id: int or str
            Scan ID, must be positive int or a string representing int. If None, then the current
            value of Scan ID is not changed.

        scan_uid: str
            Scan UID. UID has defined format, but it is not checked by the function, so it may
            be any string. If None, then the current value of Scan UID is not changed.
        """

        set_quant_fluor_data_dict_optional(self.fluorescence_data_dict, scan_id=scan_id, scan_uid=scan_uid)

    def get_suggested_json_fln(self):
        r"""
        Get suggested file name for the reference standard data. File name
        serial number of the standard, therefore the fluorescence data dict must be filled
        before calling this function
        """
        fln = f"standard_{self.fluorescence_data_dict['serial']}.json"
        return fln

    def get_fluorescence_data_dict_text_preview(self, enable_warnings=True):
        r"""
        Generate preview of fluorescence data dictionary. The preview is a human-readable
        multiline string, which displays the dictionary in YAML format. Warnings are
        added to the beginning of the string if ``scaler_name`` or ``distance_to_sample``
        are not set. Warnings can be disabled if ``enable_warnings`` is set to ``False``.

        Parameters
        ----------

        enable_warnings: bool

            True - warnings are enabled, False - warnings are disabled and not be included
            in the output string

        Returns
        -------
        str, dict(str, str)
            A string that contains the preview of the fluorescence data dictionary
            (calibration data) and the dictionary of warnings. Currently two warnings are
            generated: "scaler_missing" - data is not normalized by a scaler,
            "distance_to_sample" - distance-to-sample is set to zero (missing), so
            calibration data can't be used later if distance-to-sample is changed.
        """
        pruned_dict = prune_quant_fluor_data_dict(self.fluorescence_data_dict)
        # Print preview in YAML format (easier to read)
        s = yaml.dump(pruned_dict, default_flow_style=False, sort_keys=False, indent=4)
        msg_warnings = {}
        if enable_warnings:
            if (pruned_dict["scaler_name"] is None) or (pruned_dict["scaler_name"] == ""):
                msg = "WARNING: Scaler is not selected, data is not normalized."
                msg_warnings["scaler_missing"] = msg
            if (pruned_dict["distance_to_sample"] is None) or (pruned_dict["distance_to_sample"] == 0):
                msg = (
                    "WARNING: Distance-to-sample is set to 0 or None. "
                    "Set it to estimated distance between the detector and the standard sample "
                    "if you expect to it to change in the series of scans. Otherwise "
                    "the respective corrections may not be computed. Ignore if the distance "
                    "stays constant throughout the series of scans."
                )
                msg_warnings["distance_to_sample"] = msg
        return s, msg_warnings

    def save_fluorescence_data_dict(self, file_path, *, overwrite_existing=False):
        r"""
        Save the results of processing of a scan data for XRF standard sample to a JSON file.
        The saved data will be used later for quantitative analysis of experimental samples.
        Before saving, the copy of the original dictionary is createda and inactive emission
        lines (with ZERO fluorescence) are removed from the copy. Original dictionary is not
        modified and the copy is saved to the file.

        Before this function is called, the standards must be loaded (``load_standards``),
        a standard selected (``set_selected_standard``),  fluorescence data dictionary generated
        (``gen_fluorescence_data_dict``) and filled (``fill_fluorescence_data_dict``).

        Parameters
        ----------

        file_path: str
            absolute or relative path to the saved JSON file. If the path does not exist, then
            it is created.

        overwrite_existing: bool
            indicates if existing file should be overwritten. Default is False, since
            overwriting of an existing parameter file will lead to loss of data.

        Returns
        -------

            no value is returned

        Raises
        ------

        IOError if the JSON file already exists and ``overwrite_existing`` is not enabled.

        jsonschema.ValidationError if schema validation fails
        """
        # Set time (in the original dictionary)
        set_quant_fluor_data_dict_time(self.fluorescence_data_dict)
        # The next step creates a copy of the dictionary with removed inactive emission lines
        pruned_dict = prune_quant_fluor_data_dict(self.fluorescence_data_dict)
        save_xrf_quant_fluor_json_file(file_path, pruned_dict, overwrite_existing=overwrite_existing)


class ParamQuantitativeAnalysis:
    r"""
    The class is used to load and manipulate quantitative calibration data and apply quantitative
    normalization to XRF maps. Calibration data is loaded as entries, one calibration entry is
    produced by processing one calibration scan based on a reference standards. Typically calibration
    entry is stored in JSON file, but it can be added to the class directly. Multiple entries
    based on different reference standards may be loaded simultaneously. The same emission line
    may be used in different reference standards and therefore be present in multiple calibration
    entries. The class allows to select the source of calibration data (entry) for each emission
    line.

    The functions should be applied in the following sequence:

    pqa = ParamQuantitativeAnalysis()
    # Load calibration entries
    pqa.load_entry(<file_path1>)
    pqa.load_entry(<file_path2>)
    ...
    # or add calibration entries (the 1st parameter doesn't need to be a file path)
    pqa.add_entry("calibration1", calibration_data=<calib1>)
    pqa.add_entry("calibration2", calibration_data=<calib2>)
    ...
    # Entry may be removed as
    pqa.remove_entry(<file_path1>)
    # Get the list of file paths (or IDs)
    file_paths = pqa.get_file_path_list()
    # Get the number of entries
    n_entries = pqa.get_n_entries()
    # Entry index (index of loaded entry in the internal dictionaries)
    entry_index = pqa.find_entry_index(<file_path>)
    # Get text preview of loaded calibration entry
    text_preview = pqa.get_entry_text_preview(<file_path>)
    # Update the internal state of the object (including the list of available emission lines)
    #   Should be run each time the fields of the object are manually modified.
    pqa.update_emission_line_list()
    # Get the complete set of loaded calibration entries for the given emission line
    #   Note, then the index of the entry in 'eline_info' may be different from the
    #   index of the entry, since 'eline_info' contains only the entries which contain calibration
    #   for the emission line. The list 'eline_info' may be useful for GUI presentation of data.
    eline_info = pqa.get_eline_info_complete(<emission line, e.g. "Cu_K">)
    # Select emission line from the standard (useful when emission line is present in mutliple standards)
    pqa.select_eline(<emission_line>, <file_path>)
    # Check if emission line for particular standard is selected
    pqa.is_eline_selected(<emission_line>, <file_path>)
    # Get calibration data for the emission line (only from the selected entry)
    #   The information is useful at the stage of applying quantitative normalization
    eline_calib = pqa.get_calibrations_selected(<emission_line>)
    # Get calibration data on all emission lines as a list of dictionaries (the function
    #   calls 'get_calibrations_selected' for each emission line in the set)
    all_elines_calib = pqa.get_calibrations_selected
    # Set parameters of the experiment (before processing XRF map)
    pqa.set_experiment_incident_energy(<incident_energy>)
    pqa.set_experiment_detector_channel(<detector_channel>)
    pqa.set_experiment_distance_to_sample(<distance_to_sample>)
    # Process XRF map using selected calibrations and set experiment parameters
    data_normalized = apply_quantitative_normalization(<data_in>, scaler_dict=<scaler_dict>,
                                                       scaler_name_default = <scaler_name_default>,
                                                       data_name = <emission_line>,
                                                       name_not_scalable=["sclr", "sclr2", "time" etc.]):
    """

    def __init__(self):
        r"""
        Constructor. Initialization.
        """

        # List of opened calibration standards
        self.calibration_data = []
        self.calibration_settings = []
        self.active_emission_lines = []

        # Parameters of the experiment for the currently processed dataset:
        #   'experiment_detector_channel', 'experiment_incident_energy', 'experiment_distance_to_sample'.
        #   Set those parameters manually before running 'apply_quantitative_normalization'.

        # Detector channel (values 'sum', 'det1', 'det2' etc.) must match
        #   the channel specified for the quantitative calibration data for the emission line.
        #   (check is performed for each emission line separately, since calibration data
        #   may come from different calibration files depending on user's choice).
        self.experiment_detector_channel = None
        # Incident energy values should be approximately equal
        self.experiment_incident_energy = None
        # Distance to sample. If 0 or None, the respective correction is not applied
        self.experiment_distance_to_sample = None

    def load_entry(self, file_path):
        r"""
        Load QA calibration data from file. The calibration data file must be JSON file which
        satisfies schema ``_xrf_quant_fluor_schema``.

        Data may be loaded from the same file only once.
        If data is loaded from the same file repeatedly, then only one (first) copy is going to be
        stored in the memory.

        Parameters
        ----------

        file_path: str

            Full path to file with QA calibration data.
        """
        # May raise exception if reading the file fails
        calib_data = load_xrf_quant_fluor_json_file(file_path)
        # Data will be added only if file path ``file_path`` was not loaded before. Otherwise
        #   the message will be printed.
        self.add_entry(file_path, calibration_data=calib_data)

    def add_entry(self, file_path, *, calibration_data):
        r"""
        Add QA calibration data to the object. Calibration data is represented as a dictionary
        which satisfies ``_xrf_quant_fluor_schema`` schema. Calibration data may be loaded
        from JSON file (with file name ``file_path``), but it may already exist in memory
        as a dictionary (for example it may be generated or stored by some other module of
        the program). ``file_path`` does not have to represent valid file path, but some
        unique meaningful string, which may be used to identify calibration entries.
        This function DOES NOT LOAD calibration data from file! Use the function
        ``load_entry()`` to load data from file.

        Parameters
        ----------

        file_path: str

            String (perferably meaningful and human readable) used to identify the data entry.
            The value of this parameter is used to distinguish between calibration data
            entries (identify duplicates, search for calibration entries etc.). If data is
            loaded from file, then this parameter is naturally set to full path to the file
            that holds calibration data entry.

        calibration_data: dict

            Calibration data entry: dictionary that satisfies schema ``_xrf_quant_fluor_schema``.

        """

        # Do not load duplicates (distinguished by file name). Different file names may contain
        #   the same data, but this is totally up to the user. Loading duplicates should not
        #   disrupt the processing, since the user may choose the source of calibration data
        #   for each emission line.
        if not any(file_path == _["file_path"] for _ in self.calibration_settings):

            self.calibration_data.append(calibration_data)

            settings = {}
            settings["file_path"] = file_path
            settings["element_lines"] = {}
            for ln in calibration_data["element_lines"]:
                settings["element_lines"][ln] = {}
                # Do not select the emission line
                settings["element_lines"][ln]["selected"] = False

            self.calibration_settings.append(settings)
            # This will also select the emission line if it occurs the first time
            self.update_emission_line_list()
        else:
            logger.info(f"Calibration data file '{file_path}' is already loaded")

    def remove_entry(self, file_path):
        r"""
        Remove QA calibration data entry identified by ``file_path`` from memory
        (from object's internal list). Removes the entry if ``file_path`` is found, otherwise
        prints error message (``logger.error``).

        Parameters
        ----------

            The string used to identify QA calibration data entry. See the docstring for
            ``add_entry`` for more detailed comments.
        """
        n_item = self.find_entry_index(file_path)
        if n_item is not None:
            self.calibration_data.pop(n_item)
            self.calibration_settings.pop(n_item)
            self.update_emission_line_list()
        else:
            raise RuntimeError(f"Calibration data from source '{file_path}' is not found.")

    def get_file_path_list(self):
        r"""
        Returns the list of file paths, which identify the available calibration data entries.
        If calibration data entries are not directly loaded from files ``load_entry``,
        but instead added using ``add_entry``, the strings may not represent actual
        file paths, but instead be any strings used to uniquely identify calibration data entries.

        Returns
        -------

        file_path_list: list(str)

            List of strings that identify QA calibration data entries, may represent paths to
            files in which calibration data entries are stored. Returns empty list if no
            QA calibration data are loaded.
        """
        file_path_list = []
        for v in self.calibration_settings:
            file_path_list.append(v["file_path"])
        return file_path_list

    def get_n_entries(self):
        r"""
        Returns the number of QA calibration data entries.

        Returns
        -------

            The number of available calibration entries. If not calibration data was loaded or added,
            then the returned value is 0.
        """
        if not self.calibration_settings:
            return 0
        else:
            return len(self.calibration_settings)

    def find_entry_index(self, file_path):
        r"""
        Find the index of calibration data entry with the given file path. Returns ``None``
        if ``file_path`` is not found.

        Parameters
        ----------

        file_path: str

            The string used to identify QA calibration data entry. See the docstring for
            ``add_entry`` for more detailed comments.

        """
        n_item = None
        for n, v in enumerate(self.calibration_settings):
            if v["file_path"] == file_path:
                n_item = n
                break
        return n_item

    def get_entry_text_preview(self, file_path):
        r"""
        Returns text (YAML) preview of the dictionary with QA calibration data entry.
        Only original data (loaded from file) is printed.

        Parameters
        ----------

        file_path: str

            The string used to identify QA calibration data entry. See the docstring for
            ``add_entry`` for more detailed comments.

        Returns
        -------

            String wich contains text (YAML) representation of QA calibration data entry.
            Empty string if the entry is not found.
        """
        n_item = self.find_entry_index(file_path)
        if n_item is not None:
            s = yaml.dump(self.calibration_data[n_item], default_flow_style=False, sort_keys=False, indent=4)
        else:
            s = ""
        return s

    def select_eline(self, eline, file_path):
        r"""
        Select emission line in the standard pointed by `file_path` as selected.
        The function will also deselect any the emission line in all other loaded standards.
        Use `get_file_path_list()` to get the list of file paths arranged in the order
        in which standards were loaded.

        It is assumed that `eline` exists in the standard `file_path`.

        Parameters
        ----------
        file_path: str
            The string used to identify QA calibration data entry. See the docstring for
            ``add_entry`` for more detailed comments.
        eline: str
            Emission line name. Must match the emission line names used in calibration data.
            Typical format: ``Fe_K``, ``S_K`` etc.
        """
        n_item = self.find_entry_index(file_path)
        for n, settings in enumerate(self.calibration_settings):
            sel_flag = n == n_item  # Set 'selected' to True only if n == n_item
            if eline in settings["element_lines"]:
                self.calibration_settings[n]["element_lines"][eline]["selected"] = sel_flag
        self.update_emission_line_list()

    def is_eline_selected(self, eline, file_path):
        r"""
        Check if the emission line in the standard `file_path` is selected.
        Use `get_file_path_list()` to get the list of file paths arranged in the order
        in which standards were loaded.

        It is assumed that `eline` exists in the standard `file_path`.

        Parameters
        ----------
        file_path: str
            The string used to identify QA calibration data entry. See the docstring for
            ``add_entry`` for more detailed comments.
        eline: str
            Emission line name. Must match the emission line names used in calibration data.
            Typical format: ``Fe_K``, ``S_K`` etc.

        Returns
        -------
        bool
            `True` if eline is selected, `False` otherwise
        """
        n_item = self.find_entry_index(file_path)
        return self.calibration_settings[n_item]["element_lines"][eline]["selected"]

    def update_emission_line_list(self):
        r"""
        Update the internal list of emission lines. Also check and adjust if necessary
        the selected (assigned) calibration data entry for each emission line.

        This function is called by ``load_entry``, ``add_entry``, ``remove_entry`` and
        ``select_eline`` functions, since the set of emission lines is changed during
        those operations. If manual changes are made to loaded calibration data entries,
        this function has to be called again before any other operation is performed.
        Particularly, it is recommended that the function is called after changing the
        selection of calibration sources for emission lines.
        """

        # The emission lines are arranged in the list in the order in which they appear
        elines_all = set()
        self.active_emission_lines = []  # Clear the list, it is recreated
        for data in self.calibration_data:
            for ln in data["element_lines"].keys():
                if ln not in elines_all:
                    elines_all.add(ln)
                    self.active_emission_lines.append(ln)

        # Make sure that emission line is selected only once
        elines_selected = set()  # We make sure that each emission line is selected
        for settings in self.calibration_settings:
            for k, v in settings["element_lines"].items():
                if v["selected"]:
                    if k in elines_selected:
                        # If eline is in the set, then deselect it (it is selected twice)
                        v["selected"] = False
                    else:
                        # Add to the list
                        elines_selected.add(k)

        elines_not_selected = elines_all - elines_selected

        if elines_not_selected:
            for settings in self.calibration_settings:
                for k, v in settings["element_lines"].items():
                    if k in elines_not_selected:
                        v["selected"] = True
                        elines_not_selected.remove(k)

    def get_eline_info_complete(self, eline):
        r"""
        Collects emission line data from each loaded calibration entry and return it as a list
        of dictionaries. The size of the list determines the number of entries in which the
        emission line is present. If the emission line is not present in any loaded calibration set,
        then empty list is returned.

        The returned representation of complete calibration data is used in GUI interface for
        selection of calibration data sources.

        Parameters
        ----------

        eline: str

            Emission line name. Must match the emission line names used in calibration data.
            Typical format: ``Fe_K``, ``S_K`` etc.

        Returns
        -------

        eline_info: list(dict)

            List of dictionaries that hold calibration data for the given emission line.
            The length of the list is equal to the number of calibration data entries that
            contain information on the emission line ``eline``. Each dictionary contains
            the following fields: ``eline_data`` - reference to ``eline`` part of the
            calibration data entry, ``eline_settings`` - reference to ``eline`` part of
            the calibration settings entry (can be used to change the selection status
            of the entry), ``standard_data`` - reference to the full calibration data entry,
            ``standard_settings`` - reference to the full settings data entry. References
            to the full entries seem redundant, but they can be used to access data,
            which is common to the entry.
        """
        eline_info = []
        for n in range(len(self.calibration_data)):
            try:
                # Include only references to
                data = self.calibration_data[n]["element_lines"][eline]
                settings = self.calibration_settings[n]["element_lines"][eline]
                record = {}
                record["standard_data"] = self.calibration_data[n]
                record["standard_settings"] = self.calibration_settings[n]
                record["eline_data"] = data
                record["eline_settings"] = settings
                eline_info.append(record)
            except Exception:
                pass
        return eline_info

    def get_eline_calibration(self, eline):
        r"""
        Returns calibration data for the emission line ``eline`` from the selected
        QA calibration data entry.

        Parameters
        ----------

        eline: str

            Emission line name. Must match the emission line names used in calibration data.
            Typical format: ``Fe_K``, ``S_K`` etc.

        Returns
        -------

        The dictionary with calibration settings. Keys ``density`` - density of the sample
        material in the standard scan (typically in ug/cm^2), ``fluorescence`` - collected
        fluorescence during calibration experiment due to the emission line ``eline``,
        ``incident_energy`` - incident beam energy (in keV) used during the calibration scan,
        ``detector_channel`` - name of the detector channel used in calibration experiment
        (``sum``, ``det1``, ``det2`` etc), ``scaler_name`` - name of the scaler used in
        calibration experiment (valid name from the scan dataset), ``distance_to_sample`` -
        distance from the sample to the detector in calibration scan (units are selected by
        the user, only the ratio between the distances in calibration and experimental scans
        matter, if distance-to-sample is 0, then no correction is applied)
        """
        for n in range(len(self.calibration_data)):
            if (eline in self.calibration_data[n]["element_lines"]) and (
                eline in self.calibration_settings[n]["element_lines"]
            ):
                data = self.calibration_data[n]["element_lines"][eline]
                settings = self.calibration_settings[n]["element_lines"][eline]
                if settings["selected"]:
                    e_info = {
                        "density": data["density"],
                        "fluorescence": data["fluorescence"],
                        "incident_energy": self.calibration_data[n]["incident_energy"],
                        "detector_channel": self.calibration_data[n]["detector_channel"],
                        "scaler_name": self.calibration_data[n]["scaler_name"],
                        "distance_to_sample": self.calibration_data[n]["distance_to_sample"],
                    }
                    return e_info
        return None

    def get_calibrations_selected(self):
        r"""
        Returns the dictionary of calibration data from selected data entries for each
        available emission line: keys - emission line names (``Fe_K``, ``S_K`` etc.),
        values - calibration data for the emission line returned by the function
        ``get_eline_calibration``. See docstring for ``get_eline_calibration`` for
        additional information.

        Returns
        -------

        info: dict(dict)

            Dictionary of dictionaries, which contains selected calibration data for each available
            emission line.
        """
        info = {}
        for eline in self.active_emission_lines:
            e_info = self.get_eline_calibration(eline)
            if e_info:
                info[eline] = e_info
        return info

    def set_experiment_detector_channel(self, detector_channel):
        r"""
        Set detector channel name for the currently processed experimental dataset. The
        channel name is used by the function ``apply_quantitative_normalization``.

        The channel names used in calibration and experimental scans must match.

        Parameters
        ----------

        detector_channel: str

            Name of the detector channel (``sum``, ``det1``, ``det2`` etc)
        """
        self.experiment_detector_channel = detector_channel

    def set_experiment_incident_energy(self, incident_energy):
        r"""
        Set incident energy value for the currently processed experimental dataset. The
        incident energy value is used by the function ``apply_quantitative_normalization``.

        The incident energy value used in calibration and experimental scans must match.

        Parameters
        ----------

        incident_energy: float

            The value of the incident energy in keV.
        """
        self.experiment_incident_energy = incident_energy

    def set_experiment_distance_to_sample(self, distance_to_sample):
        r"""
        Set distance-to-sample value for the currently processed experimental dataset. The
        distance-to-sample value is used by the function ``apply_quantitative_normalization``
        to compute the correction coefficient in case the calibration and experimental scans
        are performed at different distances.

        Parameters
        ----------

        distance-to-sample: float

            The distance between the sample and the detector. Any convenient units can
            be used. The same units must be used for calibration and experimental scans.
            Only the ratio between distance-to-sample values during the experimental
            and calibration scans is used for computations. If distance-to-sample is set
            to 0 for calibration or experimental scan, then the correction coefficient
            is not computed or applied (it is assumed that the calibration and experimental
            scans are performed with the same distance).
        """
        self.experiment_distance_to_sample = distance_to_sample

    def apply_quantitative_normalization(
        self, data_in, *, scaler_dict, scaler_name_default, data_name, name_not_scalable=None
    ):

        r"""
        Apply quantitative normalization to the experimental XRF map ``data_in``. The quantitative
        normalization is applied if the following conditions are met:

        -- ``data_name`` is not in the list ``name_not_scalable``.

        -- ``data_name`` represents an emission line for which quanitative calibration is available
           in one of the entries.

        -- scaler name specified the selected quantitative calibration entry is available
           in ``scaler_dict``.

        -- detector channel selected for the experimental data processing matches the detector
           channel used during calibration.

        If there is significant difference in incident beam energy set for calibration and
        experimental scans, the normalization is still performed, but warning is printed in
        the terminal.

        If quantitative normalization is not performed, but ``scaler_name_fixed`` is a valid key
        of ``scaler_dict``, then regular normalization by the scaler is performed.

        If normalization can not be performed, then the reference to the input array ``data_in``
        is returned.

        The boolean value ``is_quant_normalization_applied`` is set True only if QUANTITATIVE
        normalization is applied.

        Parameters
        ----------

        data_in: ndarray

            XRF map (shape (ny,nx)) needs to be normalized.

        scaler_dict: dict(key: str, value: ndarray)

            Dictionary of the available scaler data for the experimental scan
            (key: scaler name, value: map of scaler values with shape (ny, nx)).

        scaler_name_default: str or None

            Name of the default scaler that is selected for the normalization of experimental
            data. If ``scaler_name_fixed`` is None or not found in the dictionary ``scaler_dict``,
            then normalization is not applied. If quanitative calibration data is available
            for the emission line ``data_name``, then ``scaler_name_fixed`` is ignored and
            the scaler name that was applied during quantitative calibration is applied.
            If ``scaler_name_fixed`` is ``None``, then normalization is applied only if
            quantitative calibration is available for the emission line ``data_name``.

        data_name: str

            The name of XRF map ``data_in``. This may be emission line name (such as ``Fe_K``, ``S_K``)
            or the name of the scalar or positional data. If ``data_name`` represents an emission line,
            then quantitative calibration may be applied to the data.

        name_not_scalable: list or None

            List of map names that are not supposed to be normalized (some scalers, time, positions
            should not be normalized by the scaler). Normalization is skipped for this data.

        Returns
        -------

        data_arr: ndarray

            Output XRF map (normalized or not normalized). If normalization is not applied, then
            output array is the reference to the input array (``data_arr is data_in == True``).

        is_quant_normalization_applied: bool

            Indicates if quantitative normalization was applied to the input map. If normalization
            by fixed (selected) scaler is applied, then the returned value is False.
        """

        logger.debug(
            f"Starting quantiative normalization with scan parameters:\n"
            f"    Detector channel: '{self.experiment_detector_channel}'\n"
            f"    Distance to sample: {self.experiment_distance_to_sample}\n"
            f"    Incident energy: {self.experiment_incident_energy}"
        )

        is_quant_normalization_applied = False

        # Check input data integrity
        if data_in is None:
            return data_in, is_quant_normalization_applied

        # Data name should not necessarily be emission line name. Some data should not be normalized.
        #   Return input data without change
        if name_not_scalable and (data_name in name_not_scalable):
            return data_in, is_quant_normalization_applied

        e_info = self.get_eline_calibration(data_name)
        # Scaler is not strictly required to perform quanitative calibration, so it is allowed
        #   to run the function without a scaler. The scaler name is None if the standard was
        #   processed without normalization, so the sample data will not be normalized as well.
        #   But the results are expected to be much better if the scaler is used.

        run_quant = False
        if e_info:
            run_quant = True
            e_info_scaler = e_info["scaler_name"]

            if (
                (self.experiment_detector_channel is None)
                or (e_info["detector_channel"] is None)
                or (self.experiment_detector_channel.lower() != e_info["detector_channel"].lower())
            ):
                # Detector channels must match. This is critical.
                logger.error(
                    f"Emission line: {data_name}. Mismatch between channels used "
                    f"for calibration ('{e_info['detector_channel']}') and current experimental "
                    f"('{self.experiment_detector_channel}') scans. "
                    f"Quantitative normalization will not be performed."
                )
                run_quant = False

            if (
                (self.experiment_incident_energy is None)
                or (e_info["incident_energy"] is None)
                or not math.isclose(self.experiment_incident_energy, e_info["incident_energy"], abs_tol=0.001)
            ):
                # Still do normalization, but print the warning (result may be inaccurate)
                logger.warning(
                    f"Emission line {data_name}. Incident energy for standard "
                    f"('{e_info['incident_energy']}') and experimental "
                    f"('{self.experiment_incident_energy}') scans don't match. "
                    f"Quantitative normalization will still be performed, but the results may "
                    f"be inaccurate."
                )

            if (e_info_scaler is not None) and (e_info_scaler not in scaler_dict):
                run_quant = False
                logger.error(
                    f"Emission line {data_name}. Scaler '{e_info_scaler}' is not available "
                    f"for this dataset. Quantitative normalization is skipped"
                )

        if run_quant:
            # Quantitative calibration for the emission line is loaded, so normalization is
            #   performed. The scaler used to obtain calibration is used. Note, that
            #   if the scaler is None, then the function returns data without change
            if e_info_scaler is None:
                scaler = None
            else:
                scaler = scaler_dict[e_info_scaler]
            data_arr = normalize_data_by_scaler(
                data_in=data_in, scaler=scaler, data_name=data_name, name_not_scalable=name_not_scalable
            )
            # Normalization function above returns reference if not transformations were applied
            #   Make a copy in this case.
            if data_arr is data_in:
                data_arr = data_in.copy()
            data_arr *= e_info["density"] / e_info["fluorescence"]
            # If distance to sample is set for calibration data and current scan, then apply correction
            #   If either value is ZERO, then don't perform the correction.
            r1 = e_info["distance_to_sample"]
            r2 = self.experiment_distance_to_sample
            if (
                (r1 is not None)
                and (r2 is not None)
                and (r1 > 0)
                and (r2 > 0)
                and not math.isclose(r1, r2, abs_tol=1e-20)
            ):
                # Element density increase as the distance becomes larger
                #   (fluorescence is reduced as r**2)
                data_arr *= (r2 / r1) ** 2
                logger.info(
                    f"Emission line {data_name}. Correction for distance-to_sample was performed "
                    f"(standard: {r1}, sample: {r2})"
                )
            else:
                logger.info(
                    f"Emission line {data_name}. Correction for distance-to_sample was skipped "
                    f"(standard: {r1}, sample: {r2})"
                )
            is_quant_normalization_applied = True
        else:
            # The following condition also takes care of the case when 'scaler_name_fixed' is None
            if scaler_name_default in scaler_dict:
                data_arr = normalize_data_by_scaler(
                    data_in=data_in,
                    scaler=scaler_dict[scaler_name_default],
                    data_name=data_name,
                    name_not_scalable=name_not_scalable,
                )
            else:
                data_arr = data_in

        return data_arr, is_quant_normalization_applied
