import re
import textwrap


class ScanMetadataBase:
    """
    This is a base class. Instances of classes should be created from
    a child class, which has meaningful implementation of the ``_gen_default_descriptions``
    and ``_gen_print_order`` functions.
    """

    def __init__(self):

        # This dictionary contains key-value pairs that represent parameter values
        self._values = {}

        # The following dictionary contains key-description pairs, that represent
        #   user-friendly description of each key, that may be used for printing.
        #   It is not assumed that there is one-to-one match for the keys
        #   in 'values' and 'key_description' dictionaries. The 'key_descriptions'
        #   should contain comprehensive list of keys. It is also assumed that
        #   some keys in 'values' dictionary may not have matching descriptions.
        #   'key_descriptions' dictionary set at the different place in the program.
        #   In principle the descriptions may not be set at all.
        self.descriptions = self._gen_default_descriptions()

        self._print_order = self._gen_default_print_order()

    def _gen_default_print_order(self):
        """
        This function must have meaningful implementation in the child class
        """
        return []

    def _gen_default_descriptions(self):
        """
        This function must have meaningful implementation in the child class
        """
        return {}

    # The following methods override standard operators to enable direct access
    #   to ``self.values`` dictionary

    def __getitem__(self, key):
        return self._values[key]

    def __setitem__(self, key, value):
        self._values[key] = value

    def __delitem__(self, key):
        del self._values[key]

    def __contains__(self, key):
        return key in self._values

    def __iter__(self):
        return self._values.__iter__()

    def keys(self):
        return self._values.keys()

    def values(self):
        return self._values.values()

    def items(self):
        return self._values.items()

    def update(self, source_dict):
        self._values.update(source_dict)

    def get_metadata_dictionary(self):
        """
        Returns reference to metadata dictionary ``self._values``
        """
        return self._values

    def is_metadata_available(self):
        """
        Returns True if at least one metadata key:value pair is available.
        Otherwise returns False.
        """
        return bool(self._values)

    def get_formatted_output(self):
        """
        Returns formatted metadata in the form ready for printing.
        Formatting is performed based on specifications located in ``self._print_order``
        list. Key names are replaced by descriptions from ``self.descriptions`` if
        the available.
        """
        str_out = ""

        printed_keys = set()
        flag_empty_line = False  # Indicates if empty line was just printed

        def cap(s):
            """
            Capitalize the first character of the string. Leave the rest of the characters intact.
            (``capitalize()`` turns all characters except the first one to lower case)
            """
            return s[0].upper() + s[1:] if s else s

        for ppattern in self._gen_default_print_order():
            # We don't want to print multiple empty lines in a row
            if ppattern == "" and not flag_empty_line:
                str_out += "\n"
                flag_empty_line = True
                continue

            for key, v in self._values.items():
                if re.search(f"^{ppattern}$", key) and (key not in printed_keys):
                    flag_empty_line = False  # Something is getting printed
                    # Mark the entry as printed
                    printed_keys.add(key)
                    # Extracted printable expression for the key
                    s_key = key
                    if key in self.descriptions and self.descriptions[key]:
                        s_key = self.descriptions[key]
                        s_key = cap(s_key)

                    if isinstance(v, str):
                        # Wrap the string if it is too long ('fill' function does not change
                        #   short strings, so call it for every value string)
                        n_indent = len(s_key) + 2
                        text_width = 60
                        indent = " " * n_indent  # Size of the left margin
                        # Create lines with the same indent. Indent width should not be included
                        #   in the text width.
                        val = textwrap.fill(
                            v, width=text_width + n_indent, initial_indent=indent, subsequent_indent=indent
                        )
                        # Now remove spaces at the beginning of the line, since the key will be
                        #   printed instead of the spaces
                        val = val.lstrip()
                    else:
                        val = v

                    # Print the key
                    str_out += f"{s_key}: {val}\n"

        return str_out


class ScanMetadataXRF(ScanMetadataBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _gen_default_print_order(self):
        """
        Generates a list of strings, used to determine printing order of metadata
        The strings in the list are treated as regex strings:
        the symbols ^ and $ are added at the beginning and the end of each string.
        Empty string "" means that the empty string is inserted in the printout.
        The metadata entries are never repeated in the printout, so in each group
        the patterns are specified from more specific to more general.
        """
        print_order = [
            "scan_id",
            "scan_uid",
            "scan_instrument_name",
            "scan_instrument_id",
            "scan_end_station",
            "scan_time_start",
            "scan_time_start_utc",
            "scan_time_stop",
            "scan_time_stop_utc",
            "scan.*",
            "",
            "sample_name",
            "sample.*",
            "",
            "param_type",
            "param_input",
            "param_shape",
            "param_snake",
            "param_dwell",
            "param_theta",
            "param_theta_units",
            "param_delta",
            "param_delta_units",
            "param_fast_axis",
            "param_fast_axis_units",
            "param_slow_axis",
            "param_slow_axis_units",
            "param_interferometer_posX",
            "param_interferometer_posY",
            "param_interferometer_posZ",
            "param.*",
            "",
            "proposal_num",
            "proposal_title",
            "proposal.*",
            "",
            "experiment.*",
            "",
            "instrument.*",
            "",
            "(?!file).*",
            "",
            "file_format",
            "file_format_version",
            "file_software",
            "file_software_version",
            "file.*",
        ]
        return print_order

    def _gen_default_descriptions(self):
        descriptions = {
            # The descriptions are not capitalized. They can be capitalized
            #   before printing if needed.
            "scan_id": "scan ID",
            "scan_uid": "scan Unique ID",
            "scan_time_start": "start time",
            "scan_time_stop": "stop time",
            "scan_time_start_utc": "start time (UTC)",
            "scan_time_stop_utc": "stop time (UTC)",
            "scan_instrument_id": "beamline ID",
            "scan_instrument_name": "beamline name",
            "scan_end_station": "end station",
            "scan_exit_status": "exit status",
            "instrument_mono_incident_energy": "incident energy",
            "instrument_beam_current": "ring current, mA",
            "instrument_detectors": "detectors",
            "sample_name": "sample name",
            "experiment_plan_name": "plan name",
            "experiment_plan_type": "plan type",
            "experiment_fast_axis": "scan fast axis",
            "experiment_slow_axis": "scan slow axis",
            "proposal_num": "proposal #",
            "proposal_title": "proposal title",
            "proposal_PI_lastname": "PI last name",
            "proposal_saf_num": "proposal SAF #",
            "proposal_cycle": "cycle",
            "file_created_time": "file creation time",
            "file_format": "file format",
            "file_format_version": "file format version",
            "file_software": "software",
            "file_software_version": "version",
            "file_type": "file type",
            "param_type": "plan type",
            "param_input": "plan input parameters",
            "param_shape": "scan shape",
            "param_snake": "snaking",
            "param_dwell": "dwell time",
            "param_theta": "angle theta",
            "param_theta_units": "angle theta (units)",
            "param_delta": "delta",
            "param_delta_units": "delta (units)",
            "param_fast_axis": "fast axis",
            "param_fast_axis_units": "fast axis (units)",
            "param_slow_axis": "slow axis",
            "param_slow_axis_units": "slow axis (units)",
            "param_interferometer_posX": "initial position X (interferometer), pm",
            "param_interferometer_posY": "initial position Y (interferometer), pm",
            "param_interferometer_posZ": "initial position Z (interferometer), pm",
        }

        return descriptions

    def is_mono_incident_energy_available(self):
        """
        Returns True if data on monochromator incident energy is available.
        Otherwise returns False
        """
        return "instrument_mono_incident_energy" in self._values

    def get_mono_incident_energy(self):
        """
        Returns the value of the incident energy. Incident energy is an important
        parameter used in processing, so a separate function is created to fetch it.
        """
        return self._values["instrument_mono_incident_energy"]
