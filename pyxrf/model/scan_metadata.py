
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
        str = ""
        for key, value in self._values:
            str += f"{key}: {value}\n"
        return str

class ScanMetadataXRF(ScanMetadataBase):

    def __init__(self):
        super().__init__()

    def _gen_default_print_order(self):
        """
        Generates a list of strings, used to determine printing order of metadata
        The strings in the list are treated as regex strings:
        the symbols ^ and $ are added at the beginning and the end of each string.
        Empty string "" means that the empty string is inserted in the printout.
        The metadata entries are never repeated in the printout, so in each group
        the patterns are specified from more specific to more general.
        """
        print_order = ["scan_id", "scan_uid", "scan_instrument_name",
                          "scan_instrument_id", "scan_time_start",
                          "", "sample_name", "sample.*",
                          "", "proposal_num", "proposal_title", "proposal.*",
                          "", "experiment.*",
                          "", "instrument.*"]
        return print_order

    def _gen_default_descriptions(self):
        descriptions = {
            # The descriptions are not capitalized. They can be capitalized
            #   before printing if needed.
            "scan_id": "scan ID",
            "scan_uid": "scan Unique ID",
            "scan_time_start": "start time",
            "scan_time_stop": "stop time",
            "scan_instrument_id": "beamline ID",
            "scan_instrument_name": "beamline name",
            "scan_exit_status": "exit status",

            "instrument_mono_incident_energy": "incident energy",
            "instrument_beam_current": "beam current",
            "instrument_detectors": "detectors",

            "sample_name": "sample name",

            "experiment_plan_name": "plan name",
            "experiment_plan_type": "plan type",

            "proposal_num": "proposal #",
            "proposal_title": "proposal title",
            "proposal_PI_lastname": "PI last name",
            "proposal_saf_num": "proposal SAF #",
            "proposal_cycle": "cycle"
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
        return self._values['instrument_mono_incident_energy']