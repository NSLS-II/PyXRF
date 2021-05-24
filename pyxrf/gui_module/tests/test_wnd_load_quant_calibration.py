import json
import yaml

# =========================================================================
#   Data for testing WndLoadQuantitativeCalibration class

# Two sets of quantitative calibration data for demonstration of GUI layout
#   Real data will be used when GUI is integrated with the program.
#   The data are representing contents of JSON files, so they should be loaded
#   using 'json' module.
json_quant_calib_1 = """
    {
        "name": "Micromatter 41147",
        "serial": "v41147",
        "description": "GaP 21.2 (Ga=15.4, P=5.8) / CaF2 14.6 / V 26.4 / Mn 17.5 / Co 21.7 / Cu 20.3",
        "element_lines": {
            "Ga_K": {
                "density": 15.4,
                "fluorescence": 6.12047035678267e-05
            },
            "Ga_L": {
                "density": 15.4,
                "fluorescence": 1.1429814846741588e-05
            },
            "P_K": {
                "density": 5.8,
                "fluorescence": 3.177988019213722e-05
            },
            "F_K": {
                "density": 7.105532786885246,
                "fluorescence": 1.8688801284649113e-07
            },
            "Ca_K": {
                "density": 7.494467213114753,
                "fluorescence": 0.0005815345261894806
            },
            "V_K": {
                "density": 26.4,
                "fluorescence": 0.00030309931019669974
            },
            "Mn_K": {
                "density": 17.5,
                "fluorescence": 0.0018328847495676865
            },
            "Co_K": {
                "density": 21.7,
                "fluorescence": 0.0014660067400157218
            },
            "Cu_K": {
                "density": 20.3,
                "fluorescence": 6.435121428993609e-05
            }
        },
        "incident_energy": 12.0,
        "detector_channel": "sum",
        "scaler_name": "i0",
        "distance_to_sample": 1.0,
        "creation_time_local": "2020-05-27T18:49:14+00:00",
        "source_scan_id": null,
        "source_scan_uid": null
    }
    """

json_quant_calib_2 = """
    {
        "name": "Micromatter 41164 Name Is Long So It Has To Be Printed On Multiple Lines (Some More Words To Make The Name Longer)",
        "serial": "41164",
        "description": "CeF3 21.1 / Au 20.6",
        "element_lines": {
            "F_K": {
                "density": 6.101050068482728,
                "fluorescence": 2.1573457185882552e-07
            },
            "Ce_L": {
                "density": 14.998949931517274,
                "fluorescence": 0.0014368335445700924
            },
            "Au_L": {
                "density": 20.6,
                "fluorescence": 4.4655757003090785e-05
            },
            "Au_M": {
                "density": 20.6,
                "fluorescence": 3.611978659032483e-05
            }
        },
        "incident_energy": 12.0,
        "detector_channel": "sum",
        "scaler_name": "i0",
        "distance_to_sample": 2.0,
        "creation_time_local": "2020-05-27T18:49:53+00:00",
        "source_scan_id": null,
        "source_scan_uid": null
    }
    """  # noqa: E501

# The data is structured the same way as in the actual program code, so transitioning
#   to real data will be simple
quant_calib = [
    [json.loads(json_quant_calib_1), {"file_path": "/path/to/quantitative/calibration/file/standard_41147.json"}],
    [
        json.loads(json_quant_calib_2),
        {
            "file_path": "/extremely/long/path/to"
            "/quantitative/calibration/file/so/it/had/to/be/"
            "printed/on/multiple/lines/standard_41164.json"
        },
    ],
]


# The following list is to demonstrate how 'View' button works. Data is treated
#   differently in the actual code, but the resulting format will be similar.
quant_calib_json = [
    yaml.dump(quant_calib[0][0], default_flow_style=False, sort_keys=False, indent=4),
    yaml.dump(quant_calib[1][0], default_flow_style=False, sort_keys=False, indent=4),
]
