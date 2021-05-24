import time
import numpy as np

# import logging
try:
    from event_model import compose_run
except ModuleNotFoundError:
    pass

import pyxrf
from pyxrf.api import db, db_analysis


def fitting_result_sender(hdr, result, **kwargs):
    """Transfer fitted results to generator.

    Parameters
    ----------
    hdr : header
    result : dict of 2D array
        fitting results
    kwargs : dict
        fitting parameters, exp summed spectrum, fitted summed spectrum
    """
    start_new = dict(hdr.start)
    param = kwargs.get("param", None)
    if param is not None:
        start_new["param"] = param
    start_new["pyxrf_version"] = pyxrf.__version__
    yield "start", start_new
    # first descriptor
    yield "descriptor", {
        "data_keys": {
            "element_name": {"shape": [1], "dtype": "string", "source": "pyxrf fitting"},
            "map": {"shape": hdr.start.shape, "dtype": "array", "source": "pyxrf fitting"},
        },
        "name": "primary",
    }
    time_when_analysis = time.time()
    for i, (k, v) in enumerate(result.items()):
        data = {"element_name": k, "map": v}
        timestamps = {"element_name": time_when_analysis, "map": time_when_analysis}
        event = {"data": data, "seq_num": i + 1, "timestamps": timestamps}
        yield "event", event
    # second descriptor
    yield "descriptor", {
        "data_keys": {
            "summed_spectrum_experiment": {"shape": [4096], "dtype": "array", "source": "experiment"},
            "summed_spectrum_fitted": {"shape": [4096], "dtype": "array", "source": "pyxrf fitting"},
        },
        "name": "spectrum",
    }
    data = {"summed_spectrum_experiment": kwargs["exp"], "summed_spectrum_fitted": kwargs["fitted"]}
    timestamps = {"summed_spectrum_experiment": time_when_analysis, "summed_spectrum_fitted": time_when_analysis}
    event = {"data": data, "seq_num": 1, "timestamps": timestamps}
    yield "event", event

    yield "stop", {}


class ComposeDataForDB:
    """Compose and validate data to format databroker
    can take.
    """

    def __init__(self):
        pass

    def __call__(self, name, doc):
        return name, getattr(self, name)(doc)

    def start(self, doc):
        metadata = {"raw_uid": doc["uid"], "raw_scan_id": doc["scan_id"], "processor_parameters": doc["param"]}
        self.compose_run_bundle = compose_run(metadata=metadata)
        return self.compose_run_bundle.start_doc

    def datum(self, doc):
        return doc

    def resource(self, doc):
        return doc

    def descriptor(self, doc):
        self.compose_descriptor_bundle = self.compose_run_bundle.compose_descriptor(
            name=doc["name"], data_keys=doc["data_keys"], object_names=None, configuration={}, hints=None
        )
        return self.compose_descriptor_bundle.descriptor_doc

    def event(self, doc):
        event_doc = self.compose_descriptor_bundle.compose_event(
            data=doc["data"], timestamps=doc["timestamps"], seq_num=doc["seq_num"]
        )
        return event_doc

    def stop(self, doc):
        return self.compose_run_bundle.compose_stop()


def save_data_to_db(uid, result, doc, db=db, db_analysis=db_analysis):
    """
    Save fitting result to analysis store.

    Parameters
    ----------
    uid : int
        run id
    result : dict of 2D array
        fitting results
    param : dict
        fitting parameters
    db : databroker, optional
        where exp data is saved
    db : analysis store, optional
        where analysis result is to be saved
    """
    print("saving data to db for {}".format(uid))
    hdr = db[uid]
    gen = fitting_result_sender(hdr, result, **doc)
    name, start_doc = next(gen)

    # assert name == 'start'
    processor = ComposeDataForDB()
    # Push the start_doc through.
    _, processed_doc = processor("start", start_doc)
    # insert to analysis store
    db_analysis.insert("start", processed_doc)

    for name, doc in gen:
        print(name)
        _, processed_doc = processor(name, doc)
        # print(processed_doc)
        processed_doc.pop("id", None)
        # insert to analysis store
        db_analysis.insert(name, processed_doc)


def get_analysis_result(raw_scan_id, db_analysis=db_analysis):
    """
    Get data from analysis store with given raw_scan_id=uid.
    More rich search can be implemented here.

    Parameters
    ----------
    raw_scan_id : int
        scan_id of original scan
    db : analysis store, optional
        where analysis result is to be saved

    Returns
    -------
    header
    """
    res = db_analysis(raw_scan_id=raw_scan_id)
    res = list(res)
    if not len(res):
        print("No analysis result is found for {}.".format(raw_scan_id))
        return
    res = sorted(res, key=lambda x: x.start.time)
    uid = res[-1].start.uid  # only latest result for now, to be modified.
    hdr = db_analysis[uid]
    return hdr


# below to be removed
def simulated_result():
    result = {}
    for i in range(5):
        result[f"data_{i}"] = np.ones([10, 10]) + i
    return result


# res = simulated_result()
# param = {'a' : 1, 'b' : 2}
# doc = {'param': param, 'exp': np.arange(10), 'fitted':np.arange(10)}
# save_data_to_db(54132, res, doc)
