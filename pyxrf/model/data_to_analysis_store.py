import time
import copy
import numpy as np
import logging
try:
    from event_model import compose_run
except ModuleNotFoundError:
    pass

from pyxrf.api import db, db_analysis


def fitting_result_sender(hdr, result, param):
    """Transfer fitted results to generator.

    Parameters
    ----------
    hdr : header
    result : dict of 2D array
        fitting results
    param : dict
        fitting parameters
    """
    start_new = dict(hdr.start)
    start_new['param'] = param
    yield 'start', start_new
    yield 'descriptor', {k: {'shape': hdr.start.shape, 'dtype': 'array', 'source': 'pyxrf fitting'} for k in result.keys()}
    time_when_analysis = time.time()
    time_result = {k : time_when_analysis  for k in result.keys()}
    events = {'data': result, 'time': time_result, 'seq_num' : 1}
    yield 'event', events
    yield 'stop', {}


class ComposeDataForDB:
    """Compose and validate data to format databroker
    can take.
    """
    def __init__(self):
        pass

    def __call__(self, name, doc):
        return name, getattr(self, name)(doc)

    def start(self, doc):
        metadata = {'raw_uid': doc['uid'],
                    'raw_scan_id': doc['scan_id'],
                    'processor_parameters': doc['param']}
        self.compose_run_bundle = compose_run(metadata=metadata)
        return self.compose_run_bundle.start_doc

    def datum(self, doc):
        return doc

    def resource(self, doc):
        return doc

    def descriptor(self, doc):
        name = 'primary'
        data_keys = doc
        self.compose_descriptor_bundle = self.compose_run_bundle.compose_descriptor(
                name=name, data_keys=data_keys,
                object_names=None, configuration={}, hints=None)
        return self.compose_descriptor_bundle.descriptor_doc

    def event(self, doc):
        event_doc = self.compose_descriptor_bundle.compose_event(
                data=doc['data'],
                timestamps=doc['time'],
                seq_num=doc['seq_num'])
        return event_doc

    def stop(self, doc):
        return self.compose_run_bundle.compose_stop()


def save_data_to_db(uid, result, param,
                    db=db, db_analysis=db_analysis):
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
    print('saving data to db for {}'.format(uid))
    hdr = db[uid]
    gen = fitting_result_sender(hdr, res, param)
    name, start_doc = next(gen)

    #assert name == 'start'
    processor = ComposeDataForDB()
    # Push the start_doc through.
    _, processed_doc = processor('start', start_doc)
    print(f"start info {processed_doc}")
    # insert to analysis store
    db_analysis.insert('start', processed_doc)

    for name, doc in gen:
        print(name)
        _, processed_doc = processor(name, doc)
        processed_doc.pop('id', None)
        # insert to analysis store
        db_analysis.insert(name, processed_doc)


# below to be removed
def simulated_result():
    result = {}
    for i in range(5):
        result[f'data_{i}'] = np.ones([10,10]) + i
    return result

res = simulated_result()
param = {'a' : 1, 'b' : 2}
save_data_to_db(54132, db, db_analysis, res, param)
