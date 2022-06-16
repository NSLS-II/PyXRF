# Use this file if you are importing into an interactive IPython session.
# Use 'pyxrf.api_dev' if you are importing PyXRF API into a custom script.


from .api_dev import *  # noqa: F401, F403
from pyxrf import __version__ as pyxrf_version


def pyxrf_api():
    r"""
    =========================================================================================
    Module ``pyxrf.api`` supports the following functions:

        Loading data:
          make_hdf - load XRF mapping data from databroker

        Data processing:
          pyxrf_batch - batch processing of XRF maps
          build_xanes_map - generation and processing of XANES maps

        Dask client:
          dask_client_create - returns Dask client for use in batch scripts

        Simulation of datasets:
          gen_hdf5_qa_dataset - generate quantitative analysis dataset
          gen_hdf5_qa_dataset_preset_1 - generate the dataset based on preset parameters

        VIEW THIS MESSAGE AT ANY TIME: pyxrf_api()

    For more detailed descriptions of the supported functions, type ``help(<function-name>)``
    in IPython command prompt.
    =========================================================================================
    """
    version = f"""
    =========================================================================================
    PyXRF version: {pyxrf_version}"""
    print(version + pyxrf_api.__doc__)


pyxrf_api()
