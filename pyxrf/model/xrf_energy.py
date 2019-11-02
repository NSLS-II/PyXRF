
from .load_data_from_db import make_hdf
from .command_tools import pyxrf_batch

import logging
logger = logging.getLogger()


def xrf_energy_batch(*args, **kwargs):
    try:
        xrf_energy_batch_api(*args, **kwargs)
    except BaseException as ex:
        msg = f"Processing is incomplete! Exception was raised during execution:\n   {ex}"
        logger.error(msg)
    else:
        logger.info("Processing was completed successfully.")


def xrf_energy_batch_api(start_id=None, end_id=None, *, param_file_name, data_files=None, wd=None,
                     processed_channel="sum"):
    print("Hello")
