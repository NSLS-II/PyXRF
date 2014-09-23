from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import six, sys, datetime
import sip
sip.setapi('QDateTime', 2)
from bubblegum import QtCore, QtGui
from bubblegum.qt_widgets.displaydict import RecursiveTreeWidget
from collections import defaultdict
from bubblegum.qt_widgets.query_widget import QueryMainWindow
import logging
logger = logging.getLogger(__name__)


def default_unique_id_func(nested_dict):
    """
    Defaults to picking out the "header_id" from the NSLS-II data broker header

    Parameters
    ----------
    nested_dict : dict
        The nested dictionary to generate a unique ID from

    Returns
    -------
    unique_id_dict: dict
        return {_defaults["unique_id_key"]:
                nested_dict[_defaults["unique_id_key"]]
        }
    """
    logger.debug("default_unique_id_func() in broker_query_example.py")
    unique_id_dict = {}
    # loop over the unique_id_keys in the defaults dictionary to create the
    # unique ID dictionary
    for key in _defaults["unique_id_key"]:
        unique_id_dict[key] = nested_dict[key]

    return unique_id_dict


def default_search_func(search_dict):
    """
    Defaults to calling the data broker's search function

    Parameters
    ----------
    search_dict : dict
        The search_dict gets unpacked into the databroker's search function

    Returns
    -------
    search_results: list
        The results from the data broker's search function

    Raises
    ------
    ImportError
        Raised if the metadatastore cannot be found
    ValueError
        Raised if the search dictionary is empty
    """
    logger.info("default_search_func() in broker_query_example.py")
    print(search_dict)
    # check to see if the dictionary is empty
    if len(search_dict) == 0:
        logger.error("search_dict has no keys. Raising a value error")
        raise ValueError("The search_dict input parameter has no keys")
    print(search_dict)
    logger.info("search_dict")
    try:
        from metadataStore.userapi.commands import search
        logger.info("Search command from metadataStore.userapi.commands "
                    "imported successfully")
    except ImportError:
        #todo add logging statement about import error
        logger.info("The data broker cannot be found, returning an empty "
                    "search")
        return _defaults["empty_search"]

    result=search(**search_dict)
    print(result)
    return result


def default_add_func(search_dict, unique_id_dict, result_dict,
                     path_to_current_node_list):
    """

    Parameters
    ----------
    search_dict : dict
        The search dictionary that was used to find the results
    unique_id_dict : dict
        A dictionary that, when unpacked into the search function, will
        guarantee to return the result_dict
    result_dict : dict
        The desired dictionary from the search function
    path_to_current_node_list : list
        A list of result_dict keys that, when iterated over, will result in the
        node that is currently selected in the tree widgt display.  See the
        docstring of QueryMainWindow.register_add_function()

    Returns
    -------
    add_successful : bool
        True: The add function executed successfully
        False: The add function did not execute successfully. Check the log.
    """
    print("default_add_func() in broker_query_example.py")
    try:
        from vistrails.api import add_module
    except ImportError:
        # todo switch this to using python logging
        print("VisTrails add_module in vistrails.api not found")
        return False

    try:
        # broker_module is a place-holder module that is meant to represent
        from vistrails.nsls2 import broker_module
    except ImportError:
        # todo switch this to using python logging
        print("broker_module in vistrails.nsls2 package not found")
        return False

    #todo add this to the vistrails canvas
    #add_module(identifier, broker_module, NSLS2|io, controller=None):

    raise NotImplementedError


_defaults = {
    "unique_id_func": default_unique_id_func,
    "unique_id_key": ["_id"],
    "search_func": default_search_func,
    "add_func": default_add_func,
}


if __name__ == "__main__":
    from metadataStore.userapi import commands
    handler = logging.StreamHandler()
    handler.setLevel('INFO')
    logger.addHandler(handler)
    commands.logger.addHandler(handler)
    logging.getLogger("bubblegum.qt_widgets.query_widget").addHandler(handler)
    app = QtGui.QApplication(sys.argv)

    test_dict = commands.search_keys_dict

    qmw = QueryMainWindow(keys=test_dict,
                          search_func=_defaults["search_func"],
                          add_func=_defaults["add_func"],
                          unique_id_func=_defaults["unique_id_func"],
                          )

    qmw.show()
    sys.exit(app.exec_())
