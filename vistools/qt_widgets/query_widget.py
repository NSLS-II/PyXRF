from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import six
from PyQt4 import QtCore, QtGui
import sys

_defaults = {
    "num_search_rows" : 1,
    "search_keys" : None,
}


class QueryMainWindow(QtGui.QMainWindow):
    """
    QueryMainWindow docstring
    """
    def __init__(self, parent=None):
        """
        init docstring
        """
        QtGui.QMainWindow.__init__(self, parent)
        self.setWindowTitle('Query example')
        self._query = QueryWidget()
        self.setCentralWidget(self._query)


class QueryWidget(QtGui.QWidget):
    """
    QueryWidget docstring
    """
    def __init__(self, keys=None):
        QtGui.QWidget.__init__(self)
        self._rows = []
        if keys is None:
            self._keys = keys
        self.construct()

    def construct(self):
        pass

    def add_row(self):
        pass

    def _add_first_row(self):
        pass

    def get_keys_combobox(self):
        pass