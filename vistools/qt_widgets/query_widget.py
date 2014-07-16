from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import six, sys
from PyQt4 import QtCore, QtGui
from vistools.qt_widgets.displaydict import RecursiveTreeWidget

_defaults = {
    "num_search_rows" : 1,
    "search_keys" : ["no search keys"],
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
    # external handle for the add button
    add_btn_sig = QtCore.Signal()


    def __init__(self, keys=None):
        """

        Parameters
        ----------
        keys : List
            List of keys to use as search terms
        """
        QtGui.QWidget.__init__(self)
        self._rows = []
        if keys is None:
            self._keys = _defaults["search_keys"]
        # set up the query widget
        self._query = self.construct_query()

        # set up the results widget
        self._results = self.construct_results()

        # declare a vertical box layout
        layout = QtGui.QVBoxLayout()
        # add the query widget as the first member of the vbox
        layout.addWidget(self._query)
        # add the results widget as the second member of the vbox
        layout.addWidget(self._results)

        self.setLayout(layout)

    def construct_query(self):
        """
        Construct the query widget

        Returns
        -------
        QtGui.QGroupBox
            group box that contains the query widget
        """
        # declare the group box
        _query = QtGui.QGroupBox(title="Query", parent=self)

        # declare the search button
        search_btn = QtGui.QPushButton(text="Search")
        # declare the query widget
        query_widg = self.construct_query_input()

        # declare the layout as a vertical box layout
        layout = QtGui.QVBoxLayout()

        # add the widgets to the layout
        layout.addWidget(query_widg)
        layout.addWidget(search_btn)
        # set the layout of the group box
        _query.setLayout(layout)
        # return the query group box
        return _query

    def construct_query_input(self, keys=None):
        """
        Construct the input boxes for the query

        Returns
        -------
        query_input : QtGui.QWidget
            The widget that contains the input boxes and labels

        Optional
        ----------
        keys : list
            List of keys to use for searching.
            Default behavior: use self._keys
        """
        # default behavior of keys input parameter
        if keys is None:
            keys = self._keys

        # declare a vertical layout
        vert_layout = QtGui.QVBoxLayout()

        # empty the input boxes dictionary
        try:
            self._input_boxes.clear()
        except Exception:
            self._input_boxes = {}

        # loop over the keys to create an input box for each key
        for key in keys:
            # declare a new horizontal layout
            horz_layout = QtGui.QHBoxLayout()
            # declare the input box
            input_box = QtGui.QLineEdit(parent=None)
            # add the input box to the input_boxes dict
            self._input_boxes[key] = input_box
            # add the widgets to the layout
            horz_layout.addWidget(QtGui.QLabel(key))
            horz_layout.addWidget(input_box)
            # set a dummy widget
            widg = QtGui.QWidget()
            widg.setLayout(horz_layout)
            # add the horizontal layout to the vertical layout
            vert_layout.addWidget(widg)

        query_input = QtGui.QWidget()
        query_input.setLayout(vert_layout)
        # return the vertical layout
        return query_input

    def construct_results(self):
        """
        Construct the results widget

        Returns
        -------
        QtGui.QGroupBox
            group box that contains the results widget along with the 'add'
            button
        """
        _results = QtGui.QGroupBox(title="Results", parent=self)
        # declare the layout as a vertical box
        layout = QtGui.QVBoxLayout()

        # declare the tree widget
        self._tree = RecursiveTreeWidget()
        # declare the "add to canvas" button
        add_btn = QtGui.QPushButton(text="Add selected to canvas", parent=self)
        # connect the add button clicked signal to the externally facing
        # "add_btn_signal" QtCore.SIGNAL
        add_btn.clicked.connect(self.add_btn_sig)

        # add the tree widget to the layout
        layout.addWidget(self._tree)
        # add the button to the layout
        layout.addWidget(add_btn)

        # set the layout of the group box
        _results.setLayout(layout)

        # return the results group box
        return _results
    
    ####################
    # Runtime behavior #
    ####################

    @QtCore.Slot(list)
    def search_results_slot(self, results):
        """

        Parameters
        ----------
        results : array, list, object

        """
        self._tree.fill_widget(results)

    def update_query_keys(self, keys):
        """
        Function re-makes the query search box based on the new keys

        Parameters
        ----------
        keys : list
            List of keys that can be used as search terms
        """
        if keys is None:
            return
        # stash the keys
        self._keys = keys
        layout = self._query.layout()
        # empty the current query widget
        # remove the old search query
        old_query = layout.takeAt(0)
        # todo probably need to delete this old_query object!

        # remove the button or buttons
        btns = layout.takeAt(1)

        new_query = self.construct_query_input(keys)

if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    qmw = QueryMainWindow()
    qmw.show()
    sys.exit(app.exec_())