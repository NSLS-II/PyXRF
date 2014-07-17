from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import six, sys, datetime
from PyQt4 import QtCore, QtGui
from vistools.qt_widgets.displaydict import RecursiveTreeWidget

_defaults = {
    "num_search_rows" : 1,
    "search_keys" : ["no search keys"],
    "search_key_descriptions" : [("No search keys were provided.  This widget "
                                 "will not do anything")],
}

class QueryMainWindow(QtGui.QMainWindow):
    """
    QueryMainWindow docstring
    """
    # dict1 : search query, dict2 : unique search id, dict3 : run_header dict
    add_btn_sig = QtCore.Signal(dict, dict, dict)
    search_btn_sig = QtCore.Signal(dict)

    def __init__(self, parent=None, keys=None, key_descriptions=None):
        """
        init docstring
        """
        QtGui.QMainWindow.__init__(self, parent)
        self.setWindowTitle('Query example')
        self._query_widget = QueryWidget(keys=keys,
                                         key_descriptions=key_descriptions)
        dock = QtGui.QDockWidget()
        dock.setWidget(self._query_widget._query_input)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, dock)
        self.setCentralWidget(self._query_widget._results_tree)
        # connect signals to signals
        self._query_widget.add_btn_sig.connect(self.add_btn_sig)
        self._query_widget.search_btn_sig.connect(self.search_btn_sig)

    def register_search_function(self, func):
        """
        Parameters
        ----------
        func : Function
            This function must take a dictionary parameter as input
        """
        self._search_func = func
        self.search_btn_sig.connect(self.search)

    def register_add_function(self, func):
        self._add_func = func

    @QtCore.Slot(list)
    def update_search_results(self, results):
        """
        Pass through function to update the search results in the results widget

        Parameters
        ----------
        results : array, list, object

        """
        self._query_widget.update_search_results(results)

    @QtCore.Slot(dict)
    def search(self, a_dict):
        """
        This function gets called when the search button is clicked
        """
        return_val = self._search_func(a_dict)
        self.update_search_results(return_val)


class QueryWidget(QtCore.QObject):
    """
    QueryWidget docstring
    """
    # external handles for the add button and search button
    add_btn_sig = QtCore.Signal(dict, dict, dict)
    search_btn_sig = QtCore.Signal(dict)


    def __init__(self, keys=None, key_descriptions=None, *args, **kwargs):
        """

        Parameters
        ----------
        keys : List
            List of keys to use as search terms
        """
        # call up the inheritance chain
        super(QueryWidget, self).__init__(
            QtGui.QWidget.__init__(self, *args, **kwargs))
        # init the defaults
        if keys is None:
            keys = _defaults["search_keys"]
        if key_descriptions is None:
            key_descriptions = _defaults["search_key_descriptions"]

        self._keys = keys
        self._key_descriptions = key_descriptions
        # set up the query widget
        self._query_input = self.construct_query()

        # set up the results widget
        self._results_tree = self.construct_results()

    def construct_query(self):
        """
        Construct the query widget

        Returns
        -------
        QtGui.QGroupBox
            group box that contains the query widget
        """
        # declare the group box
        _query = QtGui.QGroupBox(title="Query")

        # declare the search button
        search_btn = QtGui.QPushButton(text="&Search")
        # connect the search buttons clicked signal to the method which parses
        # the text boxes to create a search dictionary that gets emitted by the
        # externally facing search_btn_sig QtCore.Signal
        search_btn.clicked.connect(self.parse_search_boxes)
        # declare the query widget
        query_widg = self.construct_query_input()

        # declare the layout as a vertical box layout
        layout = QtGui.QVBoxLayout()

        # add the widgets to the layout
        layout.addWidget(query_widg)
        layout.addWidget(search_btn)
        # set the layout of the group box
        _query.setLayout(layout)
        # return the widget
        return _query

    def construct_results(self):
        """
        Construct the results widget

        Returns
        -------
        QtGui.QGroupBox
            group box that contains the results widget along with the 'add'
            button
        """
        # declare a group box
        _results = QtGui.QGroupBox(title="Results")
        # declare the layout as a vertical box
        layout = QtGui.QVBoxLayout()

        # declare the tree widget
        self._tree = RecursiveTreeWidget()
        # declare the "add to canvas" button
        add_btn = QtGui.QPushButton(text="&Add selected to canvas")
        # connect the add button clicked signal to the externally facing
        # "add_btn_signal" QtCore.SIGNAL
        add_btn.clicked.connect(self.add_clicked)

        # add the tree widget to the layout
        layout.addWidget(self._tree)
        # add the button to the layout
        layout.addWidget(add_btn)

        # set the layout of the group box
        _results.setLayout(layout)

        # return the results group box
        return _results

    def construct_query_input(self, keys=None, key_descriptions=None):
        """
        Construct the input boxes for the query.

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
        if key_descriptions is None:
            if self._key_descriptions is None:
                key_descriptions = keys
            else:
                key_descriptions = self._key_descriptions

        self._keys = keys
        self._key_descriptions = key_descriptions
        # declare a vertical layout
        vert_layout = QtGui.QVBoxLayout()

        try:
            # if the input boxes dictionary exists, empty it
            self._input_boxes.clear()
        except AttributeError:
            # create a new dicrionary
            self._input_boxes = {}

        # loop over the keys to create an input box for each key
        for key in keys:
            # declare a new horizontal layout
            horz_layout = QtGui.QHBoxLayout()
            # declare the label
            lbl = QtGui.QLabel(key)
            lbl.setToolTip(self._key_descriptions[key])
            # declare the input box
            input_box = QtGui.QLineEdit(parent=None)
            # add the input box to the input_boxes dict
            self._input_boxes[key] = input_box
            # add the widgets to the layout
            horz_layout.addWidget(lbl)
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

    ####################
    # Runtime behavior #
    ####################
    @QtCore.Slot()
    def add_clicked(self):
        """
        Figure out which result is clicked and emit the add_btn_sig with that
        dictionary
        """
        print("add_clicked")
        result_dict = {}
        # ask the tree nicely for its currently selected dictionary
        # unique_id = tree.get_current()
        self.add_btn_sig.emit(self._search_dict, result_dict)
        pass

    @QtCore.Slot()
    def parse_search_boxes(self):
        """
        Parse the search boxes to set up the query dictionary and emit it with
        the search_btn_sig Signal
        """
        # declare the search dict
        print("parse_search_boxes")
        self._search_dict = {}
        try:
            # loop over the list of input boxes to extract the search string
            for key in self._input_boxes.keys():
                # get the text from the input box
                qtxt = self._input_boxes[key].text()
                # convert the qstring to a python string
                txt = str(qtxt)
                if txt != '':
                    # add the search key to the dictionary if the input box is
                    # not empty
                    self._search_dict[key] = str(txt)
        except AttributeError:
            # the only time this will be caught is in the initial setup and it
            # is therefore ok to ignore this error
            pass
        # once the dictionary is constructed, emit it as a signal
        self.search_btn_sig.emit(self._search_dict)

    @QtCore.Slot(list)
    def update_search_results(self, results):
        """

        Parameters
        ----------
        results : array, list, object

        """
        self._tree.fill_widget(results)

    def update_query_keys(self, keys, key_descriptions=None):
        """
        Function re-makes the query search box based on the new keys

        Parameters
        ----------
        keys : list
            List of keys that can be used as search terms
        """
        if keys is None:
            return
        if key_descriptions is None:
            self._key_descriptions = keys
        # stash the keys
        self._keys = keys
        layout = self._query_input.layout()
        # empty the current query widget
        # remove the old search query
        old_query = layout.takeAt(0)
        # todo probably need to delete this old_query object!

        # remove the stretch
        stretch = layout.takeAt(0)
        # todo probably need to delete this stretch!

        # remove the button or buttons
        btns = layout.takeAt(0)

        new_query = self.construct_query_input(
            keys=keys, key_descriptions=key_descriptions)

        layout.addWidget(new_query)
        layout.addWidget(btns)
        layout.addStretch()



if __name__ == "__main__":
    from metadataStore.userapi.commands import search_keys_dict, search_dict
    app = QtGui.QApplication(sys.argv)

    test_dict = search_keys_dict
    key_descriptions = {}
    for key in test_dict.keys():
        key_descriptions[key] = test_dict[key]["description"]
    qmw = QueryMainWindow(keys=test_dict.keys(),
                          key_descriptions=key_descriptions)

    qmw.register_search_function(search_dict)

    qmw.show()
    sys.exit(app.exec_())