from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import six, sys, datetime
from PyQt4 import QtCore, QtGui
from vistools.qt_widgets.displaydict import RecursiveTreeWidget
from collections import defaultdict


_defaults = {
    "empty_search": {
        "No search results": None
    },
    "add_btn_text": "Add",
}


class QueryMainWindow(QtGui.QMainWindow):
    """
    QueryMainWindow docstring
    """
    # dict1 : search query
    # dict2 : unique search id
    # dict3 : run_header dict
    add_btn_sig = QtCore.Signal(dict, dict, dict)
    # dict :
    search_btn_sig = QtCore.Signal(dict)

    def __init__(self, keys, key_descriptions=None, parent=None,
                 search_func=None, add_func=None, add_btn_text=None,
                 unique_id_func=None):
        """
        init docstring

        Parameters
        ----------
        keys : list
            List of keys to use as search terms
        key_descriptions : list
            List of key descriptions which are used as the tool tips for the
            search key labels
        parent : QWidget
            Parent widget that knows about this one
        search_func : function
            Executes when the "search" button is pressed. search_func must take
            a dictionary as input
        add_btn_text : str
            Label for the add button
        """
        QtGui.QMainWindow.__init__(self, parent)
        self.setWindowTitle('Query example')
        self._query_controller = QueryController(keys=keys,
                                            key_descriptions=key_descriptions,
                                            add_btn_text=add_btn_text)
        dock = QtGui.QDockWidget()
        dock.setWidget(self._query_controller._query_input)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, dock)
        self.setCentralWidget(self._query_controller._results_tree)

        # connect the widget signals to the main window signals
        self._query_controller.add_btn_sig.connect(self.add_btn_sig)
        self._query_controller.search_btn_sig.connect(self.search_btn_sig)

        # connect the search button to the main window search function
        self.search_btn_sig.connect(self.search)

        # connect the add button to the main window add function
        self.add_btn_sig.connect(self.add)

        # set the defaults

        # register the functions
        self.register_search_function(search_func)
        self.register_add_function(add_func)
        self.register_unique_id_gen_func(unique_id_func)


    def register_search_function(self, search_func):
        """
        Function that sets the behavior on clicking the 'search' button

        Parameters
        ----------
        func : Function
            This function must take a dictionary parameter as input with the
            following signature: some_search_function(search_dict)

        """
        self._search_func = search_func
        search_btn_enabled = True
        if self._search_func is None:
            search_btn_enabled = False

        self._query_controller.enable_search_btn(is_enabled=search_btn_enabled)

    def register_add_function(self, add_func):
        """
        Function that sets the behavior on clicking the 'add' button

        Parameters
        ----------
        func : Function
            function that executes when the 'add' button is clicked. This
            function must have the signature;
            some_add_function(query_dict, unique_id_dict, result_dict,
                              path_to_node_list)
            where path_to_node_list is a series of keys that uniquely identify
            the currently selected node in the add widget when iterated over.

        Examples
        --------
        the following code will result in "node" being the currently selected
        node in the tree widget

        >>> node = result_dict
        >>> for key in path_to_node_list:
        >>>     node = node[key]

        """
        self._add_func = add_func
        add_btn_enabled = True
        if self._add_func is None:
            add_btn_enabled = False

        self._query_controller.enable_add_btn(is_enabled=add_btn_enabled)

    def register_unique_id_gen_func(self, unique_id_func):
        """

        Parameters
        ----------
        unique_id_func : function
            Function that generates a unique ID for a results dictionary.  For
            now, this function should probably just pick out the header_id
        """
        self._unique_id_func = unique_id_func

    @QtCore.Slot(list)
    def update_search_results(self, results):
        """

        Pass through function to update the search results in the
        results widget

        Parameters
        ----------
        results : array, list, object

        """
        self._query_controller.update_search_results(results)

    @QtCore.Slot(dict)
    def search(self, a_dict):
        """
        This function gets called when the search button is clicked
        """
        print("search() function in QueryMainWindow")
        return_val = self._search_func(a_dict)
        self.update_search_results(return_val)

    @QtCore.Slot(dict, dict, dict, list)
    def add(self, search_query_dict, unique_id_dict, run_header_dict,
            path_to_current_node_list):
        """
        This function gets called when the add button is clicked
        """
        print("add() function in QueryMainWindow")
        self._add_func(search_query_dict, unique_id_dict, run_header_dict,
                       path_to_current_node_list)

    def update_query_keys(self, query_keys, query_key_descriptions):
        """
        Simple pass-through function to update the query keys
        """
        self._query_controller.update_query_keys(
            query_keys=query_keys,
            query_key_descriptions=query_key_descriptions
        )


class QueryController(QtCore.QObject):
    """
    The QueryController is a QObject that contains the search widget which is a
    QDockWidget and the tree widget which is a QTreeWidget

    Attributes
    ----------
    _keys : list
        List of search keys that will be displayed in the _query_input widget
    _key_descriptions : list
        List of descriptions for the keys that will appear as a tool tip on
        mouse hover
    _query_input : QtGui.QWidget
        The widget that displays a series of text input boxes with a 'search'
        button
    _results_tree : vistools.qt_widgets.displaydict.RecursiveTreeWidget
        The widget that displays the results as a tree with an 'add' button
    _search_dict : dict
        Dictionary that was unpacked into the search function. This attribute
        gets stored every time the 'search' button gets clicked
    _search_results : list
        List of dictionaries that the search function returns

    Methods
    -------
    update_search_results(results_list)
        Populate the RecursiveTreeWidget with the results_list
    enable_add_btn(bool)
        Enable/disable the add button
    enable_search_btn(bool)
        Enable/disable the search button
    add()
        Function that executes when the 'add' button is clicked
    search()
        Function that executes when the 'search' button is clicked
    parse_search_boxes()
        Read the text from the search boxes to form a search dictionary, stored
        as _search_dict
    update_query_keys(keys, key_descriptions=None)
        Remake the query widget with new query keys and key_descriptions

    """
    # external handles for the add button and search button
    add_btn_sig = QtCore.Signal(dict, dict, dict, list)
    search_btn_sig = QtCore.Signal(dict)


    ############################################################################
    #                      Construction time behavior                          #
    ############################################################################
    def __init__(self, keys, key_descriptions=None, add_btn_text=None,
                 *args, **kwargs):
        """

        Parameters
        ----------
        keys : list
            List of keys to use as search terms
        key_descriptions : list
            List of key descriptions which are used as the tool tips for the
            search key labels
        add_btn_text : str
            Label for the add button
        """
        # call up the inheritance chain
        super(QueryController, self).__init__(
            QtGui.QWidget.__init__(self, *args, **kwargs))
        # init the defaults
        if key_descriptions is None:
            key_descriptions = {}
        if add_btn_text is None:
            add_btn_text = _defaults["add_btn_text"]

        self._keys = keys
        self._key_descriptions = defaultdict(lambda: '')
        self._key_descriptions.update(key_descriptions)
        # set up the query widget
        self._query_input = self.construct_query()

        # set up the results widget
        self._results_tree = self.construct_results(add_btn_text)

        self._search_dict = _defaults["empty_search"]

        self.update_search_results(self._search_dict)

    def construct_query(self):
        """
        Construct the query widget

        Returns
        -------
        QtGui.QGroupBox
            group box that contains the query widget
        """
        # declare the group box
        query = QtGui.QGroupBox(title="Query")

        # declare the search button
        self._search_btn = QtGui.QPushButton(text="&Search")
        # connect the search buttons clicked signal to the method which parses
        # the text boxes to create a search dictionary that gets emitted by the
        # externally facing search_btn_sig QtCore.Signal
        self._search_btn.clicked.connect(self.parse_search_boxes)
        # declare the query widget
        query_widg = self.construct_query_input()

        # declare the layout as a vertical box layout
        layout = QtGui.QVBoxLayout()

        # add the widgets to the layout
        layout.addWidget(query_widg)
        layout.addWidget(self._search_btn)
        # set the layout of the group box
        query.setLayout(layout)
        # return the widget
        return query

    def construct_results(self, add_btn_text):
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
        self._add_btn = QtGui.QPushButton(text=add_btn_text)
        # connect the add button clicked signal to the externally facing
        # "add_btn_signal" QtCore.SIGNAL
        self._add_btn.clicked.connect(self.add)

        # add the tree widget to the layout
        layout.addWidget(self._tree)
        # add the button to the layout
        layout.addWidget(self._add_btn)

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

    ############################################################################
    #                        Runtime behavior                                  #
    ############################################################################
    def enable_search_btn(self, is_enabled):
        """
        Function to enable/disable the search button

        Parameters
        ----------
        is_enabled : bool
            enables/disables the search button

        """
        self._search_btn.setEnabled(is_enabled)

    def enable_add_btn(self, is_enabled):
        """
        Function to enable/disable the search button

        Parameters
        ----------
        is_enabled : bool
            enables/disables the search button

        """
        self._add_btn.setEnabled(is_enabled)

    @QtCore.Slot()
    def add(self):
        """
        Figure out which result is clicked and emit the add_btn_sig with that
        dictionary
        """
        # TODO Change this to debugger level logging
        print("add_clicked")
        result_dict = {}
        # todo ask the tree nicely for its currently selected dictionary
        # unique_id = tree.get_current()
        self.add_btn_sig.emit(self._search_dict, result_dict, {}, [])

    @QtCore.Slot()
    def search(self):
        """
        Parse the search boxes and emit it as a signal
        """
        self.parse_search_boxes()
        # once the dictionary is constructed, emit it as a signal
        self.search_btn_sig.emit(self._search_dict)

    @QtCore.Slot()
    def parse_search_boxes(self):
        """
        Parse the search boxes to set up the query dictionary and store it as an
        instance variable "_search_dict"
        """
        # declare the search dict
        # TODO Change this to debugger level logging @tacaswell
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
        if len(self._search_dict) == 0:
            self._search_dict = _defaults["empty_search"]

    @QtCore.Slot(list)
    def update_search_results(self, results):
        """
        Pass the search results to the recursive tree widget which displays them

        Parameters
        ----------
        results : array, list, object

        """
        # stash the search results for later use
        self._search_results = results
        self._tree.fill_widget(results)
        self.enable_add_btn(is_enabled=True)

    def update_query_keys(self, keys, key_descriptions=None):
        """
        Function re-makes the query search box based on the new keys

        Parameters
        ----------
        keys : list
            List of keys that can be used as search terms

        Optional
        --------
        key_descriptions : list

        """
        # todo reimplement key_descriptions as a nested dict keyed on "keys"
        # with a "description" field and a "type" field so that the query input
        # boxes can be restricted to things like strings, numbers only, etc...
        if keys is None:
            return
        if key_descriptions is None:
            self._key_descriptions = keys

        # validate that the lists are the same length
        if len(keys) != len(key_descriptions):
            raise ValueError("keys and key_descriptions must be the same length"
                             "len(keys) = {0} and len(key_descriptions) = {1}".
                             format(len(keys), len(key_descriptions)))
        # stash the keys
        self._keys = keys
        layout = self._query_input.layout()
        # empty the current query widget
        # remove the old search query
        layout.takeAt(0)

        # remove the stretch
        layout.takeAt(0).deleteLater()

        # remove the button or buttons
        btns = layout.takeAt(0)

        new_query = self.construct_query_input(
            keys=keys, key_descriptions=key_descriptions)

        layout.addWidget(new_query)
        layout.addStretch()
        layout.addWidget(btns)


# todo enable add button only when something is selected
# todo status bar to display feedback
# todo sequence diagrams for runtime behavior
