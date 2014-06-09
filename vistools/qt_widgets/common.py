from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import six


from matplotlib.backends.qt4_compat import QtGui, QtCore
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas  # noqa
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar  # noqa
from matplotlib.figure import Figure


class PlotWidget(QtGui.QMainWindow):
    """
    Class doc-string
    """
    def __init__(self, parent=None, embed=True):
        """
        init doc-string

        Parameters
        ----------
        parent : QWidget
            parent widget

        embed : bool
            if widget should be embeddable in another widget (if true)
            or a stand-alone window (false)
        """
        # explictily call up the stack
        # TODO sort out qt5 super rules
        QtGui.QMainWindow.__init__(self, parent)
        # set flag is this should be widget or a main window
        if embed:
            self.setWindowFlags(QtCore.Qt.Widget)

        # dummy figure size in inches
        width = height = 12
        # create the Figure object
        self._fig = Figure(figsize=(width, height))
        # create the canvas (and connect it to the figure)
        self._canvas = FigureCanvas(self.fig, parent=self)
        # set the canvas to fill available space
        self._canvas.setSizePolicy(QtGui.QSizePolicy.Expanding,
                                   QtGui.QSizePolicy.Expanding)
        self._canvas.updateGeometry()

        # set the central widget to be the canvas
        self.setCentralWidget(self._canvas)

        # create the control box
        control_box_name = 'controls'
        self._config  = QtGui.QDockWidget(control_box_name)
        self._ctl_widget = ControlContainer(self.ctrl_box_2)

    @property
    def fig(self):
        """
        The `Figure` object for this widget.
        """
        return self._fig


class ControlContainer(QtGui.QWidget):
    _delim = '.'

    def __init__(self, parent=None):
        # call parent constructor
        QtGui.QWidget.__init__(self, parent=parent)

        # nested containers
        self._containers = dict()
        # all non-container contents of this container
        self._contents = dict()

        # specialized listings
        self._comboboxs = dict()
        self._sliders = dict()
        self._labels = dict()
        self._dict_disp = dict()

    def __getitem__(self, key):
        # TODO make this sensible un-wrap KeyException errors

        # split the key
        split_key = key.strip(self._delim).split(self._delim, 1)
        # if one element back -> no splitting needed
        if len(split_key) == 1:
            return self._contents[split_key[0]]
        # else, at least one layer of testing
        else:
            # unpack the key parts
            outer, inner = split_key
            # get the container and pass through the remaining key
            return self._containers[outer][inner]

    def add_container(self, key, container):
        pass

    def add_combobox(self, key, key_list, editable=False):
        pass

    def add_slider(self, key, min_val, max_val, inc_spinbox=True):
        pass

    def add_label(self, key, text):
        pass

    def add_dict_display(self, key, input_dict):
        pass

    def iter_containers(self):
        return six.iterkeys(self._containers)


class DictDisplay(QtGui.QGroupBox):
    """
    A generic widget for displaying dictionaries

    Parameters
    ----------
    title : string
       Widget title

    ignore_list : iterable or None
        keys to ignore

    parent : QWidget or None
        Parent widget, passed up stack
    """
    def __init__(self, title, ignore_list=None, parent=None):
        # pass up the stack, GroupBox takes care of the title
        QtGui.QGroupBox.__init__(self, title, parent=parent)

        if ignore_list is None:
            ignore_list = ()
        # make layout
        self.full_layout = QtGui.QVBoxLayout()
        # set layout
        self.setLayout(self.full_layout)
        # make a set of the ignore list
        self._ignore = set(ignore_list)
        self._disp_table = []

    @QtCore.Slot(dict)
    def update(self, in_dict):
        """
        updates the table

        Parameters
        ----------
        in_dict : dict
            The dictionary to display
        """
        # remove everything that is there
        for c in self._disp_table:
            c.deleteLater()

        # make a new list
        self._disp_table = []

        # add the keys alphabetically
        for k, v in sorted(list(in_dict.iteritems())):
            # if key in the ignore list, continue
            if k in self._ignore:
                continue
            self._add_row(k, v)

    def _add_row(self, k, v):
        """
        Private function

        Adds a row to the table

        Parameters
        ----------
        k : object
           The key

        v : object
           The value
        """
        # make a widget for our row
        tmp_widget = QtGui.QWidget(self)
        tmp_layout = QtGui.QHBoxLayout()
        tmp_widget.setLayout(tmp_layout)

        # add the key and value to the row widget
        tmp_layout.addWidget(QtGui.QLabel(str(k) + ':'))
        tmp_layout.addStretch()
        tmp_layout.addWidget(QtGui.QLabel(str(v)))

        # add the row widget to the full layout
        self.full_layout.addWidget(tmp_widget)
        # add the row widget to
        self._disp_table.append(tmp_widget)
