from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import six
from collections import defaultdict, OrderedDict

from matplotlib.backends.qt4_compat import QtGui, QtCore
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas  # noqa
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar  # noqa
from matplotlib.figure import Figure
from matplotlib.cm import datad
from matplotlib import colors
import numpy as np

_CMAPS = datad.keys()
_CMAPS.sort()


class PlotWidget(QtGui.QMainWindow):
    """
    Top level container for one control widget and one data view widget
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
        self._config = QtGui.QDockWidget(control_box_name)
        self._ctl_widget = ControlContainer(self.ctrl_box_2)

    @property
    def fig(self):
        """
        The `Figure` object for this widget.
        """
        return self._fig

    @property
    def control(self):
        return self._ctl_widget


class AbstractDataView(object):
    """
    AbstractDataView class docstring.  Defaults to a single matplotlib axes
    """
    _default_cmap = _CMAPS[0]
    _default_norm = colors.Normalize(0, 1, clip=True)

    def __init__(self, fig=None, cmap=None, norm=None):
        """
        init docstring

        Parameters
        ----------
        data_dict : Dictionary
            Dictionary of data sets
        fig : mpl.Figure

        ax1 : mpl.Axes

        cmap :
        norm :
        """
        if self._fig is None:
            # no Figure has been stashed yet
            if fig is None:
                # no stashed Figure and no Figure passed in. That's a problem
                raise Exception("Figure is needed before artists can be rendered")
            else:
                # stash the Figure that was passed in
                self._fig = fig

        if cmap is None:
            # no color map was passed in. Init to default value
            self._cmap = AbstractDataView._default_cmap
        else:
            # stash the color map
            self._cmap = cmap

        if norm is None:
            # no normalization policy was passed in. Init to default value
            self._norm = AbstractDataView._default_norm
        else:
            # stash the normalization policy that was passed in
            self._norm = norm

    @QtCore.Slot()
    def update_colormap(self, new_cmap):
        """
        Update the color map used to display the image
        """
        self._cmap = new_cmap

    def replot(self):
        """
        Do nothing in the abstract base class. Needs to be implemented
        in the concrete classes
        """
        raise Exception("Must override the replot() method in the concrete base class")

    @QtCore.Slot(name="colors.Normalize")
    def sl_update_norm(self, new_norm):
        """
        Updates the normalization function used for the color mapping
        """
        self._norm = new_norm


class AbstractDataView1D(AbstractDataView):
    """
    AbstractDataView1D class docstring.  Defaults to a single matplotlib axes
    """
    def __init__(self, fig=None, data_dict=None, cmap=None, norm=None):
        """
        __init__ docstring__

        Parameters
        ----------
        fig :
        data_dict :
        cmap :
        norm :
        """
        # pass the init up the toolchain
        AbstractDataView.__init__(self, fig=fig, cmap=cmap, norm=norm)
        # stash the parameters not taken care of by parent class
        if data_dict is not None:
            self._data_dict = data_dict

    def add_data(self, x, y, lbl):
        """
        add data with the name 'lbl'.  Will overwrite data if
        'lbl' already exists in the data dictionary

        Parameters
        ----------
        lbl : String
            Name of the data set
        x : np.ndarray
            single vector of x-coordinates
        y : np.ndarray
            single vector of y-coordinates
        """
        self._data[lbl] = (x, y)

    def remove_data(self, lbl):
        """
        Remove the key:value pair from the dictionary

        Parameters
        ----------
        lbl : String
            name of dataset to remove
        """
        try:
            # delete the key:value pair from the dictionary
            del self._data[lbl]
        except NameError:
            # do nothing because the data at 'lbl' doesn't exist
            pass

    def append_data(self, lbl, x, y):
        """
        Append (x, y) coordinates to a dataset.  If there is no dataset
        called 'lbl', add the (x_data, y_data) tuple to a new entry
        specified by 'lbl'

        Parameters
        ----------
        lbl : String
            name of data set to append
        x : np.ndarray
            single vector of x-coordinates to add.
            x_data must be the same length as y_data
        y : np.ndarray
            single vector of y-coordinates to add.
            y_data must be the same length as x_data
        """
        try:
            # get the current vectors at 'lbl'
            (prev_x, prev_y) = self._data[lbl]
            # set the concatenated data to 'lbl'
            self._data[lbl] = (np.concatenate((prev_x, x)),
                               np.concatenate((prev_y, y)))
        except NameError:
            # key doesn't exist, add data to a new entry called 'lbl'
            self._data[lbl] = (x, y)


class AbstractDataView2D(AbstractDataView):
    """
    AbstractDataView2D class docstring
    """


class AbstractCanvas(FigureCanvas):
    """
    The AbstractCanvas is the abstract base class for the thin layer
    between the Qt side of the figure and the matplotlib GUI-independent
    layer.  The AbstractCanvas contains the slots that are common across
    all widgets in this library
    """
    default_height = 24
    default_width = 24

    def __init__(self, fig, view=None):
        """
        docstring

        Parameters
        ----------
        parent :
        fig :
        view :
        """
        if fig is None:
            raise Exception("fig cannot be None. A FigureCanvas needs a Figure")

        # call parent class constructor
        FigureCanvas.__init__(self, fig)

        # set some default behavior
        FigureCanvas.setSizePolicy(self,
                                   QtGui.QSizePolicy.Expanding,
                                   QtGui.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

        if view is None:
            # give it a default data view
            self._view = AbstractDataView(self.figure)
        else:
            # stash the view that was passed in
            self._view = view

    @QtCore.Slot(str)
    def sl_update_color_map(self, cmap):
        """
        Updates the color map.  Currently takes a string, should probably be
        redone to take a cmap object and push the look up function up a layer
        so that we do not need the try..except block.
        """
        try:
            self._view.update_colormap(str(cmap))
        except ValueError:
            pass
        self._view.replot()
        self.draw()

    @QtCore.Slot()
    def sl_clear_data(self):
        """
        Remove all data
        """
        self._view._data = OrderedDict()
        self._view._ax1.clear()
        self._view.replot()
        self.draw()

    @QtCore.Slot(int)
    def sl_remove_dataset(self, lbl):
        """
        Remove dataset specified by idx

        Parameters
        ----------
        lbl : String
            name of dataset to remove
        """
        self._view.remove_data(lbl)
        self._view.replot()
        self.draw()


class AbstractCanvas1D(AbstractCanvas):
    """
    AbstractCanvas1D class docstring
    """
    def __init__(self, fig=None, view=None):
        # call up the stack to initialize
        AbstractCanvas.__init__(self, fig=fig, view=view)

    @QtCore.Slot(list, list, list)
    def sl_add_data(self, lbls, x_data, y_data):
        """
        Add a new dataset named 'lbl'

        Parameters
        ----------
        lbl : list
            names of the (x,y) coordinate lists.
        x : list
            1 or more columns of x-coordinates.  Must be the same shape as y.
        y : list
            1 or more columns of y-coordinates.  Must be the same shape as x.
        """
        for (lbl, x, y) in zip(lbls, x_data, y_data):
            self._view.add_data(x=x, y=y, lbl=lbl)
        self._view.replot()
        self.draw()

    @QtCore.Slot(list, list, list)
    def sl_append_data(self, lbls, x_data, y_data):
        """
        Append data to the dataset specified by 'lbl'

        Parameters
        ----------
        lbl : list
            names of the (x,y) coordinate lists.
        x : list
            1 or more columns of x-coordinates.  Must be the same shape as y.
        y : list
            1 or more columns of y-coordinates.  Must be the same shape as x.
        """
        for (lbl, x, y) in zip(lbls, x_data, y_data):
            self._view.append_data(x=x, y=y, lbl=lbl)
        self._view.replot()
        self.draw()


class AbstractCanvas2D(AbstractCanvas):
    """
    AbstractCanvas2D class docstring
    """


class AbstractMPLWidget(QtGui.QWidget):
    """
    AbstractDatatWidget class docstring
    """
    def __init__(self, canvas=None, data=None, page_size=10, parent=None):
        # init the QWidget
        QtGui.QWidget.__init__(self, parent)
        if self._canvas is None:
            # no canvas has been stored yet
            if canvas is not None:
                # a canvas has been passed to the init method. store it and init
                self._canvas = canvas
                self.init_canvas()
            else:
                # no canvas exists, create the default abstract canvas
                self._canvas = AbstractCanvas(self, data, parent=self)
                self.init_canvas()
        else:
            # canvas has already been stored, do nothing
            pass

        # do nothing else

    def init_canvas(self):
        """
        Initialize the mpl canvas by adding a Toolbar and adding the
        FigureCanvas to this QWidget
        """
        # create the mpl toolbar
        self._mpl_toolbar = NavigationToolbar(self._canvas, self)
        # create a layout manager
        layout = QtGui.QVBoxLayout()
        # add the mpl toolbar to the layout
        layout.addWidget(self._mpl_toolbar)
        # add the mpl canvas to the layout
        layout.addWidget(self._canvas)
        # add the layout to the widget
        self.setLayout(layout)


class LimitSpinners(QtGui.QGroupBox):
    # signal to be emitted when the spin boxes are changed
    # and settled
    valueChanged = QtCore.Signal(float, float)

    def __init__(self, title='', parent=None):
        QtGui.QGroupBox.__init__(self, title, parent=parent)

        self._spinbox_min_intensity = QtGui.QDoubleSpinBox(parent=self)
        self._spinbox_max_intensity = QtGui.QDoubleSpinBox(parent=self)
        self._spinbox_intensity_step = QtGui.QDoubleSpinBox(parent=self)

        ispiner_form = QtGui.QFormLayout()
        ispiner_form.addRow("min", self._spinbox_min_intensity)
        ispiner_form.addRow("max", self._spinbox_max_intensity)
        ispiner_form.addRow("step", self._spinbox_intensity_step)
        self.setLayout(ispiner_form)

    # TODO

    @QtCore.Slot(float, float)
    def setValue(self, bottom, top):
        """
        """
        pass

    @QtCore.Slot(float)
    def setStep(self, step):
        """
        """
        pass


class _mapping_mixin(object):
    """
    This is lifted from _abccoll.py but strips out the
    meta-class stuff.  This is because the qt-classes have
    a non-trivial meta class so the standard classes from
    collections can not be used

    client classes need to define:
    __len__
    __contains__
    __getitem__
    __iter__

    or this will fail in unpredictable ways
    """
    def get(self, key, default=None):
        'D.get(k[,d]) -> D[k] if k in D, else d.  d defaults to None.'
        try:
            return self[key]
        except KeyError:
            return default

    def __contains__(self, key):
        try:
            self[key]
        except KeyError:
            return False
        else:
            return True

    def iterkeys(self):
        'D.iterkeys() -> an iterator over the keys of D'
        return iter(self)

    def itervalues(self):
        'D.itervalues() -> an iterator over the values of D'
        for key in self:
            yield self[key]

    def iteritems(self):
        'D.iteritems() -> an iterator over the (key, value) items of D'
        for key in self:
            yield (key, self[key])

    def keys(self):
        "D.keys() -> list of D's keys"
        return list(self)

    def items(self):
        "D.items() -> list of D's (key, value) pairs, as 2-tuples"
        return [(key, self[key]) for key in self]

    def values(self):
        "D.values() -> list of D's values"
        return [self[key] for key in self]

    # Mappings are not hashable by default, but subclasses can change this
    __hash__ = None

    def __eq__(self, other):
        return dict(self.items()) == dict(other.items())

    def __ne__(self, other):
        return not (self == other)


class ControlContainer(QtGui.QGroupBox, _mapping_mixin):
    _delim = '.'

    def __len__(self):
        print('len')
        return len(list(iter(self)))

    def __contains__(self, key):
        print('contains')
        return key in iter(self)

    def __init__(self, parent=None):
        # call parent constructor
        QtGui.QWidget.__init__(self, parent=parent)

        # nested containers
        self._containers = dict()
        # all non-container contents of this container
        self._contents = dict()

        # specialized listings
        # this is a dict keyed on type of dicts
        # The inner dicts are keyed on name
        self._by_type = defaultdict(dict)

        # make the layout
        self._layout = QtGui.QVBoxLayout()
        # add it to self
        self.setLayout(self._layout)

    def __getitem__(self, key):
        print(key)
        # TODO make this sensible un-wrap KeyException errors
        try:
            # split the key
            split_key = key.strip(self._delim).split(self._delim, 1)
        except TypeError:
            raise KeyError("key is not a string")
        # if one element back -> no splitting needed
        if len(split_key) == 1:
            return self._contents[split_key[0]]
        # else, at least one layer of testing
        else:
            # unpack the key parts
            outer, inner = split_key
            # get the container and pass through the remaining key
            return self._containers[outer][inner]

    def create_container(self, key, container_name):
        pass

    def create_combobox(self, key, key_list, editable=False):
        pass

    def create_slider(self, key, min_val, max_val, stepsize=None,
                    inc_spinbox=True):
        """

        Parameters
        ----------
        """
        pass

    def create_label(self, key, text):
        """
        Create and add a text label to the control panel
        """
        # create label
        tmp_label = QtGui.QLabel(text)
        self._add_widget(key, tmp_label)

    def _add_widget(self, key, in_widget):
        split_key = key.strip(self._delim).rsplit(self._delim, 1)
        # key is not nested, add to this object
        if len(split_key) == 1:
            key = split_key[0]
            # add to the type dict
            self._by_type[type(in_widget)][key] = in_widget
            # add to the contents list
            self._contents[key] = in_widget
            # add to layout
            self._layout.addWidget(in_widget)
        # else, grab the nested container and add it to that
        else:
            container, key = split_key
            self[container]._add_widget(key, in_widget)

    def add_dict_display(self, key, input_dict):
        pass

    def iter_containers(self):
        return six.iterkeys(self._containers)

    def _iter_helper(self, cur_path_list):
        """
        Recursively (depth-first) walk the tree and return the namesq
        of the leaves

        Parameters
        ----------
        cur_path_list : list of str
            A list of the current path
        """
        for k, v in six.iteritems(self._containers):
            for inner_v in v._iter_helper(cur_path_list + [k]):
                yield inner_v
        for k in six.iterkeys(self._contents):
            yield self._delim.join(cur_path_list + [k])

    def __iter__(self):
        return self._iter_helper([])


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

