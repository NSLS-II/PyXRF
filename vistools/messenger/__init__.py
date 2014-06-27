# imports for future compatibility
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from six.moves import zip

# imports to smooth over differences between PyQt4, PyQt5, PyQt4.1 and PySides
from matplotlib.backends.qt4_compat import QtGui, QtCore

# other relevant imports
import numpy as np
from collections import OrderedDict

# local package imports
from ..backend import AbstractDataView, AbstractDataView2D, AbstractDataView1D


class AbstractMessenger(QtCore.QObject):
    """
    The AbstractMessenger is the abstract base class for the thin layer
    between the Qt side of the figure and the matplotlib GUI-independent
    layer.  The AbstractMessenger contains the slots that are common across
    all widgets in this library
    """

    def __init__(self, data_dict, *args, **kwargs):
        super(AbstractMessenger, self).__init__(*args, **kwargs)
        self._view = AbstractDataView(data_dict=data_dict)

    @QtCore.Slot()
    def sl_clear_data(self):
        """
        Remove all data
        """
        self._view._data = OrderedDict()
        self._view.replot()
        self.draw()

    @QtCore.Slot(str)
    def sl_remove_datasets(self, lbl_list):
        """
        Removes datasets specified by lbl_list

        Parameters
        ----------
        lbl : list
            str
            name(s) of dataset(s) to remove
        """
        self._view.remove_data(lbl_list=lbl_list)
        self._view.replot()
        self.draw()


class AbstractMessenger1D(AbstractMessenger):
    """
    AbstractMessenger1D class docstring
    """

    def __init__(self, data_dict, *args, **kwargs):
        super(AbstractMessenger1D, self).__init__(data_dict=data_dict, *args,
                                                  **kwargs)
        self._view = AbstractDataView(data_dict=data_dict)

    @QtCore.Slot(list, list, list)
    def sl_add_data(self, lbl_list, x_list, y_list):
        """
        Add a new dataset named 'lbl'

        Parameters
        ----------
        lbl_list : list
            names of the (x,y) coordinate lists.
        x_list : list
            1 or more columns of x-coordinates.  Must be the same shape as y.
        y_list : list
            1 or more columns of y-coordinates.  Must be the same shape as x.
        """
        self._view.add_data(lbl_list=lbl_list, x_list=x_list, y_list=y_list)
        self._view.replot()
        self.draw()

    @QtCore.Slot(list, list, list)
    def sl_append_data(self, lbl_list, x_list, y_list):
        """
        Append data to the dataset specified by 'lbl'

        Parameters
        ----------
        lbl_list : list
            names of the (x,y) coordinate lists.
        x_list : list
            1 or more columns of x-coordinates.  Must be the same shape as y.
        y_list : list
            1 or more columns of y-coordinates.  Must be the same shape as x.
        """
        self._view.append_data(lbl_list=lbl_list, x_list=x_list, y_list=y_list)
        self._view.replot()
        self.draw()


class AbstractMessenger2D(AbstractMessenger):
    """
    AbstractMessenger2D class docstring
    """

    def __init__(self, data_dict, *args, **kwargs):
        super(AbstractMessenger1D, self).__init__(data_dict=data_dict, *args,
                                                  **kwargs)
        self._view = AbstractDataView(data_dict=data_dict)

    @QtCore.Slot(list, list, list)
    def sl_add_data(self, lbl_list, xy_list, corners_list):
        """
        Add new datasets

        Parameters
        ----------
        lbl_list : list
            names of the (x,y) coordinate lists.
        xy_list : list
            list of 2D arrays of image data
        corners_list : list
            list of corners that provide information about the relative
            position of the axes (corners_list is a tuple of 4: x0, y0, x1,
            y1, where x0,y0 is the lower-left corner and x1,y1 is the
            upper-right corner
        """
        self._view.add_data(lbl_list=lbl_list, xy_list=xy_list,
                            corners_list=corners_list)
        self._view.replot()
        self.draw()

    @QtCore.Slot(list, list, list, list)
    def sl_append_data(self, lbl_list, xy_list, axis_list, append_to_end_list):
        """
        Append data to the dataset specified by 'lbl'

        Parameters
        ----------
        lbl_list : list
            names of the (x,y) coordinate lists.
        xy_list : list
            list of 2D arrays of image data
        axis : list
            int
            axis == 0 is appending in the horizontal direction
            axis == 1 is appending in the vertical direction
        append_to_end : list
            bool
            if false, prepend to the dataset
        """
        self._view.append_data(lbl_list=lbl_list, xy_list=xy_list,
                               axis_list=axis_list,
                               append_to_end_list=append_to_end_list)
        self._view.replot()
        self.draw()

    @QtCore.Slot(list, list, list, list)
    def sl_append_data(self, lbl_list, x_list, y_list, val_list):
        """
        Add a single data point to an array

        Parameters
        ----------
        lbl : list
            str
            name of the dataset to add one datum to
        x : list
            int
            index of x coordinate
        y : list
            int
            index of y coordinate
        val : list
            float
            value of datum at the coordinates specified by (x,y)
        """
        self._view.append_data(lbl_list=lbl_list, x_list=x_list,
                               y_list=y_list,
                               val_list=val_list)
        self._view.replot()
        self.draw()


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

