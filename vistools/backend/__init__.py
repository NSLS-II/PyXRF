from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from six.moves import zip

import numpy as np
from collections import OrderedDict

__author__ = 'Eric-hafxb'


class AbstractDataView(object):
    """
    AbstractDataView class docstring.  Defaults to a single matplotlib axes
    """

    default_data_structure = OrderedDict

    # no init because this class should not be used directly
    def __init__(self, data_dict=None, *args, **kwargs):
        """
        Parameters
        ----------
        data_dict : OrderedDictionary
        """
        super(AbstractDataView, self).__init__(*args, **kwargs)
        self._data_dict = data_dict

    def replot(self):
        """
        Do nothing in the abstract base class. Needs to be implemented
        in the concrete classes
        """
        raise Exception("Must override the replot() method in the concrete base class")

    def clear_data(self):
        data_dict = self.default_data_structure()

    def remove_data(self, lbl_list):
        """
        Remove the key:value pair from the dictionary as specified by the
        labels in lbl_list

        Parameters
        ----------
        lbl_list : String
            name of dataset to remove
        """
        for lbl in lbl_list:
            try:
                del self._data_dict[lbl]
            except KeyError:
                # do nothing
                pass


class AbstractDataView1D(AbstractDataView):
    """
    AbstractDataView1D class docstring.  Defaults to a single matplotlib axes
    """

    # no init because it contains no new attributes

    def add_data(self, lbl_list, x_list, y_list):
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
        for (lbl, x, y) in zip(lbl_list, x_list, y_list):
            self._data_dict[lbl] = (x, y)

    def append_data(self, lbl_list, x_list, y_list):
        """
        Append (x, y) coordinates to a dataset.  If there is no dataset
        called 'lbl', add the (x_data, y_data) tuple to a new entry
        specified by 'lbl'

        Parameters
        ----------
        lbl : list
            str
            name of data set to append
        x : list
            np.ndarray
            single vector of x-coordinates to add.
            x_data must be the same length as y_data
        y : list
            np.ndarray
            single vector of y-coordinates to add.
            y_data must be the same length as x_data
        """
        for (lbl, x, y) in zip(lbl_list, x_list, y_list):
            try:
                # get the current vectors at 'lbl'
                (prev_x, prev_y) = self._data_dict[lbl]
                # set the concatenated data to 'lbl'
                self._data_dict[lbl] = (np.concatenate((prev_x, x)),
                                   np.concatenate((prev_y, y)))
            except KeyError:
                # key doesn't exist, add data to a new entry called 'lbl'
                self._data_dict[lbl] = (x, y)


class AbstractDataView2D(AbstractDataView):
    """
    AbstractDataView2D class docstring
    """
# no init because it contains no new attributes

    def add_data(self, lbl_list, xy_list, corners_list):
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
        for (lbl, x, y) in zip(lbl_list, xy_list):
            self._data_dict[lbl] = (x, y)

    def append_data(self, lbl_list, xy_list, axis=[], append_to_end=[]):
        """
        Append (x, y) coordinates to a dataset.  If there is no dataset
        called 'lbl', add the (x_data, y_data) tuple to a new entry
        specified by 'lbl'

        Parameters
        ----------
        lbl : list
            str
            name of data set to append
        xy : list
            np.ndarray
            List of 2D arrays
        axis : list
            int
            axis == 0 is appending in the horizontal direction
            axis == 1 is appending in the vertical direction
        append_to_end : list
            bool
            if false, prepend to the dataset
        """
        for (lbl, xy, ax, end) in zip(lbl_list, xy_list, axis, append_to_end):
            try:
                # set the concatenated data to 'lbl'
                if end:
                    self._data_dict[lbl] = np.r_[str(ax), self._data_dict[lbl], xy]
                    # TODO: Need to update the corners_list also...
                else:
                    self._data_dict[lbl] = np.r_[str(ax), xy, self._data_dict[lbl]]
                    # TODO: Need to update the corners_list also...
            except KeyError:
                # key doesn't exist, add data to a new entry called 'lbl'
                self._data_dict[lbl] = xy

    def add_datum(self, lbl_list, x_list, y_list, val_list):
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
        raise NotImplementedError("Not yet implemented")