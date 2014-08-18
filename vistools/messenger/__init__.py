# ######################################################################
# Copyright (c) 2014, Brookhaven Science Associates, Brookhaven        #
# National Laboratory. All rights reserved.                            #
#                                                                      #
# Redistribution and use in source and binary forms, with or without   #
# modification, are permitted provided that the following conditions   #
# are met:                                                             #
#                                                                      #
# * Redistributions of source code must retain the above copyright     #
#   notice, this list of conditions and the following disclaimer.      #
#                                                                      #
# * Redistributions in binary form must reproduce the above copyright  #
#   notice this list of conditions and the following disclaimer in     #
#   the documentation and/or other materials provided with the         #
#   distribution.                                                      #
#                                                                      #
# * Neither the name of the Brookhaven Science Associates, Brookhaven  #
#   National Laboratory nor the names of its contributors may be used  #
#   to endorse or promote products derived from this software without  #
#   specific prior written permission.                                 #
#                                                                      #
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS  #
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT    #
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS    #
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE       #
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,           #
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES   #
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR   #
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)   #
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,  #
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OTHERWISE) ARISING   #
# IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE   #
# POSSIBILITY OF SUCH DAMAGE.                                          #
########################################################################
# imports for future compatibility
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from .. import QtCore, QtGui

# other relevant imports

# local package imports
from ..backend import AbstractDataView, AbstractDataView1D, AbstractDataView2D

import logging
logger = logging.getLogger(__name__)


class AbstractMessenger(QtCore.QObject):
    """
    The AbstractMessenger is the abstract base class for the thin layer
    between the Qt side of the figure and the matplotlib GUI-independent
    layer.  The AbstractMessenger contains the slots that are common across
    all widgets in this library
    """

    def __init__(self, *args, **kwargs):
        super(AbstractMessenger, self).__init__(*args, **kwargs)

    @QtCore.Slot()
    def sl_clear_data(self):
        """
        Remove all data
        """
        self._view.clear_data()
        self.sl_update_view()

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
        self.sl_update_view()

    def sl_update_view(self):
        raise NotImplementedError("Concrete classes must override this method")


class AbstractMessenger1D(AbstractMessenger):
    """
    AbstractMessenger1D class docstring
    """

    def __init__(self, *args, **kwargs):
        super(AbstractMessenger1D, self).__init__(*args, **kwargs)

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
        self.sl_update_view()

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
        self.sl_update_view()


class AbstractMessenger2D(AbstractMessenger):
    """
    AbstractMessenger2D class docstring
    """

    def __init__(self, *args, **kwargs):
        super(AbstractMessenger2D, self).__init__(*args, **kwargs)

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
        self.sl_update_view()

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
        self.sl_update_plot()
        
    @QtCore.Slot(list, list, list, list)
    def sl_add_datum(self, lbl_list, x_list, y_list, val_list):
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
        self.sl_update_plot()


class AbstractDisplayWidget(QtGui.QWidget):
    """
    AbstractDisplayWidget class docstring.
    The purpose of this class and its daughter classes is simply to render the
    figure that the various plotting libraries use to present themselves
    """
    def __init__(self, parent=None, *args, **kwargs):
        # init the QWidget
        super(AbstractDisplayWidget, self).__init__(parent=parent, *args,
                                                    **kwargs)
        # do nothing else