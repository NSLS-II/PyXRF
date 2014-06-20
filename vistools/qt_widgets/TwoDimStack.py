'''
Created on Jun 19, 2014

@author: Eric
'''

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import six
from vistools.qt_widgets import common

class TwoDimStackViewer(common.AbstractDataView2D):
    """
    The TwoDimStackViewer provides a UI widget for viewing a number of 
    2-D datasets with x- and y- slices shown at the top and left of the 
    widget, respectively
    """
    def __init__(self):
        pass


class TwoDimStackCanvas(common.AbstractCanvas2D):
    """
    TwoDimStackCanvas class docstring
    """
    def __init__(self, init_img, parent=None):
        