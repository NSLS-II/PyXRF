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
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import six
from collections import defaultdict
from .util import mapping_mixin
from .. import QtCore, QtGui


def _update_header(header):
    """Internal helper function to update the GUI with a new header

    This function should tell the GUI to do something with a new header
    """
    pass


def _update_ev_desc(ev_desc):
    """Internal helper function to update the GUI with a new event descriptor

    Additionally, this function is responsible for setting up any listening

    """
    pass


def _update_event(event):
    """Internal helper function to update the GUI with a new event(s)

    Parameters
    ----------
    event : pandas.DataFrame
        Time-aligned data with one or more named columns
    """
    pass


def update(msg, msg_obj=None):
    """ Message passing function

    Update takes a message and an optional object.  The message is used to
    determine behavior of the qt widget. If the object is present, the qt widget
    will respond if it understands the message

    Parameters
    ----------
    msg : {'header', 'event_descriptor', 'event'}
        Message that is used to instruct the widget to perform some action
    obj : object, optional
        Object that is used
    """
    pass


_update_dict = {'header': _update_header,
                'event_descriptor': _update_ev_desc,
                'event': _update_event,
}