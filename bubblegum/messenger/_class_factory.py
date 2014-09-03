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
from .. import QtCore, QtGui

import logging
logger = logging.getLogger(__name__)

class mock(object):
    def __getattr__(self, key):
        def fun(*args):
            print('{}: {}'.format(key, args))
        fun.__name__ = key
        return fun


class tester(QtGui.QWidget):
    def __init__(self, parent=None):
        super(tester, self).__init__(parent=parent)
        self._view = mock()


def make_messenger(backend_class,
                   slots_dict, controls_dict):
    """
    This is the top level function for generating messenger
    classes.  This is function takes a backend class (the
    thing that talk to what ever library is actually doing
    the plotting), a list of slots which maps the functions
    on the backend-object to slot names + signatures, and a
    dictionary specifies how to set up and connect control
    widgets to the base-class.

    Parameters
    ----------
    backend_class : type
        The class object to wrap

    slots_dict : dict
       dictionary keyed on slot name.  The values are
       a tuple of ('function_name', 'signature') which
       are the name of the member method to call for the slot as
       a string and the signature as  tuple of types. ex

          {'set_range': ('set_range', (float, float)),
           'set_label': ('set_label', (str,))}

    controls_dict : dict
       A nested dictionary to specify the control layout.
       This still needs to be sorted out.  This will turn
       into somewhat ugly json, but will be nicer than writing
       the qt boiler plate our selves

         {'control_name':
          {'type': 'box',
            'parameters':
             {'contents':{ 'frame': {'type': 'slider',
                                     'parameters': {'min': 0,
                                                    'max': 32},
                                        'connections': (
                                         (signal_name, slot_name),
                                         (signal_name, slot_name))}
                                         }
                                         }
                                         }
                                         }
    """
    # this will be used by the upstream init to
    member_dict = {'_backend_class': backend_class}
    # sort out what the class name should be
    backend_name = backend_class.__name__
    _suffix = 'View'
    if backend_name.endswith(_suffix):
        backend_name = backend_name[:-len(_suffix)]
    messenger_name = backend_name + 'Messenger'

    for key, (func_name, signature) in six.iteritems(slots_dict):
        def tmp_fun(self, *args):
            getattr(self._view, func_name)(*args)

        tmp_fun.__name__ = key
        member_dict[key] = QtCore.Slot(*signature)(tmp_fun)

    tmp_class = type(str(messenger_name),
                     (tester,),
                     member_dict)

    return tmp_class


def recursive_control_maker(parent_box, controls_dict, obj):
    for key, vals in six.iteritems(controls_dict):
        print(key)
        print(vals)
        if vals['type'] == 'box':
            # recurse
            new_box = parent_box.create_container(key)
            recursive_control_maker(new_box, vals['contents'], obj)
        else:
            # base case
            ctl_widget = parent_box.create_widget(key,
                                                  vals['type'],
                                                  vals['parameters'])
            print('here')
            # hook up the signals/slots
            for signal, slot in vals['connections']:
                print(signal)
                sig = getattr(ctl_widget, signal)
                print(sig)
                slt = getattr(obj, slot)
                print(slt)
                sig.connect(slt)
