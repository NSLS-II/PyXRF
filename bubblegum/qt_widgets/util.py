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

"""
Module for holding useful non Qt code
"""


class mapping_mixin(object):
    """
    This is lifted from _abccoll.py but strips out the
    meta-class stuff.  This is because the qt-classes have
    a non-trivial meta class so the standard ABCs from
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
