from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import six

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
