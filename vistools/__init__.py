# imports to smooth over differences between PyQt4, PyQt5, PyQt4.1 and PySides
import matplotlib
matplotlib.rcParams["backend"] = "Qt4Agg"

# use the PySide rcParams if that's your preference
usePyQt4 = True
if usePyQt4:
    matplotlib.rcParams["backend.qt4"] = "PyQt4"
else:
    matplotlib.rcParams["backend.qt4"] = "PySide"

from matplotlib.backends.qt4_compat import QtCore, QtGui

import logging
logger = logging.getLogger(__name__)