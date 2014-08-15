# imports to smooth over differences between PyQt4, PyQt5, PyQt4.1 and PySides
import matplotlib
matplotlib.rcParams["backend"] = "Qt4Agg"
matplotlib.rcParams["backend.qt4"] = "PySide"
from matplotlib.backends.qt4_compat import QtCore, QtGui
