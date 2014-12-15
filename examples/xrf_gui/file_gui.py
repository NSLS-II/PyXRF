__author__ = 'edill'

import enaml
from enaml.qt.qt_application import QtApplication
from bubblegum.xrf.model.fileio_model import FileIOModel
from bubblegum.xrf.model.lineplot_model import LinePlotModel
from bubblegum.xrf.model.guessparam_model import GuessParamModel


def run():
    app = QtApplication()
    with enaml.imports():
        from bubblegum.xrf.view.main_window import XRFGui

    xrfview = XRFGui()
    xrfview.file_M = FileIOModel()
    xrfview.p_guess_M = GuessParamModel()
    xrfview.plot_M = LinePlotModel()

    xrfview.show()
    app.start()


if __name__ == "__main__":
    run()