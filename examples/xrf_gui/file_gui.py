__author__ = 'edill'

import enaml
from enaml.qt.qt_application import QtApplication
from bubblegum.xrf.model.xrf_model import FileIOModel, LinePlotModel

def run():
    app = QtApplication()
    with enaml.imports():
        from bubblegum.xrf.view.main_window import XRFGui

    view = XRFGui()
    view.model1 = FileIOModel()
    view.model2 = LinePlotModel()

    view.show()
    app.start()


if __name__ == "__main__":
    run()