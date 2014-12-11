__author__ = 'edill'

import enaml
from enaml.qt.qt_application import QtApplication
from bubblegum.xrf.model.xrf_model import XRF

def run():
    app = QtApplication()
    with enaml.imports():
        from bubblegum.xrf.view.xrf_view import XrfGui

    view = XrfGui()
    view.xrf_model = XRF()

    view.show()
    app.start()


if __name__ == "__main__":
    run()