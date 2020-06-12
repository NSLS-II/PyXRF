from pyxrf.gui_module.main_window import MainWindow
from PyQt5.QtWidgets import QMessageBox
from pyxrf.gui_support.gpc_class import GlobalProcessingClasses


def test_MainWindow(qtbot, monkeypatch):
    """
    Simple test that opens and closes the main window and
    checks the window dimensions
    """
    gpc = GlobalProcessingClasses()
    gpc.initialize()

    window = MainWindow(gpc=gpc)
    window.show()

    qtbot.addWidget(window)
    qtbot.waitForWindowShown(window)

    monkeypatch.setattr(QMessageBox, "exec", lambda *args: QMessageBox.No)
    window.close()
    assert not window._is_closed, "Window was closed prematurely"

    monkeypatch.setattr(QMessageBox, "exec", lambda *args: QMessageBox.Yes)
    window.close()
    assert window._is_closed, "Window was not closed properly"
