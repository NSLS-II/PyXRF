
from pyxrf.gui_module.main_window import MainWindow, _main_window_geometry


def test_MainWindow(qtbot):
    """
    Simple test that opens and closes the main window and
    checks the window dimensions
    """

    window = MainWindow()
    window.show()

    qtbot.addWidget(window)
    qtbot.waitForWindowShown(window)

    w_size = window.size()
    assert w_size.width() == _main_window_geometry["initial_width"], \
        "Main window width is set incorrectly"
    assert w_size.height() == _main_window_geometry["initial_height"], \
        "Main window height is set incorrectly"

    window.close()
    assert window._is_closed, "Window was not closed properly"
