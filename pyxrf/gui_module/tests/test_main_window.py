from pyxrf.gui_module.main_window import MainWindow


def test_MainWindow(qtbot):
    """
    Simple test that opens and closes the main window and
    checks the window dimensions
    """

    window = MainWindow()
    window.show()

    qtbot.addWidget(window)
    qtbot.waitForWindowShown(window)

    window.close()
    assert window._is_closed, "Window was not closed properly"
