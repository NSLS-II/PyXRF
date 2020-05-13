from PyQt5.QtWidgets import QMainWindow


_main_window_geometry = {
    "initial_height": 800,
    "initial_width": 1000,
    "min_height": 400,
    "min_width": 500,
}


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        # Indicates that the window was closed (used mostly for testing)
        self._is_closed = False

        self.initialize()

    def initialize(self):
        self.resize(_main_window_geometry["initial_width"],
                    _main_window_geometry["initial_height"])

        self.setMinimumWidth(_main_window_geometry["min_width"])
        self.setMinimumHeight(_main_window_geometry["min_height"])

        self.setWindowTitle("PyXRF window title")

    def closeEvent(self, event):

        event.accept()

        # This flag is used for CI tests
        self._is_closed = True
