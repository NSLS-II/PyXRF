from PyQt5.QtWidgets import QMainWindow, QMessageBox


_main_window_geometry = {
    "initial_height": 800,
    "initial_width": 1200,
    "min_height": 800,
    "min_width": 1200,
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

        mb_close = QMessageBox(QMessageBox.Question, "Exit",
                               "Are you sure you want to EXIT the program?",
                               QMessageBox.Yes | QMessageBox.No)
        if mb_close.exec() == QMessageBox.Yes:
            event.accept()
            # This flag is used for CI tests
            self._is_closed = True
        else:
            event.ignore()
