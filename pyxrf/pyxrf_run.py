import sys
from PyQt5.QtWidgets import QApplication

from .gui.main_window import MainWindow


def run():
    """Run the application"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    app.setApplicationName("PyXRF")

    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    run()
