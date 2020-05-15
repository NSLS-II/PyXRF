import sys
from PyQt5.QtWidgets import QApplication

from .gui_module.main_window import MainWindow


def run():
    """Run the application"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    app.setApplicationName("PyXRF")
    # app.setStyleSheet('QWidget {font: "Roboto Mono"; font-size: 14px}')
    app.setStyleSheet('QWidget {font-size: 16px}')

    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    run()
