import sys
import platform
from PyQt5.QtWidgets import QApplication, QStyleFactory
from PyQt5.QtCore import Qt

from .gui_module.main_window import MainWindow

# if hasattr(Qt, 'AA_EnableHighDpiScaling'):
#     QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)

# if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
#     QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)


def run():
    """Run the application"""
    app = QApplication(sys.argv)
    current_os = platform.system()
    if current_os == "Linux":
        style = "Fusion"
    elif current_os == "Windows":
        style = "Windows"
    elif current_os == "Darwin":
        style = "Macintosh"

    available_styles = list(QStyleFactory().keys())
    print(f"Available styles: {available_styles}")
    if style not in available_styles:
        print(f"Current OS: {current_os}")
        print(f"Style '{style}' is not in the list of available styles {available_styles}.")
    app.setStyle(style)
    app.setApplicationName("PyXRF")
    # app.setStyleSheet('QWidget {font: "Roboto Mono"; font-size: 14px}')
    app.setStyleSheet('QWidget {font-size: 14px}')

    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    run()
