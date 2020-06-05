import sys
import platform
from PyQt5.QtWidgets import QApplication, QStyleFactory
from PyQt5.QtGui import QFontDatabase
# from PyQt5.QtCore import Qt

from .gui_module.main_window import MainWindow

# if hasattr(Qt, 'AA_EnableHighDpiScaling'):
#     QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)

# if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
#     QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)


def run():
    """Run the application"""
    app = QApplication(sys.argv)

    # The default font looks bad on Windows, so one of the following (commonly available)
    #   fonts will be selected in the listed order
    windows_font_selection = ["Segoe UI", "Verdana", "Microsoft Sans Serif"]
    available_font_families = list(QFontDatabase().families())
    selected_font_family = None

    current_os = platform.system()
    if current_os == "Linux":
        style = "Fusion"
    elif current_os == "Windows":
        style = "Fusion"
        # Select font
        for font_family in windows_font_selection:
            if font_family in available_font_families:
                selected_font_family = font_family
                break
    elif current_os == "Darwin":
        style = "Fusion"

    available_styles = list(QStyleFactory().keys())
    if style not in available_styles:
        print(f"Current OS: {current_os}")
        print(f"Style '{style}' is not in the list of available styles {available_styles}.")
    app.setStyle(style)
    app.setApplicationName("PyXRF")
    # app.setStyleSheet('QWidget {font: "Roboto Mono"; font-size: 14px}')
    # app.setStyleSheet('QWidget {font-size: 14px}')

    # Set font
    font = app.font()
    font.setPixelSize(14)
    if selected_font_family:
        print(f"Replacing the default font with '{selected_font_family}'")
        font.setFamily(selected_font_family)
    app.setFont(font)

    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    run()
