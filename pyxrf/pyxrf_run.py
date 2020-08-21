import sys
import platform
import argparse
from PyQt5.QtWidgets import QApplication, QStyleFactory
from PyQt5.QtGui import QFontDatabase, QPalette, QColor
from PyQt5.QtCore import Qt

from .gui_support.gpc_class import GlobalProcessingClasses
from .gui_module.main_window import MainWindow
from .gui_module.useful_widgets import global_gui_variables

import logging
logger = logging.getLogger("pyxrf")


try:
    # import databroker  # noqa: F401
    pass
except ImportError:
    global_gui_variables["gui_state"]["databroker_available"] = False
else:
    global_gui_variables["gui_state"]["databroker_available"] = True

# if hasattr(Qt, 'AA_EnableHighDpiScaling'):
#     QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)

# if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
#     QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)


def run():
    """Run the application"""
    parser = argparse.ArgumentParser(prog='pyxrf', description='Command line arguments')
    parser.add_argument("-l", "--loglevel", default="INFO", type=str, dest="loglevel",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Logger level. Set to 'DEBUG' in order to see debug information.")
    args = parser.parse_args()

    # Setup the Logger
    logger.setLevel(args.loglevel)
    formatter = logging.Formatter(fmt='%(asctime)s : %(levelname)s : %(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(args.loglevel)
    logger.addHandler(stream_handler)

    gpc = GlobalProcessingClasses()
    gpc.initialize()

    app = QApplication(sys.argv)

    # The default font looks bad on Windows, so one of the following (commonly available)
    #   fonts will be selected in the listed order
    windows_font_selection = ["Verdana", "Microsoft Sans Serif", "Segoe UI"]
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
        logger.info(f"Current OS: {current_os}")
        logger.info(f"Style '{style}' is not in the list of available styles {available_styles}.")
    app.setStyle(style)
    app.setApplicationName("PyXRF")
    # app.setStyleSheet('QWidget {font: "Roboto Mono"; font-size: 14px}')
    # app.setStyleSheet('QWidget {font-size: 14px}')

    # Run the application in 'Dark' theme
    dark_theme_on = False

    if dark_theme_on:
        # Custom palette for Dark Mode
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(53, 53, 53))
        palette.setColor(QPalette.WindowText, Qt.white)
        palette.setColor(QPalette.Base, QColor(25, 25, 25))
        palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
        palette.setColor(QPalette.ToolTipBase, Qt.white)
        palette.setColor(QPalette.ToolTipText, Qt.white)
        palette.setColor(QPalette.Text, Qt.white)
        palette.setColor(QPalette.Button, QColor(53, 53, 53))
        palette.setColor(QPalette.ButtonText, Qt.white)
        palette.setColor(QPalette.BrightText, Qt.red)
        palette.setColor(QPalette.Link, QColor(42, 130, 218))
        palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        palette.setColor(QPalette.HighlightedText, Qt.black)
        app.setPalette(palette)

    # Set font
    font = app.font()
    font.setPixelSize(14)
    if selected_font_family:
        logger.info(f"Replacing the default font with '{selected_font_family}'")
        font.setFamily(selected_font_family)
    app.setFont(font)

    main_window = MainWindow(gpc=gpc)
    main_window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    run()
