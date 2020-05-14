from PyQt5.QtWidgets import QMainWindow, QMessageBox, QLabel, QAction


_main_window_geometry = {
    "initial_height": 800,
    "initial_width": 1200,
    "min_height": 800,
    "min_width": 1200,
}


def position_window(child, parent):
    """
    Position `child` window in respect to the `parent` window in visually
    acceptable way.

    Parameters
    ----------
    child: QWidget
        reference to the child window

    parent: QWidget
        reference to the parent window
    """
    frm_parent = parent.geometry()

    x_center = frm_parent.width() // 3 + frm_parent.x()
    y_center = frm_parent.height() // 3 + frm_parent.y()

    child.move(x_center, y_center)


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

        # Status bar
        self.statusLabel = QLabel()
        self.statusBar().addWidget(self.statusLabel)
        self.statusLabelDefaultText = \
            "Load data by selecting one of the options in 'Load Data' menu ..."
        self.statusLabel.setText(self.statusLabelDefaultText)

        # Define some actions
        action_open_file = QAction("&Open data file", self)
        action_open_file.setStatusTip('Load data from file')

        action_load_from_db = QAction("Load data from &database", self)
        action_load_from_db.setStatusTip('Load data from database')

        action_load_params = QAction("Load &processing parameters", self)
        action_load_params.setStatusTip('Load processing parameters from file')

        action_refine_params = QAction("Re&fine processing parameters", self)
        action_refine_params.setStatusTip(
            "Refine processing parameters by fitting total spectrum")

        action_gen_xrf_map = QAction("&Compute XRF map", self)
        action_gen_xrf_map.setStatusTip(
            "Compute XRF map by individually fitting of spectra for each pixel")

        action_show_about = QAction("&About", self)
        action_show_about.setStatusTip("Show information about this program")

        # Main menu
        menubar = self.menuBar()
        loadData = menubar.addMenu('&Load Data')
        loadData.addAction(action_open_file)
        loadData.addAction(action_load_from_db)
        loadData.addSeparator()
        loadData.addAction(action_load_params)

        proc = menubar.addMenu('&Processing')
        proc.addAction(action_refine_params)
        proc.addAction(action_gen_xrf_map)

        help = menubar.addMenu('&Help')
        help.addAction(action_show_about)

    def closeEvent(self, event):

        mb_close = QMessageBox(QMessageBox.Question, "Exit",
                               "Are you sure you want to EXIT the program?",
                               QMessageBox.Yes | QMessageBox.No)
        mb_close.setDefaultButton(QMessageBox.No)

        position_window(mb_close, self)

        if mb_close.exec() == QMessageBox.Yes:
            event.accept()
            # This flag is used for CI tests
            self._is_closed = True
        else:
            event.ignore()
