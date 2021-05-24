from qtpy.QtWidgets import QFrame, QSplitter, QHBoxLayout
from qtpy.QtCore import Qt
from .left_panel import LeftPanel
from .right_panel import RightPanel


class TwoPanelWidget(QSplitter):
    def __init__(self, *, gpc, gui_vars):

        super().__init__(Qt.Horizontal)

        # Global processing classes
        self.gpc = gpc
        # Global GUI variables (used for control of GUI state)
        self.gui_vars = gui_vars

        self.frame_left = QFrame(self)
        self.frame_left.setFrameShape(QFrame.StyledPanel)

        self.frame_right = QFrame(self)
        self.frame_right.setFrameShape(QFrame.StyledPanel)

        self.addWidget(self.frame_left)
        self.addWidget(self.frame_right)

        hbox = QHBoxLayout()
        self.left_panel = LeftPanel(gpc=self.gpc, gui_vars=self.gui_vars)
        hbox.addWidget(self.left_panel)
        self.frame_left.setLayout(hbox)

        hbox = QHBoxLayout()
        self.right_panel = RightPanel(gpc=self.gpc, gui_vars=self.gui_vars)
        hbox.addWidget(self.right_panel)
        self.frame_right.setLayout(hbox)

        self._show_first_time = True

        # Set stretch factor of the left panel to 0 (we want it to keep its width
        #   as the window is resized. The panel can still be resized manually
        self.setStretchFactor(0, 0)
        # Set stretch factor for the right panel to some non-zero value, e.g. 1
        self.setStretchFactor(1, 1)

    def showEvent(self, event):
        # Set the ratio for the splitter (only the first time the window is shown)
        if self._show_first_time:
            self.setSizes([460, self.width() - 460])
            self._show_first_time = False

    def update_widget_state(self, condition=None):
        self.left_panel.update_widget_state(condition)
        self.right_panel.update_widget_state(condition)
