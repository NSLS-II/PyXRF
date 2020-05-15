from PyQt5.QtWidgets import (QFrame, QSplitter, QHBoxLayout)
from PyQt5.QtCore import Qt
from .left_panel import LeftPanel
from .right_panel import RightPanel


class TwoPanelWidget(QSplitter):

    def __init__(self, *args, **kwargs):

        super().__init__(Qt.Horizontal)

        self.frame_left = QFrame(self)
        self.frame_left.setFrameShape(QFrame.StyledPanel)

        self.frame_right = QFrame(self)
        self.frame_right.setFrameShape(QFrame.StyledPanel)

        self.addWidget(self.frame_left)
        self.addWidget(self.frame_right)

        hbox = QHBoxLayout()
        left_panel = LeftPanel()
        hbox.addWidget(left_panel)
        self.frame_left.setLayout(hbox)

        hbox = QHBoxLayout()
        right_panel = RightPanel()
        hbox.addWidget(right_panel)
        self.frame_right.setLayout(hbox)

        self._show_first_time = True

    def showEvent(self, event):

        # Set the ratio for the splitter (only the first time the window is shown)
        if self._show_first_time:
            self.setSizes([400, self.width() - 400])
            self._show_first_time = False
