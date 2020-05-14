from PyQt5.QtWidgets import (QFrame, QSplitter)
from PyQt5.QtCore import Qt


class TwoPanelWidget(QSplitter):

    def __init__(self, *args, **kwargs):

        super().__init__(Qt.Horizontal)

        self.frame_left = QFrame(self)
        self.frame_left.setFrameShape(QFrame.StyledPanel)

        self.frame_right = QFrame(self)
        self.frame_right.setFrameShape(QFrame.StyledPanel)

        self.addWidget(self.frame_left)
        self.addWidget(self.frame_right)

        # Don't stretch the left panel of the splitter
        self.setStretchFactor(0, 2)
        self.setStretchFactor(1, 5)
