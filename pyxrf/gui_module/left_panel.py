from PyQt5.QtWidgets import QTabWidget, QWidget


class LeftPanel(QTabWidget):

    def __init__(self):
        super().__init__()

        self.setTabPosition(QTabWidget.West)

        self.addTab(QWidget(), "Left 1")
        self.addTab(QWidget(), "Left 2")
