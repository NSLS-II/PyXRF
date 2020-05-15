from PyQt5.QtWidgets import QTabWidget, QWidget


class RightPanel(QTabWidget):

    def __init__(self):
        super().__init__()

        self.addTab(QWidget(), "Right 1")
        self.addTab(QWidget(), "Right 2")
