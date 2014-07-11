__author__ = 'edill'

from PyQt4 import QtCore, QtGui
import sys
from metadataStore.userapi.commands import search

class DisplayDictionary(QtGui.QTreeWidget):
    """
    fill_item and fill_widget were taken from:
    http://stackoverflow.com/questions/21805047/qtreewidget-to-mirror-python-dictionary
    """
    def __init__(self):
        QtGui.QTreeWidget.__init__(self)


    def fill_item(self, item, value):
        item.setExpanded(True)
        if type(value) is dict:
            for key, val in sorted(value.iteritems()):
                child = QtGui.QTreeWidgetItem()
                child.setText(0, unicode(key))
                item.addChild(child)
                self.fill_item(child, val)
        elif type(value) is list:
            for val in value:
                child = QtGui.QTreeWidgetItem()
                item.addChild(child)
                if type(val) is dict:
                    child.setText(0, '[dict]')
                    self.fill_item(child, val)
                elif type(val) is list:
                    child.setText(0, '[list]')
                    self.fill_item(child, val)
                else:
                    child.setText(0, unicode(val))
                child.setExpanded(True)
        else:
            child = QtGui.QTreeWidgetItem()
            child.setText(0, unicode(value))
            item.addChild(child)

    def fill_widget(self, value):
        """

        Parameters
        ----------
        value : list or dict
        """
        self.clear()
        self.fill_item(self.invisibleRootItem(), value)


class main_window(QtGui.QMainWindow):
     def __init__(self, parent=None):
        QtGui.QMainWindow.__init__(self, parent)

        # init the 1d stack main window
        self.setWindowTitle('Display dictionary example')
        self._display_dict = DisplayDictionary()
        self.setCentralWidget(self._display_dict)

        query = {"owner" : "arkilic"}
        result = search(**query)
        self._display_dict.fill_widget(result)

if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    tt = main_window()
    tt.show()
    sys.exit(app.exec_())