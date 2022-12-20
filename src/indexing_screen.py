from PyQt5 import QtWidgets
from PyQt5.uic import loadUi
import sys


class IndexingScreen(QtWidgets.QMainWindow):

    def __init__(self, widgets_stack):
        super(IndexingScreen, self).__init__()
        self.widgets_stack = widgets_stack

        loadUi("indexing.ui", self)
        self.menu_btn.clicked.connect(self.go_to_menu_screen)
        self.quit_btn.clicked.connect(sys.exit)

    def go_to_menu_screen(self):
        self.widgets_stack.setCurrentIndex(self.widgets_stack.currentIndex() - 2)