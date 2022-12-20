from PyQt5 import QtWidgets
from PyQt5.uic import loadUi
import sys


class MenuScreen(QtWidgets.QMainWindow):

    def __init__(self, widgets_stack):
        super(MenuScreen, self).__init__()
        self.widgets_stack = widgets_stack

        loadUi("menu.ui", self)
        self.search_btn.clicked.connect(self.go_to_search_screen)
        self.indexing_btn.clicked.connect(self.go_to_indexing_screen)
        self.quit_btn.clicked.connect(sys.exit)

    def go_to_search_screen(self):
        self.widgets_stack.setCurrentIndex(self.widgets_stack.currentIndex() + 1)

    def go_to_indexing_screen(self):
        self.widgets_stack.setCurrentIndex(self.widgets_stack.currentIndex() + 2)
