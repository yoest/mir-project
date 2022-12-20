from PyQt5 import QtWidgets
from PyQt5.uic import loadUi


class SearchScreen(QtWidgets.QMainWindow):

    def __init__(self, widgets_stack):
        super(SearchScreen, self).__init__()
        self.widgets_stack = widgets_stack

        loadUi("search.ui", self)
        self.menu_btn.clicked.connect(self.go_to_menu_screen)

    def go_to_menu_screen(self):
        self.widgets_stack.setCurrentIndex(self.widgets_stack.currentIndex() - 1)