from PyQt5 import QtWidgets
from PyQt5.uic import loadUi
import sys


class MenuScreen(QtWidgets.QMainWindow):

    def __init__(self, widgets_stack):
        """ Initialize the menu screen and display the three button to access the indexing screen, search screen or quit the application """
        super(MenuScreen, self).__init__()
        self.widgets_stack = widgets_stack

        # Load the UI
        loadUi("menu.ui", self)
        self.search_btn.clicked.connect(self.go_to_search_screen)
        self.indexing_btn.clicked.connect(self.go_to_indexing_screen)
        self.quit_btn.clicked.connect(sys.exit)

    def go_to_search_screen(self):
        """ Navigate through the widgets stack to the search screen """
        self.widgets_stack.setCurrentIndex(self.widgets_stack.currentIndex() + 1)
        self.widgets_stack.currentWidget().initialize_desc()

    def go_to_indexing_screen(self):
        """ Navigate through the widgets stack to the indexing screen """
        self.widgets_stack.setCurrentIndex(self.widgets_stack.currentIndex() + 2)
