import sys
from PyQt5 import QtWidgets
from PyQt5.uic import loadUi

from menu_screen import MenuScreen
from search_screen import SearchScreen
from indexing_screen import IndexingScreen


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    widgets_stack = QtWidgets.QStackedWidget()

    widgets_stack.addWidget(MenuScreen(widgets_stack))
    widgets_stack.addWidget(SearchScreen(widgets_stack))
    widgets_stack.addWidget(IndexingScreen(widgets_stack))

    widgets_stack.setFixedWidth(1200)
    widgets_stack.setFixedHeight(600)
    widgets_stack.show()

    sys.exit(app.exec_())