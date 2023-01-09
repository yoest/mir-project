import os
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.uic import loadUi
import sys

from hist_generator import * #import generate_color_hist, generate_hsv_hist


class IndexingScreen(QtWidgets.QMainWindow):

    def __init__(self, widgets_stack):
        super(IndexingScreen, self).__init__()
        self.widgets_stack = widgets_stack

        loadUi("indexing.ui", self)
        self.menu_btn.clicked.connect(self.go_to_menu_screen)
        
        self.charger.clicked.connect(self.open)

        self.tableView.clicked.connect(self.display_selected_image)
        
        self.indexer.clicked.connect(self.extract_features)

        self.quit_btn.clicked.connect(sys.exit)

    def go_to_menu_screen(self):
        self.widgets_stack.setCurrentIndex(self.widgets_stack.currentIndex() - 2)

    def open(self):
    
        self.images = []
        self.images_folder = QtWidgets.QFileDialog.getExistingDirectory(None,
            'Select directory', "../db", QtWidgets.QFileDialog.ShowDirsOnly)+"/"
        self.images = [self.images_folder + filename for filename in os.listdir(self.images_folder)]

        pixmap = QtGui.QPixmap(self.images[0])
        pixmap = pixmap.scaled(self.image.width(), self.image.height(), QtCore.Qt.KeepAspectRatio)
        self.image.setPixmap(pixmap)
        self.image.setAlignment(QtCore.Qt.AlignCenter)

        model = QtGui.QStandardItemModel()
        headerNames = ["File name"]
        model.setHorizontalHeaderLabels(headerNames)
        model.setColumnCount(1)

        for image_path in self.images:
            row = []
            first = image_path  # full path
            second = os.path.basename(image_path)  # just the file name
            item = QtGui.QStandardItem(first)
            item.setEditable(False)
            row.append(item)
            model.appendRow(row)

        self.tableView.setModel(model)
    
    def display_selected_image(self):
        index = self.tableView.selectionModel().currentIndex()
        image_url = index.sibling(index.row(), index.column()).data()
        pixmap = QtGui.QPixmap(image_url)
        pixmap = pixmap.scaled(self.image.width(), self.image.height(), QtCore.Qt.KeepAspectRatio)
        self.image.setPixmap(pixmap)
        self.image.setAlignment(QtCore.Qt.AlignCenter)

    def extract_features(self):
        
        """ Extract features using selected method(s) """
        # create a list of selected feature methods
        feature_methods = []
        if self.checkBox_HistC.isChecked():
            feature_methods.append("BGR")
        if self.checkBox_HSV.isChecked():
            feature_methods.append("HSV")
        if self.checkBox_SIFT.isChecked():
            feature_methods.append("SIFT")
        if self.checkBox_ORB.isChecked():
            feature_methods.append("ORB")
        if self.checkBox_HOG.isChecked():
            feature_methods.append("HOG")
        if self.checkBox_LBP.isChecked():
            feature_methods.append("LBP")
        if self.checkBox_GLCM.isChecked():
            feature_methods.append("GLCM")

        # check if any feature extraction method has been selected
        if not feature_methods:
            return
        combining = False
        # extract features using the selected method(s)
        if not combining:
            for feature_method in feature_methods:
                if feature_method == "BGR":
                    generate_color_hist(self.images_folder, self.progressBar)
                if feature_method == "HSV":
                    generate_hsv_hist(self.images_folder, self.progressBar)
                if feature_method == "SIFT":
                    generate_sift(self.images_folder, self.progressBar)
                if feature_method == "ORB":
                    generate_orb(self.images_folder, self.progressBar)
                if feature_method == "HOG":
                    generate_hog(self.images_folder, self.progressBar)
                if feature_method == "LBP":
                    generate_lbp(self.images_folder, self.progressBar)
                if feature_method == "GLCM":
                    generate_glcm(self.images_folder, self.progressBar)
        else:
            pass
            #global_generator(self.images_folder, self.progressBar, feature_methods)
        # calculate and display the size of the generated feature file
        file_name = f"../output/{feature_method}"
        file_stats = os.stat(file_name)
        print(f'File Size in Bytes is {file_stats.st_size}')
        print(f'File Size in KiloBytes is {file_stats.st_size / (1024)}')
        print(f'File Size in MegaBytes is {file_stats.st_size / (1024 * 1024)}')
