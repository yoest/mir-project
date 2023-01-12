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
        if len(feature_methods) > 1:
            combining = True
        else:
            combining = False
        # extract features using the selected method(s)
        if not combining:
            for feature_method in feature_methods:
                if feature_method == "BGR":
                    #compute_execution_time(generate_color_hist, (self.images_folder, self.progressBar))
                    #compute_file_size("../output/BGR")
                    generate_color_hist(self.images_folder, self.progressBar)
                if feature_method == "HSV":
                    #compute_file_size("../output/HSV")
                    #compute_execution_time(generate_hsv_hist, (self.images_folder, self.progressBar))
                    generate_hsv_hist(self.images_folder, self.progressBar)
                if feature_method == "SIFT":
                    #compute_file_size("../output/SIFT")
                    #compute_execution_time(generate_sift,(self.images_folder, self.progressBar))
                    generate_sift(self.images_folder, self.progressBar)
                if feature_method == "ORB":
                    #compute_file_size("../output/ORB")
                    #compute_execution_time(generate_orb,(self.images_folder, self.progressBar))
                    generate_orb(self.images_folder, self.progressBar)
                if feature_method == "HOG":
                    #compute_file_size("../output/ORB")
                    #compute_execution_time(generate_hog,(self.images_folder, self.progressBar))
                    generate_hog(self.images_folder, self.progressBar)
                if feature_method == "LBP":
                    #compute_file_size("../output/LBP")
                    #compute_execution_time(generate_lbp, (self.images_folder, self.progressBar))
                    generate_lbp(self.images_folder, self.progressBar)
                if feature_method == "GLCM":
                    #compute_execution_time(generate_glcm,(self.images_folder, self.progressBar))
                    #compute_file_size("../output/GLCM")
                    generate_glcm(self.images_folder, self.progressBar)
        else:
            global_generator(self.images_folder, self.progressBar, feature_methods)
       
