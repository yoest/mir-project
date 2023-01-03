import time
from PyQt5 import QtWidgets
from PyQt5.uic import loadUi

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import QFileDialog

import cv2
import numpy as np
import os
import math
import matplotlib.pyplot as plt

from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QMessageBox

import features_extractor
import distances



class SearchScreen(QtWidgets.QMainWindow):

    def __init__(self, widgets_stack):
        super(SearchScreen, self).__init__()
        self.widgets_stack = widgets_stack

        self.search_times = []

        loadUi("search.ui", self)
        self.menu_btn.clicked.connect(self.go_to_menu_screen)
        self.load_btn.clicked.connect(self.load_db)
        self.load_desc_btn.clicked.connect(self.load_features)
        self.search_btn.clicked.connect(self.search)
        self.calcul_rp_curve_btn.clicked.connect(self.plot_recall_precision)
        self.calcul_metric_btn.clicked.connect(self.show_metrics)

    def go_to_menu_screen(self):
        self.widgets_stack.setCurrentIndex(self.widgets_stack.currentIndex() - 1)

    def load_db(self):
        self.filename, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Select Image", "../db", "Image Files (*.png *.jpeg *.jpg *.bmp)")
        pixmap = QtGui.QPixmap(self.filename)
        pixmap = pixmap.scaled(self.label_requete.width(),
        self.label_requete.height(), QtCore.Qt.KeepAspectRatio)
        self.label_requete.setPixmap(pixmap)
        self.label_requete.setAlignment(QtCore.Qt.AlignCenter)

    def load_features(self, MainWindow):
        """ Load the features of the descriptors """
        choices = {
            1: (self.checkbox_hist_color, './BGR'),
            2: (self.checkbox_hsv, './HSV'),
            3: (self.checkbox_sift, './SIFT'),
            4: (self.checkbox_orb, './ORB'),
            5: (self.checkbox_glcm, './GLCM'),
            6: (self.checkbox_lbp, './LBP'),
            7: (self.checkbox_hog, './HOG'),
            8: (self.checkbox_moments, './MOMENTS'),
            9: (self.checkbox_other, './OTHER')
        }

        # Check if at least one checkbox is checked
        one_is_checked = False
        for _, (checkbox, _) in choices.items():
            if checkbox.isChecked():
                one_is_checked = True
                break
            
        if not one_is_checked:
            self.show_no_selected_dialog()
            return

        # Select the right folder and algorithm based on the checked checkboxes
        folder_model = ""
        for idx, (checkbox, folder) in choices.items():
            if checkbox.isChecked():
                folder_model = folder
                self.algo_choice = idx

        for i in reversed(range(self.gridLayout.count())):
            self.gridLayout.itemAt(i).widget().setParent(None)

        # Change the combobox items based on the selected algorithm
        if self.algo_choice == 3 or self.algo_choice == 4:
            self.comboBox.clear()
            self.comboBox.addItems(["Brute force", "Flann"])
        else :
            self.comboBox.clear()
            self.comboBox.addItems(["Euclidienne", "Correlation", "Chi carre", "Intersection", "Bhattacharyya"])
        
        # Load the features into the below variable
        self.features = []
        folder_model = '../output/' + folder_model
        all_files = os.listdir(folder_model)
        for i, file in enumerate(all_files): 
            data = os.path.join(folder_model, file)

            if not data.endswith(".txt"):
                continue

            feature = np.loadtxt(data)
            self.features.append((os.path.join("../db", os.path.basename(data).split('.')[0] + '.jpg'), feature))

            self.progress_bar.setValue(int(100 * ((i + 1) / len(all_files))))

        self.progress_bar.setValue(0)

    def show_no_selected_dialog(self):
        """ Show a dialog box when no descriptor is selected """
        msgBox = QMessageBox()
        msgBox.setIcon(QMessageBox.Information)
        msgBox.setText("Vous devez sélectionner un descripteur via le menu ci-dessus")
        msgBox.setWindowTitle("Pas de Descripteur sélectionné")
        msgBox.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        return msgBox.exec()

    def search(self, MainWindow):
        """ Search the nearest images related to the query image """
        start_time = time.time()

        # Reset the grid layout
        for i in reversed(range(self.gridLayout.count())):
            self.gridLayout.itemAt(i).widget().setParent(None)

        # Generate features of the query image
        req = features_extractor.extract_req_features(self.filename, self.algo_choice)

        # Get the nearest images
        self.number_neighboor = 9

        distance_name = self.comboBox.currentText()
        neighboor = distances.get_k_voisins(self.features, req, self.number_neighboor, distance_name)

        self.path_nearer_image = []
        self.name_nearer_image = []
        for k in range(self.number_neighboor):
            self.path_nearer_image.append(neighboor[k][0])
            self.name_nearer_image.append(os.path.basename(neighboor[k][0]))

        # Show the images on a given number of columns
        number_columns = 3
        k = 0
        for i in range(math.ceil(self.number_neighboor / number_columns)):
            for j in range(number_columns):
                img = cv2.imread(self.path_nearer_image[k], 1)

                # Swith the image to rgb
                b, g, r = cv2.split(img) 
                img = cv2.merge([r, g, b]) 

                # Convert the image to QImage
                height, width, channel = img.shape
                bytes_per_line = 3 * width
                qImg = QtGui.QImage(img.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
                pixmap = QtGui.QPixmap.fromImage(qImg)
                label = QtWidgets.QLabel("")
                label.setAlignment(QtCore.Qt.AlignCenter)
                label.setPixmap(pixmap.scaled(0.3*width, 0.3 * height, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
                self.gridLayout.addWidget(label, i, j)
                k += 1

        # Compute the mean search time by image
        end_time = time.time()
        time_taken = end_time - start_time
        self.search_times.append(time_taken)

        print('Time taken:', time_taken, 'seconds')
        print('Mean search time:', np.mean(self.search_times), 'seconds (for', len(self.search_times), 'images)\n')


    def compute_metrics(self):
        """ Compute the recall and precision based on the nearest images """
        recall_precision, self.recalls, self.precisions = [], [], []
        filename_req = os.path.basename(self.filename)
        filename_without_extension, _ = filename_req.split(".")
        self.num_image = filename_without_extension.split("_")[-1]
        class_image_query = int(self.num_image) / 100
        val = 0

        # Add all the pertinent images in the list
        for j in range(self.number_neighboor):
            class_near_image = (int(self.name_nearer_image[j].split('.')[0].split("_")[-1]))/100
            class_image_query = int(class_image_query)
            class_near_image = int(class_near_image)

            if class_image_query == class_near_image:
                recall_precision.append(True) # Right class 
                val += 1
            else:
                recall_precision.append(False) # Wrong class

        for i in range(self.number_neighboor):
            j = i
            val = 0
            while(j >= 0):
                if recall_precision[j]:
                    val += 1
                j -= 1

            precision = val / (i + 1)
            recall = val / self.number_neighboor

            self.recalls.append(recall)
            self.precisions.append(precision)


    def show_metrics(self):
        """ Show a dialog box when no descriptor is selected """
        if not hasattr(self, 'recalls'):
            self.compute_metrics()

        msgBox = QMessageBox()
        msgBox.setIcon(QMessageBox.Information)
        msgBox.setText("Rappel: " + str(self.recalls[-1]) + "\nPrécision: " + str(self.precisions[-1])) 
        msgBox.setWindowTitle("Métriques")
        msgBox.setStandardButtons(QMessageBox.Ok)
        return msgBox.exec()

    def plot_recall_precision(self):
        """ Plot the recall precision curve """
        if not hasattr(self, 'recalls'):
            self.compute_metrics()

        # Create and plot the recall precision curve
        plt.plot(self.recalls, self.precisions)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("R/P" + str(self.number_neighboor) + " voisins de l'image n°" + self.num_image)

        # Save the recall precision curve
        save_folder = os.path.join("../search_output", self.num_image)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        save_name = os.path.join(save_folder, self.num_image + '.png')
        plt.savefig(save_name, format='png', dpi=600)
        plt.close()

        # Show the recall precision curve
        img = cv2.imread(save_name,1) 
        b, g, r = cv2.split(img)
        img = cv2.merge([r, g, b]) 

        # Convert the image to QImage
        height, width, channel = img.shape
        bytesPerLine = 3 * width
        q_img = QtGui.QImage(img.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
        pixmap=QtGui.QPixmap.fromImage(q_img)
        width = self.label_requete.frameGeometry().width()
        height = self.label_requete.frameGeometry().height()
        self.label_courbe.setAlignment(QtCore.Qt.AlignCenter)
        self.label_courbe.setPixmap(pixmap.scaled(width, height, QtCore.Qt.KeepAspectRatio,QtCore.Qt.SmoothTransformation))