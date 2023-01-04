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
        """ This class is used to display the screen where the user can select a query image and search for similar images in the database
        """
        super(SearchScreen, self).__init__()
        self.widgets_stack = widgets_stack

        self.search_times = []
        self.average_precision_values = []
        self.metrics_computed_for_query = False
        self.already_search_for_query = False

        # Load the UI file and link the buttons to their functions
        loadUi("search.ui", self)

        self.menu_btn.clicked.connect(self.go_to_menu_screen)
        self.load_btn.clicked.connect(self.load_image_query)
        self.load_desc_btn.clicked.connect(self.load_features)
        self.search_btn.clicked.connect(self.search)
        self.calcul_rp_curve_btn.clicked.connect(self.plot_recall_precision)
        self.calcul_metric_btn.clicked.connect(self.show_metrics)

    def go_to_menu_screen(self):
        """ Change the current screen to the menu screen """
        self.widgets_stack.setCurrentIndex(self.widgets_stack.currentIndex() - 1)

    def load_image_query(self):
        """ Let the user select an image in the database to use as a query """
        self.filename, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Select Image", "../db", "Image Files (*.png *.jpeg *.jpg *.bmp)")
        pixmap = QtGui.QPixmap(self.filename)
        pixmap = pixmap.scaled(self.label_requete.width(),
        self.label_requete.height(), QtCore.Qt.KeepAspectRatio)
        self.label_requete.setPixmap(pixmap)
        self.label_requete.setAlignment(QtCore.Qt.AlignCenter)

        # Compute the maximum number of images that can be displayed in the scroll area (to have the top max) to add it in the combo box
        self.combo_box_k.clear()

        filename_req = os.path.basename(self.filename)
        filename_without_extension, _ = filename_req.split(".")

        # NB: The class is the first digit (which is the category) that we sum with the second digit (which is the sub-category)
        class_image_query = int(filename_without_extension[0]) * 10 + int(filename_without_extension[2])

        nb_images_same_class = 0
        for file in os.listdir("../db"):
            class_image = int(file[0]) * 10 + int(file[2])
            if class_image == class_image_query:
                nb_images_same_class += 1

        k_value_to_add = [10, 20, 50, 100, 200]
        for k in k_value_to_add:
            if k <= nb_images_same_class:
                self.combo_box_k.addItem(str(k))
        self.combo_box_k.addItem(str(nb_images_same_class))

    def load_features(self):
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

        # Reset the area where the images will be displayed
        self.scrollarea_content = QtWidgets.QWidget()
        self.scroll_area.setWidget(self.scrollarea_content)

        # Change the combo_box_distance items based on the selected algorithm
        if self.algo_choice == 3 or self.algo_choice == 4:
            self.combo_box_distance.clear()
            self.combo_box_distance.addItems(["Brute force", "Flann"])
        else :
            self.combo_box_distance.clear()
            self.combo_box_distance.addItems(["Euclidienne", "Correlation", "Chi carre", "Intersection", "Bhattacharyya"])
        
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
        message_box = QMessageBox()
        message_box.setIcon(QMessageBox.Information)
        message_box.setText("Vous devez sélectionner un descripteur via le menu ci-dessus")
        message_box.setWindowTitle("Pas de Descripteur sélectionné")
        message_box.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        return message_box.exec()

    def search(self):
        """ Search the nearest images related to the query image """
        start_time = time.time()

        # Reset to avoid computing the metrics for the previous query
        self.metrics_computed_for_query = False
        self.already_search_for_query = False

        # Reset the grid layout to avoid displaying the previous results
        self.scrollarea_content = QtWidgets.QWidget()
        self.scroll_area.setWidget(self.scrollarea_content)

        # Generate features of the query image
        req = features_extractor.extract_req_features(self.filename, self.algo_choice)

        # Get the nearest images based on a chosen distance
        self.number_neighboor = int(self.combo_box_k.currentText())

        distance_name = self.combo_box_distance.currentText()
        neighboor = distances.get_k_voisins(self.features, req, self.number_neighboor, distance_name)

        self.path_nearer_image = []
        self.name_nearer_image = []
        for k in range(self.number_neighboor):
            self.path_nearer_image.append(neighboor[k][0])
            self.name_nearer_image.append(os.path.basename(neighboor[k][0]))

        # Show the images on a given number of columns
        gridLayout = QtWidgets.QGridLayout()

        number_columns = 2
        k = 0
        for i in range(math.ceil(self.number_neighboor / number_columns)):
            for j in range(number_columns):
                if k >= len(self.path_nearer_image):
                    break

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
                label.setPixmap(pixmap.scaled(0.3 * width, 0.3 * height, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
                gridLayout.addWidget(label, i, j)
                k += 1

        self.scrollarea_content.setLayout(gridLayout)

        # Compute the mean search time by image
        end_time = time.time()
        time_taken = end_time - start_time
        self.search_times.append(time_taken)

        print('Time taken:', time_taken, 'seconds')
        print('Mean search time:', np.mean(self.search_times), 'seconds (for', len(self.search_times), 'images)\n')

        self.already_search_for_query = True


    def compute_metrics(self):
        """ Compute the recall and precision based on the nearest images """
        # Do not compute the metrics if they have already been computed (for the same query image)
        if self.metrics_computed_for_query:
            return
        self.metrics_computed_for_query = True

        recall_precision, self.recalls, self.precisions = [], [], []
        filename_req = os.path.basename(self.filename)
        filename_without_extension, _ = filename_req.split(".")
        self.num_image = filename_without_extension.split("_")[-1]

        # The class is the first digit (which is the category) that we sum with the second digit (which is the sub-category)
        class_image_query = int(filename_without_extension[0]) * 10 + int(filename_without_extension[2])

        # Add all the pertinent images in the list
        for j in range(self.number_neighboor):
            class_near_image = int(self.name_nearer_image[j][0]) * 10 + int(self.name_nearer_image[j][2])

            if class_image_query == class_near_image:
                recall_precision.append(True) # Right class 
            else:
                recall_precision.append(False) # Wrong class

        # -- Compute the recall and precision -- 
        for i in range(self.number_neighboor):
            j = i
            number_relevant_images_retrieved = 0
            while(j >= 0):
                if recall_precision[j]:
                    number_relevant_images_retrieved += 1
                j -= 1

            precision = number_relevant_images_retrieved / (i + 1)
            recall = number_relevant_images_retrieved / self.number_neighboor

            self.recalls.append(recall)
            self.precisions.append(precision)

        # -- Compute the other metrics (AP, mAP, R-precision) --        
        self.average_precision = sum(self.precisions) / len(self.precisions)
        self.average_precision_values.append(self.average_precision)
        self.mean_average_precision = sum(self.average_precision_values) / len(self.average_precision_values)

        # The position R of the R-precision is either the number of images in the db with the same class as the query image or the number of images retrieved if there is less images retrieved than the number of relevant images
        number_images_with_same_class = 0
        for file in os.listdir('../db/'):
            class_image_db = int(file[0]) * 10 + int(file[2])
            if class_image_db == class_image_query:
                number_images_with_same_class += 1

        idx_in_precision = min(self.number_neighboor, number_images_with_same_class)
        self.r_precision = sum(self.precisions[:idx_in_precision + 1]) / idx_in_precision


    def show_metrics(self):
        """ Show a dialog box with the metrics """
        # Do not show the metrics if the search has not been done
        if not self.already_search_for_query:
            self.show_no_search_applied()
            return

        if not self.metrics_computed_for_query:
            self.compute_metrics()

        message_box = QMessageBox()
        message_box.setIcon(QMessageBox.Information)
        message_box.setText(f'Rappel (R) : {self.recalls[-1]}\nPrécision (P) : {self.precisions[-1]}\nAverage Precision (AP) : {self.average_precision}\nMean Average Precision (MaP) : {self.mean_average_precision} (pour {len(self.average_precision_values)} requêtes)\nR-Precision : {self.r_precision}')
        message_box.setWindowTitle("Métriques")
        message_box.setStandardButtons(QMessageBox.Ok)
        return message_box.exec()

    def plot_recall_precision(self):
        """ Plot the recall precision curve """
        # Do not show the plot if the search has not been done
        if not self.already_search_for_query:
            self.show_no_search_applied()
            return

        if not self.metrics_computed_for_query:
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
        pixmap = QtGui.QPixmap.fromImage(q_img)
        width = self.label_requete.frameGeometry().width()
        height = self.label_requete.frameGeometry().height()
        self.label_courbe.setAlignment(QtCore.Qt.AlignCenter)
        self.label_courbe.setPixmap(pixmap.scaled(width, height, QtCore.Qt.KeepAspectRatio,QtCore.Qt.SmoothTransformation))

    def show_no_search_applied(self):
        """ Show a dialog box when the search button has not been pressed yet """
        message_box = QMessageBox()
        message_box.setIcon(QMessageBox.Information)
        message_box.setText("Vous devez d'abord appuyer sur le bouton 'Rechercher' avant d'effectuer cette action")
        message_box.setWindowTitle("Pas encore de recherche effectuée")
        message_box.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        return message_box.exec()