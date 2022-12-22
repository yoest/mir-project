#Defintion de toute les fonctions à appeller dans l'interface
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QMessageBox
import os
import cv2
import numpy as np
from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure
from skimage import io, color, img_as_ubyte
from matplotlib import pyplot as plt
from skimage.feature import hog, greycomatrix, graycoprops, local_binary_pattern, greycoprops



def show_no_selected_dialog():
    """ Show a dialog box when no descriptor is selected """
    msgBox = QMessageBox()
    msgBox.setIcon(QMessageBox.Information)
    msgBox.setText("Merci de sélectionner un descripteur via le menu ci-dessus")
    msgBox.setWindowTitle("Pas de Descripteur sélectionné")
    msgBox.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
    return msgBox.exec()


def generate_color_hist(filenames, progress_bar):
    """ Generate color histogram for each image in the folder """
    if not os.path.isdir("BGR"):
        os.mkdir("BGR")

    for i, path in enumerate(os.listdir(filenames)):
        img = cv2.imread(filenames + "/" + path)
        hist_b = cv2.calcHist([img], [0], None, [256], [0,256])
        hist_g = cv2.calcHist([img], [1], None, [256], [0,256])
        hist_r = cv2.calcHist([img], [2], None, [256], [0,256])
        feature = np.concatenate((hist_b, np.concatenate((hist_g, hist_r), axis=None)), axis=None)

        num_image, _ = path.split(".")
        np.savetxt("BGR/" + str(num_image) + ".txt", feature)
        progress_bar.setValue(100 * ((i + 1) / len(os.listdir(filenames))))

    print("Indexing hist color done")


def generate_hsv_hist(filenames, progress_bar):
    """ Generate hsv histogram for each image in the folder """
    if not os.path.isdir("HSV"):
        os.mkdir("HSV")

    for i, path in enumerate(os.listdir(filenames)):
        img = cv2.imread(filenames + "/" + path)
        hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hist_h = cv2.calcHist([hsv_image], [0], None, [256], [0,256])
        hist_s = cv2.calcHist([hsv_image], [1], None, [256], [0,256])
        hist_v = cv2.calcHist([hsv_image], [2], None, [256], [0,256])

        feature = np.concatenate((hist_v, np.concatenate((hist_s, hist_h), axis=None)), axis=None)

        num_image, _ = path.split(".")
        np.savetxt("HSV/" + str(num_image) + ".txt", feature)
        progress_bar.setValue(100 * ((i + 1) / len(os.listdir(filenames))))
    
    print("Indexing hist hsv done")

        
def generate_sift(filenames, progress_bar):
    """ Generate sift for each image in the folder """
    if not os.path.isdir("SIFT"):
        os.mkdir("SIFT")

    for i, path in enumerate(os.listdir(filenames)):
        img = cv2.imread(filenames + "/" + path)
        featureSum = 0
        sift = cv2.SIFT_create()  
        kps , des = sift.detectAndCompute(img,None)

        num_image, _ = path.split(".")
        np.savetxt("SIFT/" + str(num_image) + ".txt" ,des)
        progress_bar.setValue(100 * ((i + 1) / len(os.listdir(filenames))))
        
        featureSum += len(kps)
    
    print("Indexing sift done")
  

def generate_orb(filenames, progress_bar):
    """ Generate orb for each image in the folder """
    if not os.path.isdir("ORB"):
        os.mkdir("ORB")

    for i, path in enumerate(os.listdir(filenames)):
        img = cv2.imread(filenames + "/" + path)
        orb = cv2.ORB_create()
        _, descriptor1 = orb.detectAndCompute(img, None)
        
        num_image, _ = path.split(".")
        np.savetxt("ORB/" + str(num_image) + ".txt", descriptor1)
        progress_bar.setValue(100 * ((i + 1) / len(os.listdir(filenames))))

    print("Indexing orb done")


def generate_glcm(filenames, progress_bar):
    """ Generate glcm for each image in the folder """
    if not os.path.isdir("GLCM"):
        os.mkdir("GLCM")

    distances = [1,-1]
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

    for i, path in enumerate(os.listdir(filenames)):
        image = cv2.imread(filenames + "/" + path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = img_as_ubyte(gray)
        glcm_matrix = greycomatrix(gray, distances=distances, angles=angles, normed=True)
        glcm_prop_1 = graycoprops(glcm_matrix, 'contrast').ravel()
        glcm_prop_2 = graycoprops(glcm_matrix, 'dissimilarity').ravel()
        glcm_prop_3 = graycoprops(glcm_matrix, 'homogeneity').ravel()
        glcm_prop_4 = graycoprops(glcm_matrix, 'energy').ravel()
        glcm_prop_5 = graycoprops(glcm_matrix, 'correlation').ravel()
        glcm_prop_6 = graycoprops(glcm_matrix, 'ASM').ravel()
        feature = np.array([glcm_prop_1, glcm_prop_2, glcm_prop_3, glcm_prop_4, glcm_prop_5, glcm_prop_6]).ravel()
        num_image, _ = path.split(".")
        np.savetxt("GLCM/" + str(num_image) + ".txt", feature)
        progress_bar.setValue(100 * ((i + 1) / len(os.listdir(filenames))))

    print("Indexing glcm done")


def generate_lbp(filenames, progress_bar):
    """ Generate lbp for each image in the folder """
    if not os.path.isdir("LBP"):
        os.mkdir("LBP")

    for i, path in enumerate(os.listdir(filenames)):
        img = cv2.imread(filenames + "/" + path)
        points = 8
        radius = 1
        method='default'
        sub_size = (70, 70)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (350, 350))
        full_lbp_matrix = local_binary_pattern(img, points, radius, method)
        histograms = []
        for k in range(int(full_lbp_matrix.shape[0] / sub_size[0])):
            for j in range(int(full_lbp_matrix.shape[1] / sub_size[1])):
                sub_vector = full_lbp_matrix[k * sub_size[0]:(k+1) * sub_size[0], j * sub_size[1]:(j+1)*sub_size[1]].ravel()
                sub_hist, _ = np.histogram(sub_vector, bins=int(2**points), range=(0, 2**points))
                histograms = np.concatenate((histograms, sub_hist), axis=None)
        num_image, _ = path.split(".")
        np.savetxt("LBP/" + str(num_image) + ".txt", histograms)
        progress_bar.setValue(100 * ((i + 1) / len(os.listdir(filenames))))

    print("Indexing lbp done")


def generate_hog(filenames, progress_bar):
    """ Generate hog for each image in the folder """
    if not os.path.isdir("HOG"):
        os.mkdir("HOG")

    cell_size = (25, 25)
    block_size = (50, 50)
    block_stride = (25, 25)
    n_bins = 9
    win_size = (350, 350)
    for i, path in enumerate(os.listdir(filenames)):
        img = cv2.imread(filenames + "/" + path)
        image = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, win_size)
        hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, n_bins)
        feature = hog.compute(image)
        num_image, _ = path.split(".")
        np.savetxt("HOG/" + str(num_image) + ".txt", feature)
        progress_bar.setValue(100 * ((i + 1) / len(os.listdir(filenames))))

    print("Indexing hog done")