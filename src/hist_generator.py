#from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QMessageBox
import os
import cv2
from utils import *
import numpy as np

from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure
from skimage import io, color, img_as_ubyte
from matplotlib import pyplot as plt
from skimage.feature import hog, greycomatrix, graycoprops, local_binary_pattern, greycoprops

def check_or_create(output_dir):
    #output_dir = "../output/" + output_dir
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

def global_generator(filenames, progress_bar, feature_methods):
    #output_dir = "../output/BGR_HSV_GLCM_LPB_HOG_SIFT_ORB"
    """given a list of features methods (BGR, HSV, GLCM, ...), combine only the descriptors present in features_methods"""
    dir = compute_feature_methods_name(feature_methods)
    output_dir = "../output/" + dir
    
    check_or_create(output_dir)

    for i, path in enumerate(os.listdir(filenames)):
        img = cv2.imread(filenames + "/" + path)
        
        dec_sift, dec_orb = None, None
        
        features = []
       
        if "BGR" in feature_methods:
            color_hist = compute_colors(img)
            features.append(color_hist)
        if "HSV" in feature_methods:
            hsv_hist = compute_hsv(img)
            features.append(hsv_hist)
        if "GLCM" in feature_methods:
            glcm = compute_glcm(img)
            features.append(glcm)
        if "LBP" in feature_methods:
            lbp = compute_lbp(img)
            features.append(lbp)
        if "HOG" in feature_methods:
            hog = compute_hog(img)
            features.append(hog)
        if "SIFT" in feature_methods:
            kps_sift, dec_sift = compute_sift(img)
            if dec_sift is not None:
                features.append(dec_sift)
        if "ORB" in feature_methods:
            kps_orb, dec_orb = compute_orb(img)
            if dec_orb is not None:
                features.append(dec_orb)
       
        feature = np.concatenate(features, axis=None)
        num_image, _ = path.split(".")
        np.savetxt(output_dir + "/" + str(num_image) + ".txt", feature)
        progress_bar.setValue(100 * ((i + 1) / len(os.listdir(filenames))))

    progress_bar.setValue(0)
    print(f"Indexing {dir} done")

def generate_color_hist(filenames, progress_bar):
    """ Generate color histogram for each image in the folder """
    if not os.path.isdir("../output/BGR"):
        os.mkdir("../output/BGR")

    for i, path in enumerate(os.listdir(filenames)):
        img = cv2.imread(filenames + "/" + path)
        feature = compute_colors(img)
        
        num_image, _ = path.split(".")
        np.savetxt("../output/BGR/" + str(num_image) + ".txt", feature)
        progress_bar.setValue(100 * ((i + 1) / len(os.listdir(filenames))))

    progress_bar.setValue(0)
    print("Indexing hist color done")


def generate_hsv_hist(filenames, progress_bar):
    """ Generate hsv histogram for each image in the folder """
    if not os.path.isdir("../output/HSV"):
        os.mkdir("../output/HSV")

    for i, path in enumerate(os.listdir(filenames)):
        img = cv2.imread(filenames + "/" + path)
        feature = compute_hsv(img)
        num_image, _ = path.split(".")
        np.savetxt("../output/HSV/" + str(num_image) + ".txt", feature)
        progress_bar.setValue(100 * ((i + 1) / len(os.listdir(filenames))))
    
    progress_bar.setValue(0)
    print("Indexing hist hsv done")

def reduce_resolution(img, new_width, new_height):

    # Réduire la résolution de l'image
    red_img = cv2.resize(img, (new_width, new_height))
    return red_img

        
def generate_sift(filenames, progress_bar):
    """Generate sift for each image in the folder"""
    if not os.path.isdir("../output/SIFT"):
        os.mkdir("../output/SIFT")

    for i, path in enumerate(os.listdir(filenames)):
        img = cv2.imread(filenames + "/" + path)
        kps, des = compute_sift(img)
        featureSum = 0
        num_image, _ = path.split(".")
       
        if(len(kps)) > 0:
            np.savetxt("../output/SIFT/" + str(num_image) + ".txt" ,des)
            progress_bar.setValue(100 * ((i + 1) / len(os.listdir(filenames))))
            featureSum += len(kps)
    
    progress_bar.setValue(0)
    print("Indexing sift done")

  

def generate_orb(filenames, progress_bar):
    """ Generate orb for each image in the folder """
    if not os.path.isdir("../output/ORB"):
        os.mkdir("../output/ORB")

    for i, path in enumerate(os.listdir(filenames)):
        img = cv2.imread(filenames + "/" + path)
        
        key_point1, descriptor1 = compute_orb(img)
        
        num_image, _ = path.split(".")
        if(descriptor1 is not None):
            np.savetxt("../output/ORB/" + str(num_image) + ".txt", descriptor1)
            progress_bar.setValue(100 * ((i + 1) / len(os.listdir(filenames))))
        

    progress_bar.setValue(0)
    print("Indexing orb done")


def generate_glcm(filenames, progress_bar):
    """ Generate glcm for each image in the folder """
    if not os.path.isdir("../output/GLCM"):
        os.mkdir("../output/GLCM")



    for i, path in enumerate(os.listdir(filenames)):
        img = cv2.imread(filenames + "/" + path)

        feature = compute_glcm(img)
        num_image, _ = path.split(".")
        np.savetxt("../output/GLCM/" + str(num_image) + ".txt", feature)
        progress_bar.setValue(100 * ((i + 1) / len(os.listdir(filenames))))

    progress_bar.setValue(0)
    print("Indexing glcm done")


def generate_lbp(filenames, progress_bar):
    """ Generate lbp for each image in the folder """
    if not os.path.isdir("../output/LBP"):
        os.mkdir("../output/LBP")

    for i, path in enumerate(os.listdir(filenames)):
        img = cv2.imread(filenames + "/" + path)
        
        histograms = compute_lbp(img)
        
        num_image, _ = path.split(".")
        np.savetxt("../output/LBP/" + str(num_image) + ".txt", histograms)
        progress_bar.setValue(100 * ((i + 1) / len(os.listdir(filenames))))

    progress_bar.setValue(0)
    print("Indexing lbp done")


def generate_hog(filenames, progress_bar):
    """ Generate hog for each image in the folder """
    if not os.path.isdir("../output/HOG"):
        os.mkdir("../output/HOG")

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
        np.savetxt("../output/HOG/" + str(num_image) + ".txt", feature)
        progress_bar.setValue(100 * ((i + 1) / len(os.listdir(filenames))))

    progress_bar.setValue(0)
    print("Indexing hog done")
