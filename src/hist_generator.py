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



def generate_color_hist(filenames, progress_bar):
    """ Generate color histogram for each image in the folder """
    if not os.path.isdir("../output/BGR"):
        os.mkdir("../output/BGR")

    for i, path in enumerate(os.listdir(filenames)):
        img = cv2.imread(filenames + "/" + path)
        hist_b = cv2.calcHist([img], [0], None, [256], [0,256])
        hist_g = cv2.calcHist([img], [1], None, [256], [0,256])
        hist_r = cv2.calcHist([img], [2], None, [256], [0,256])
        feature = np.concatenate((hist_b, np.concatenate((hist_g, hist_r), axis=None)), axis=None)

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
        hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hist_h = cv2.calcHist([hsv_image], [0], None, [256], [0,256])
        hist_s = cv2.calcHist([hsv_image], [1], None, [256], [0,256])
        hist_v = cv2.calcHist([hsv_image], [2], None, [256], [0,256])

        feature = np.concatenate((hist_v, np.concatenate((hist_s, hist_h), axis=None)), axis=None)

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
        old_img = cv2.imread(filenames + "/" + path)
        img = reduce_resolution(old_img, 400, 300)
        featureSum = 0
        sift = cv2.xfeatures2d.SIFT_create()
        #print(img.shape)
        kps , des = sift.detectAndCompute(img,None)

        num_image, _ = path.split(".")
        #print(len(kps), "len(kps)") 
        #print(len(des), "len(des)") 
        if(len(kps)) > 0:
            np.savetxt("../output/SIFT/" + str(num_image) + ".txt" ,des)
            progress_bar.setValue(100 * ((i + 1) / len(os.listdir(filenames))))
        
            featureSum += len(kps)
        else:
            print(filenames + "/" + path)
    
    progress_bar.setValue(0)
    print("Indexing sift done")

  

def generate_orb(filenames, progress_bar):
    """ Generate orb for each image in the folder """
    if not os.path.isdir("../output/ORB"):
        os.mkdir("../output/ORB")

    for i, path in enumerate(os.listdir(filenames)):
        img = cv2.imread(filenames + "/" + path)
        orb = cv2.ORB_create()
        key_point1, descriptor1 = orb.detectAndCompute(img, None)
        
        num_image, _ = path.split(".")
        if(descriptor1 is not None):
            np.savetxt("../output/ORB/" + str(num_image) + ".txt", descriptor1)
            progress_bar.setValue(100 * ((i + 1) / len(os.listdir(filenames))))
        else:
            print(filenames + "/" + path)

    progress_bar.setValue(0)
    print("Indexing orb done")


def generate_glcm(filenames, progress_bar):
    """ Generate glcm for each image in the folder """
    if not os.path.isdir("../output/GLCM"):
        os.mkdir("../output/GLCM")

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

""" J'ai essayé un truc mais pas sur que ce soit la bonne approche, je comprends pas trop comment on peut combiner
def global_generator(filenames, progress_bar, feature_methods):
    if not os.path.isdir("FEATURES"):
        os.mkdir("FEATURES")
        
    # Create a dictionary mapping descriptor names to feature extraction functions
    descriptor_functions = {
        "SIFT": cv2.SIFT_create,
        "ORB": cv2.ORB_create,
        # Ajouter les autres
    }
    
    for i, path in enumerate(os.listdir(filenames)):
        img = cv2.imread(filenames + "/" + path)
        num_image, _ = path.split(".")
        
        # Initialize an empty list to store all the features for this image
        features = []
        
        for descriptor in feature_methods:
            # Extract features using the appropriate function from the dictionary
            descriptor_function = descriptor_functions[descriptor]
            feature_extractor = descriptor_function()
            _, descriptor1 = feature_extractor.detectAndCompute(img, None)
            
            # Make sure all feature arrays have the same number of dimensions
            if len(descriptor1.shape) == 1:
                descriptor1 = descriptor1.reshape(-1, 1)
            
            # Add the features to the list
            features.extend(descriptor1)
        
        # Convert the list of features to a numpy array and save it to a file
        features = np.array(features)
        np.savetxt("FEATURES/" + str(num_image) + ".txt", features)
        
        progress_bar.setValue(100 * ((i + 1) / len(os.listdir(filenames))))
    
    print("Indexing features done")
"""