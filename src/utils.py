import os
import time
import cv2
import numpy as np
from skimage.transform import resize
import cv2
import numpy as np
from skimage.feature import hog, greycomatrix, graycoprops, local_binary_pattern, greycoprops


def compute_execution_time(fct, args):
    """ Compute the execution time of a function. """
    start = time.time()
    fct(*args)
    end = time.time()
    print(f'Time taken (in seconds): {end - start}')

def compute_feature_methods_name(feature_methods):
    res = ""
    for method in feature_methods:
        res = res + "_" + method 
    return res 

def compute_file_size(filename):
    """ Compute the size of a file given by its name. """
    file_stats = os.stat(filename)

    print(f'File Size in Bytes is {file_stats.st_size}')
    print(f'File Size in MegaBytes is {file_stats.st_size / (1024 * 1024)}')


def compute_colors(img):
    """ Compute color vectors features for an image """
    hist_b = cv2.calcHist([img], [0], None, [256], [0,256])
    hist_g = cv2.calcHist([img], [1], None, [256], [0,256])
    hist_r = cv2.calcHist([img], [2], None, [256], [0,256])
    return np.concatenate((hist_b, np.concatenate((hist_g, hist_r), axis=None)), axis=None)

def compute_hsv(img):
    """ Compute hsv vectors features for an image """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [256], [0,256])
    hist_s = cv2.calcHist([hsv], [1], None, [256], [0,256])
    hist_v = cv2.calcHist([hsv], [2], None, [256], [0,256])
    return np.concatenate((hist_h, np.concatenate((hist_s, hist_v), axis=None)), axis=None)

def compute_glcm(img):
    """ Compute GLCM features for an image """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image_glcm = greycomatrix(img, distances=[1], angles=[0, np.pi / 4, -np.pi / 2], symmetric=True, normed=True)
    return greycoprops(image_glcm, 'contrast')


def compute_lbp(image):
    """ Compute LBP features for an image """
    def cal_hist_gray(image):
        image = image.astype(np.uint8)
        hist = cv2.calcHist(image, [0], None, [256], [0, 256])
        hist = cv2.normalize(hist, hist)
        return hist

    def lbp_descriptor(image):             
        METHOD = 'uniform'
        radius = 3
        n_points = 8 * radius
        gray = cal_hist_gray(image)
        lbp = local_binary_pattern(gray, n_points, radius, METHOD)
        return lbp

    def cal_lbp(image):                   
        if image is not None:
            des = lbp_descriptor(image)
            lbp_features = des.tolist()
        lbp_features = np.array(lbp_features)
        return lbp_features

    #image = cv2.imread(filename, 3)
    return cal_lbp(image)

def compute_hog(image):
    """ Compute HOG features for an image """
    def hog_desc(image):
        resized_img = resize(image, (128*4, 64*4))
        fd, hog_image = hog(resized_img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True, multichannel=True)
        return hog_image

    #image = cv2.imread(filename, 3)
    return hog_desc(image)

def reduce_resolution(img, new_width, new_height):

    # Réduire la résolution de l'image
    red_img = cv2.resize(img, (new_width, new_height))
    return red_img

def compute_sift(old_img):
        #old_img = cv2.imread(filename)
        img = reduce_resolution(old_img, 200, 150)
        sift = cv2.xfeatures2d.SIFT_create()
        kps, des = sift.detectAndCompute(img,None)
        
        return kps, des
            
def compute_orb(img):
    orb = cv2.ORB_create()
    key_point1, descriptor1 = orb.detectAndCompute(img, None)
    return key_point1, descriptor1    
    


       
