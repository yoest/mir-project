from skimage.transform import resize
import cv2
import numpy as np
from skimage.feature import hog, greycomatrix, graycoprops, local_binary_pattern, greycoprops



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


def compute_glcm(filename):
    """ Compute GLCM features for an image """
    img = cv2.imread(filename, 3)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image_glcm = greycomatrix(img, distances=[1], angles=[0, np.pi / 4, -np.pi / 2], symmetric=True, normed=True)
    return greycoprops(image_glcm, 'contrast')


def compute_lbp(filename):
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

    image = cv2.imread(filename, 3)
    return cal_lbp(image)


def compute_hog(filename):
    """ Compute HOG features for an image """
    def hog_desc(image):
        resized_img = resize(image, (128*4, 64*4))
        fd, hog_image = hog(resized_img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True, multichannel=True)
        return hog_image

    image = cv2.imread(filename, 3)
    return hog_desc(image)



def extract_req_features(filename, algo_choice):  
    """ Extract features from an image in a file an chose a specific algorithm """
    if filename: 
        img = cv2.imread(filename)
        resized_img = resize(img, (128 * 4, 64 * 4))
            
        # -- Colors --
        if algo_choice == 1: 
            vect_features = compute_colors(img)
        
        # -- HSV --
        elif algo_choice == 2:
            vect_features = compute_hsv(img)

        # -- SIFT --
        elif algo_choice == 3:
            sift = cv2.SIFT_create() 
            key_points, vect_features = sift.detectAndCompute(img, None)
    
        # -- ORB --
        elif algo_choice == 4:
            orb = cv2.ORB_create()
            key_points, vect_features = orb.detectAndCompute(img, None)

        # -- GLCM --
        elif algo_choice == 5: 
            vect_features = compute_glcm(filename)

        # -- LBP --
        elif algo_choice == 6:
            vect_features = compute_lbp(filename)

        # -- HOG --
        elif algo_choice == 7:
            vect_features = compute_hog(filename)

        else:
            raise ValueError("Invalid algorithm choice: " + str(algo_choice))
			
        np.savetxt("method_" + str(algo_choice) + "_request.txt", vect_features)
        print("saved")

        return vect_features