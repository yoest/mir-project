from skimage.transform import resize
import cv2
import numpy as np
from skimage.feature import hog, greycomatrix, graycoprops, local_binary_pattern, greycoprops
import utils


def compute_combined_descriptor(img, algo_choice):
    """ Compute the descriptor of an image which is the combination of multiple descriptor

    Args:
        img (np.array): image to compute the descriptor
        algo_choice (str): name of the algorithm to use

    Returns:
        np.array: descriptor of the image
    """
    dec_sift, dec_orb = None, None

    feature_methods = algo_choice.split("_")
    features = []
    
    if "BGR" in feature_methods:
        color_hist = utils.compute_colors(img)
        features.append(color_hist)
    if "HSV" in feature_methods:
        hsv_hist = utils.compute_hsv(img)
        features.append(hsv_hist)
    if "GLCM" in feature_methods:
        glcm = utils.compute_glcm(img)
        features.append(glcm)
    if "LBP" in feature_methods:
        lbp = utils.compute_lbp(img)
        features.append(lbp)
    if "HOG" in feature_methods:
        hog = utils.compute_hog(img)
        features.append(hog)
    if "SIFT" in feature_methods:
        kps_sift, dec_sift = utils.compute_sift(img)
        if dec_sift is not None:
            features.append(dec_sift)
    if "ORB" in feature_methods:
        kps_orb, dec_orb = utils.compute_orb(img)
        if dec_orb is not None:
            features.append(dec_orb)
    
    return np.concatenate(features, axis=None)



def extract_req_features(filename, algo_choice):  
    """ Extract features from an image in a file an chose a specific algorithm

    Args:
        filename (str): filename of the image
        algo_choice (str): name of the algorithm to use

    Raises:
        ValueError: if the algorithm name does not exists

    Returns:
        vect_features (np.array): features vector
    """
    if filename: 
        img = cv2.imread(filename)
        # resized_img = resize(img, (128 * 4, 64 * 4))
            
        # -- Colors --
        if algo_choice == 'BGR': 
            vect_features = utils.compute_colors(img)
        
        # -- HSV --
        elif algo_choice == 'HSV':
            vect_features = utils.compute_hsv(img)

        # -- SIFT --
        elif algo_choice == 'SIFT':
            key_points, vect_features = utils.compute_sift(img)
    
        # -- ORB --
        elif algo_choice == 'ORB':
            key_points, vect_features = utils.compute_orb(img)

        # -- GLCM --
        elif algo_choice == 'GLCM': 
            vect_features = utils.compute_glcm(img)

        # -- LBP --
        elif algo_choice == 'LBP':
            vect_features = utils.compute_lbp(img)

        # -- HOG --
        elif algo_choice == 'HOG':
            vect_features = utils.compute_hog(img)

        # -- Combined descriptor --
        elif '_' in algo_choice:
            vect_features = compute_combined_descriptor(img, algo_choice)

        else:
            raise ValueError("Invalid algorithm choice: " + str(algo_choice))
			
        np.savetxt("../search_output/method_" + str(algo_choice) + "_request.txt", vect_features)
        print("saved")

        return vect_features