import numpy as np
import math
import cv2
import numpy as np
import operator



def compute_euclidean_dist(v1, v2):
    """ Compute the euclidean distance between two vectors. """
    a = np.array(v1)
    b = np.array(v2)
    return np.linalg.norm(a - b)


def compute_chi_square_distance(v1, v2):
    """ Compute the chi square distance between two vectors. """
    s = 0.0
    for i, j in zip(v1, v2):
        if i == j == 0.0:
            continue
        s += (i - j)**2 / (i + j)
    return s


def compute_bhatta_dist(v1, v2):
    """ Compute the Bhattacharyya distance between two vectors. """
    v1 = np.array(v1)
    v2 = np.array(v2)

    num = np.sum(np.sqrt(np.multiply(v1, v2, dtype=np.float64)), dtype=np.float64)
    den = np.sqrt(np.sum(v1, dtype=np.float64) * np.sum(v2, dtype=np.float64))
    return math.sqrt(1 - num / den)


def compute_flann_dist(v1, v2):
    """ Compute the Flann distance between two vectors. """
    v1 = np.float32(np.array(v1))
    v2 = np.float32(np.array(v2))

    if v1.shape[0] == 0 or v2.shape[0] == 0:
        return np.inf

    index_params = dict(algorithm=1, trees=5)
    sch_params = dict(checks=50)
    flannMatcher = cv2.FlannBasedMatcher(index_params, sch_params)
    matches = list(map(lambda x: x.distance, flannMatcher.match(v1, v2)))
    return np.mean(matches)


def compute_brute_force_matching(v1, v2):
    """ Compute the Brute Force Matching algorithm between two vectors. """
    v1 = np.array(v1).astype('uint8')
    v2 = np.array(v2).astype('uint8')

    if v1.shape[0] == 0 or v2.shape[0] == 0:
        return np.inf

    # This exeception is handled for some case of sift
    try:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = list(map(lambda x: x.distance, bf.match(v1, v2)))
    except:
        return np.inf

    return np.mean(matches)


def compute_distance_with_name(v1, v2, distance_name):
    """ Compute the distance between two vectors based on a given distance name. If the distance name is not found, an exception is raised. """
    if distance_name == "Euclidienne":
        distance = compute_euclidean_dist(v1, v2)
    elif distance_name == "Correlation":
        methode = cv2.HISTCMP_CORREL
        distance = cv2.compareHist(np.float32(v1), np.float32(v2), methode)
    elif distance_name == "Chi carre":
        distance = compute_chi_square_distance(v1, v2)
    elif distance_name == "Intersection":
        methode = cv2.HISTCMP_INTERSECT
        distance = cv2.compareHist(np.float32(v1), np.float32(v2), methode)
    elif distance_name == "Bhattacharyya":
        distance = compute_bhatta_dist(v1, v2)    
    elif distance_name == "Brute force":
        distance = compute_brute_force_matching(v1, v2)
    elif distance_name == "Flann":
        distance = compute_flann_dist(v1, v2)
    else:
        raise Exception("Distance name not found.")

    return distance


def get_k_voisins(lfeatures, req, k, distance_name):
    """ Get the k nearest neighbors of a given request.

    Args:
        lfeatures (list): list of features for each image in the database.
        req: feature extracted from the request image.
        k (int): number of neighbors to return.
        distance_name (str): name of the distance to use.

    Returns:
        list: list of k nearest neighbors.
    """
    ldistances = [] 
    for i in range(len(lfeatures)): 
        dist = compute_distance_with_name(req, lfeatures[i][1], distance_name)
        ldistances.append((lfeatures[i][0], lfeatures[i][1], dist)) 

    ordre = distance_name in ["Correlation", "Intersection"]
    ldistances.sort(key=operator.itemgetter(2), reverse=ordre) 

    lneighboor = [] 
    for i in range(k): 
        lneighboor.append(ldistances[i]) 
    return lneighboor