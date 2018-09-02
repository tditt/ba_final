from math import pi, sin
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.neighbors import DistanceMetric as DM

radius_moon = 1737.5
circumference_moon = 2 * pi * radius_moon


def calculate_pair_distances(origin_coords, target_coords, origin_radius):
    # origin_coords are supposed to be the coordinates of the bigger crater (relevant later for database structure)
    haversine_dist = calculate_haversine_distance(origin_coords, target_coords)
    approx_dist = approximate_visual_distance(haversine_dist)
    rel_dist = approx_dist / origin_radius
    return rel_dist, haversine_dist, approx_dist


def calculate_haversine_distance(origin_coords, target_coords):
    dist = DM.get_metric('haversine')
    origin_coords, target_coords = map(np.radians, [origin_coords, target_coords])
    return (radius_moon * dist.pairwise([origin_coords, target_coords]))[0][1]


def approximate_visual_distance(haversine_dist):
    theta = haversine_dist / radius_moon
    return 2 * radius_moon * sin(theta / 2)


def calculate_radius(box):
    x1, y1 = box[0], box[1]
    x2, y2 = box[2], box[3]
    return (x2 - x1 + y2 - y1) / 4


def calculate_center(box):
    x1, y1 = box[0], box[1]
    x2, y2 = box[2], box[3]
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    return center_x, center_y


def calculate_distance(point_a, point_b, correction_factor=.97):
    return correction_factor * cdist(np.array([point_a]), np.array([point_b]))[0][0]


def calculate_dynamic_r_rel_tol(r, w_max, w_min):
    tol = 1 + ((w_max / r) ** 2 / (w_max / w_min) ** 1.8)
    return tol


# progress bar source code taken https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
def print_progress(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
    # Print New Line on Complete
    if iteration == total:
        print()


def intersect_arrays(arr1, arr2):
    # returns intersections of both arrays. If one of them is empty,
    # returns the other one. If both are empty, returns an empty list
    if isinstance(arr1, list): arr1 = np.array(arr1)
    if isinstance(arr2, list): arr2 = np.array(arr2)
    if arr1.shape[0] == 0:
        if arr2.shape[0] == 0:
            print("both arrays empty, returning empty")
            return []
        print("arr1 empty, returning arr2")
        return arr2
    if arr2.shape[0] == 0:
        print("arr2 empty, returning arr1")
        return arr1
    return np.intersect1d(arr1, arr2)
