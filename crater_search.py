import k_vector
import h5py
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from itertools import combinations as combos
from math import sqrt, factorial


# parameters
tolerance_radius = .18
tolerance_distance = .08

minimum_size_difference = 1.05
w_min = 16
w_max = 816
max_rel_distance = dist_precision_factor * 2 * sqrt(2) * (w_max / w_min - w_min / 2)
min_rel_radius = size_precision_factor * w_min / w_max


def extract_coordinates(boxes, scores):
    radii = np.zeros(boxes.shape[0])
    # approximate all radii in pixels
    for i, box in enumerate(boxes):
        radii[i] = calculate_radius(box)
    # get sorting vector
    sort = radii.argsort()[::-1]
    # sort dat shit
    radii = radii[sort]
    boxes = boxes[sort]
    scores = scores[sort]
    find_candidates(boxes, radii)


def find_candidates(boxes, radii):
    n = boxes.shape[0]
    database = check_tuples(boxes, radii)
    min_candidates = 9999999
    avg = 0
    for k, v in database.items():
        size = v.shape[0]
        avg += size
        if size == 0:
            print('warning: zero candidates found for crater ', str(k))
        if size < min_candidates: min_candidates = size
    if n != 0:  avg /= n
    print('finish, looped through all combinations! lowest number of potential candidates: ', str(min_candidates),
          ', average: ',
          str(avg))


def check_tuples(boxes, radii):
    n = boxes.shape[0]
    # combos_n = factorial(n) / (2 * factorial(n - 2))
    tuple_generator = combos(np.arange(n), 2)
    database = {}
    for count, (a, b) in enumerate(tuple_generator):
        # get pair-database indices of pair candidates
        crater_a_candidates, crater_b_candidates = check_single_tuple([boxes[a], boxes[b]], [radii[a], radii[b]])
        add_to_db(database, a, crater_a_candidates)
        add_to_db(database, b, crater_b_candidates)
        print(str(count))
        if count % 10 == 0:
            check_db(database)
            if count >= 850:
                print('maximum reached, cancelling loop')
                break

    check_db(database)
    return database


def check_single_tuple(boxes, radii):
    if len(boxes) != 2 or len(radii) != 2:
        print('error: need two boxes and radii as input')
        return
    if radii[0] < radii[1]: print('error : radius sorting is wrong')
    size, dist = calculate_size_and_distance(boxes[0], boxes[1], radii[0], radii[1])
    pair_candidates = get_pair_candidates(size, dist)
    # if radii[0] < radii[1] * minimum_size
    #
    #_difference:
    #     print('radii almost the same, swapping pair...')
    #     size, dist = calculate_size_and_distance(boxes[1], boxes[0], 1, 1)
    #     pair_candidates = np.union1d(get_pair_candidates(size, dist), pair_candidates)
    return index[..., 0][pair_candidates], index[..., 1][pair_candidates]


def check_triplets(boxes, radii):
    n = boxes.shape[0]
    triplet_generator = combos(np.arange(n), 3)
    database = {}
    for count, (one, two, three) in enumerate(triplet_generator):
        triplet_boxes = [boxes[one], boxes[two], boxes[three]]
        triplet_radii = [radii[one], radii[two], radii[three]]
        results = check_single_triplet(triplet_boxes, triplet_radii)
        add_to_db(database, one, results[0])
        add_to_db(database, two, results[1])
        add_to_db(database, three, results[2])
        check_db(database)
        if (count >= 150):
            print('maximum reached, cancelling loop')
            break
    return database


def add_to_db(database, key, data):
    # add possible crater candidates to database if there are any.
    # this is done via intersection of new crater candidates with
    # already existing ones (if there is new ones and already
    # existing ones)
    entry = database.get(key, None)
    if entry is None or len(entry) == 0:
        database[key] = data
    elif data is not None:
        database[key] = np.intersect1d(entry, data)


def check_db(database):
    found = False
    for k, v in database.items():
        if v.shape[0] <= 4:
            for crater in v:
                print('Potential candidate out of ', str(v.shape[0]), ' altogether: ',
                      str(get_lunar_coordinates(crater)))
            found = True
    return found


def get_pair_candidates(size, dist):
    if size < min_rel_radius: print('size too small: ', str(size))
    if dist > max_rel_distance: print('distance too large: ', str(dist))
    results_size = search_size(size)
    results_dist = search_dist(dist)
    if results_size is None:
        if results_dist is None:
            return []
        return results_dist
    if results_dist is None:
        if results_size is None:
            return []
        return results_size
    intersection = np.intersect1d(results_dist, results_size)
    if intersection is None: return []
    return intersection