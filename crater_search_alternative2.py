import k_vector
import h5py
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from itertools import combinations as combos
from itertools import combinations_with_replacement as combos_repl

from math import sqrt, factorial

# load the whole database directly into RAM as numpy arrays
db_basic_path = 'crater_database/'
file_name_k_radius = 'pairs_data_radius_k'
file_name_k_distance = 'pairs_data_distance_k'
file_name_qm_radius = 'pairs_data_radius_qm'
file_name_qm_distance = 'pairs_data_distance_qm'
file_name_index = 'pairs_index'
file_name_data = 'pairs_data'
hdf = pd.HDFStore(db_basic_path + 'crater_db.h5', 'r')
crater_db = hdf.get('/db')

# k_index_big = h5py.File(db_basic_path + 'pairs_index_big_k.h5', 'r')['pairs_index_big_k']
# k_index_small = h5py.File(db_basic_path + 'pairs_index_small_k.h5', 'r')['pairs_index_small_k']
print('loading database into RAM...')
pairs_len = h5py.File(db_basic_path + 'pairs_data_radius_k.h5', 'r')['pairs_data_radius_k'][0].shape[0]
k_vector_radius = np.zeros(pairs_len, dtype=np.uint32)
k_sort_radius = np.zeros(pairs_len, dtype=np.uint32)
qm_radius = np.zeros(2, dtype=np.float64)

k_vector_distance = np.zeros(pairs_len, dtype=np.uint32)
k_sort_distance = np.zeros(pairs_len, dtype=np.uint32)
qm_distance = np.zeros(2, dtype=np.float64)
index = np.zeros((pairs_len, 2), dtype=np.uint32)
pairs = np.zeros((pairs_len, 3), dtype=np.uint16)


# h5py.File(db_basic_path + 'pairs_data_radius_k.h5', 'r')['pairs_data_radius_k'].read_direct(k_vector_radius, np.s_[0],
#                                                                                             np.s_)
# h5py.File(db_basic_path + 'pairs_data_radius_k.h5', 'r')['pairs_data_radius_k'].read_direct(k_sort_radius, np.s_[1],
#                                                                                             np.s_)
# h5py.File(db_basic_path + 'pairs_data_distance_k.h5', 'r')['pairs_data_distance_k'].read_direct(k_vector_distance,
#                                                                                                 np.s_[0],
#                                                                                                 np.s_)
# h5py.File(db_basic_path + 'pairs_data_distance_k.h5', 'r')['pairs_data_distance_k'].read_direct(k_sort_distance,
#                                                                                                 np.s_[1],
#                                                                                                 np.s_)
#
# h5py.File(db_basic_path + 'pairs_data_radius_qm.h5', 'r')['pairs_data_radius_qm'].read_direct(qm_radius, np.s_,
#                                                                                               np.s_)
# h5py.File(db_basic_path + 'pairs_data_distance_qm.h5', 'r')['pairs_data_distance_qm'].read_direct(qm_distance, np.s_,
#                                                                                                   np.s_)

#
# h5py.File(db_basic_path + 'pairs_index.h5', 'r')['pairs_index'].read_direct(index, np.s_,
#                                                                             np.s_)
# h5py.File(db_basic_path + 'pairs_data.h5', 'r')['pairs_data'].read_direct(pairs, np.s_,
#                                                                           np.s_)
#

# k_sort_radius = h5py.File(db_basic_path + 'pairs_data_radius_k.h5', 'r')['pairs_data_radius_k'][1]
# q_radius = h5py.File(db_basic_path + 'pairs_data_radius_qm.h5', 'r')['pairs_data_radius_qm'][0]
# m_radius = h5py.File(db_basic_path + 'pairs_data_radius_qm.h5', 'r')['pairs_data_radius_qm'][1]
# k_vector_distance = np.array(h5py.File(db_basic_path + 'pairs_data_distance_k.h5', 'r')['pairs_data_distance_k'][0])
# k_sort_distance = np.array(h5py.File(db_basic_path + 'pairs_data_distance_k.h5', 'r')['pairs_data_distance_k'][1])
# q_distance = h5py.File(db_basic_path + 'pairs_data_distance_qm.h5', 'r')['pairs_data_distance_qm'][0]
# m_distance = h5py.File(db_basic_path + 'pairs_data_distance_qm.h5', 'r')['pairs_data_distance_qm'][1]
def read_hdf_to_ram(index, target_arr, filename):
    datafile = h5py.File(db_basic_path + filename + '.h5', 'r')
    if index is None:
        datafile[filename].read_direct(target_arr)
    else:
        datafile[filename].read_direct(target_arr, np.s_[index])
    datafile.close()


read_hdf_to_ram(0, k_vector_radius, file_name_k_radius)
read_hdf_to_ram(1, k_sort_radius, file_name_k_radius)
read_hdf_to_ram(0, k_vector_distance, file_name_k_distance)
read_hdf_to_ram(1, k_sort_distance, file_name_k_distance)
read_hdf_to_ram(None, qm_radius, file_name_qm_radius)
read_hdf_to_ram(None, qm_distance, file_name_qm_distance)
read_hdf_to_ram(None, index, file_name_index)
read_hdf_to_ram(None, pairs, file_name_data)
q_radius = qm_radius[0]
m_radius = qm_radius[1]
q_distance = qm_distance[0]
m_distance = qm_distance[1]
print('... done!')

# parameters
tolerance_radius = .07
tolerance_distance = .2
size_precision_factor = 5000
dist_precision_factor = 500
minimum_size_difference = 1.05
w_min = 16
w_max = 816
radius_fact = w_max / w_min
max_rel_distance = dist_precision_factor * 2 * sqrt(2) * (w_max / w_min - w_min / 2)
min_rel_radius = size_precision_factor * w_min / w_max
database = {}
search_results = {}


# TODO: make tolerances scale inversely with size?
# TODO: use large tolerance if bounding box starts or ends at picture edge

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
    check_tuples(boxes, radii)
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
    for count, (a, b) in enumerate(tuple_generator):
        # get pair-database indices of pair candidates
        crater_a_candidates, crater_b_candidates = check_single_tuple([boxes[a], boxes[b]], [radii[a], radii[b]])
        add_to_db(a, crater_a_candidates)
        add_to_db(b, crater_b_candidates)
        search_results[(a, b)] = np.array(list(zip(crater_a_candidates, crater_b_candidates)))
        if count % 50 == 0:
            check_db()
            print(str(count))
        if count >= 850:
            print('maximum reached, cancelling loop')
            break
    final_votes = np.zeros(n)
    indices = None
    start = 0
    while final_votes.argmax(axis=0) < 4:
        final_votes, indices = do_final_validation(start, np.zeros(n), indices)
        start += 1
    print("found something!")
    for i in indices:
        print("crater ", i, ": ", get_lunar_coordinates(indices[i][start]))
    check_db()


def check_single_tuple(boxes, radii):
    if len(boxes) != 2 or len(radii) != 2:
        print('error: need two boxes and radii as input')
        return
    if radii[0] < radii[1]: print('error : radius sorting is wrong')
    r, r_lower, r_upper, d, d_lower, d_upper = calculate_size_and_distance(boxes[0], boxes[1], radii[0], radii[1])
    pair_candidates = get_pair_candidates(r, r_lower, r_upper, d, d_lower, d_upper)
    # if radii[0] < radii[1] * minimum_size
    #
    # _difference:
    #     print('radii almost the same, swapping pair...')
    #     size, dist = calculate_size_and_distance(boxes[1], boxes[0], 1, 1)
    #     pair_candidates = np.union1d(get_pair_candidates(size, dist), pair_candidates)
    return index[..., 0][pair_candidates], index[..., 1][pair_candidates]


def do_final_validation(start, votes, indices):
    if indices is None:
        indices = []
    n = votes.shape[0]
    for v in database.values():
        sort_by_most_votes = np.argsort(v)[::-1]
        highest_vote = v[sort_by_most_votes][0] - 3
        candidates_with_highest_vote = []
        for i in range(sort_by_most_votes.shape[0]):
            if highest_vote > v[sort_by_most_votes][i]:
                candidates_with_highest_vote = sort_by_most_votes[:i]
                break
        indices.append(candidates_with_highest_vote)

    # combos_n = factorial(n) / (2 * factorial(n - 2))
    tuple_generator = combos(np.arange(n), 2)
    tuples = []
    for t in tuple_generator: tuples.append(t)
    votes = np.zeros(n)
    for a, b in tuples[::-1]:
        votes = check_pair(start, votes, indices, a, b)
    return votes, indices


def check_pair(start, votes, indices, a, b):
    top_a = indices[a][start]
    top_b = indices[b][start]
    if (top_a, top_b) in search_results[(a, b)]:
        votes[a] += 1
        votes[b] += 1
    # n_a = indices[a].shape[0]
    # n_b = indices[b].shape[0]
    # tuple_generator = combos_repl(np.arange(n_a), 2)
    # if n_a < n_b:
    #     tuple_generator = combos_repl(np.arange(n_b), 2)
    # for i, j in tuple_generator:
    #     potential_a = indices[a][i]
    #     potential_b = indices[b][j]
    #     if (potential_a, potential_b) in search_results[(a, b)]:
    #         database[a] = add_votes(database[a], potential_a)
    #         database[b] = add_votes(database[b], potential_b)
    return votes


def check_triplets(boxes, radii):
    n = boxes.shape[0]
    triplet_generator = combos(np.arange(n), 3)
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


def check_single_triplet(boxes, radii):
    # outputs three numpy arrays containing possible IDs for the triplet elements in same order of input
    pair_generator = combos(np.arange(3), 2)
    collect = []
    intersections = []
    # Three pairs: 0-1, 0-2, 1-2
    for i, j in pair_generator:
        if radii[i] < radii[j]: print('error : radius sorting is wrong')
        if radii[i] < radii[j] * minimum_size_difference:
            print('radii too close together, ignoring pair')

        size, dist = calculate_size_and_distance(boxes[i], boxes[j], radii[i], radii[j])
        collect.append(get_pair_candidates(size, dist))

    # print('0 :', collect[0].shape[0], '1 :', collect[1].shape[0], '2 :', collect[2].shape[0])
    # try to match the first indices of 0-1 and 0-2
    # element 0 is contained in this
    intersections.append(intersect_arrays(index[..., 0][collect[0]], index[..., 0][collect[1]]))
    # now update everything that contains element 0 accordingly
    # collect[0] = np.intersect1d(collect[0], search_big_craters(intersect[0]), assume_unique=True)
    # collect[1] = np.intersect1d(collect[1], search_big_craters(intersect[0]), assume_unique=True)

    # try to match the 1-indices of 0-1 1-2
    # element 1 is contained in this
    intersections.append(intersect_arrays(index[..., 1][collect[0]], index[..., 0][collect[2]]))
    # now update everything that contains element 1 accordingly
    # collect[0] = np.intersect1d(collect[0], search_small_craters(intersect[1]), assume_unique=True)
    # collect[2] = np.intersect1d(collect[2], search_big_craters(intersect[1]), assume_unique=True)

    # try to match the second indices of 0-2 1-2
    # element 2 is contained in this
    intersections.append(intersect_arrays(index[..., 1][collect[1]], index[..., 1][collect[2]]))
    # now update everything that contains element 2 accordingly
    # collect[1] = np.intersect1d(collect[1], search_small_craters(intersect[2]), assume_unique=True)
    # collect[2] = np.intersect1d(collect[2], search_small_craters(intersect[2]), assume_unique=True)
    # print('0 :', collect[0].shape[0], '1 :', collect[1].shape[0], '2 :', collect[2].shape[0])
    # get sort vector for beginning with lowest element intersect
    return intersections


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
    # n = boxes.shape[0]
    # pair_generator = combos(np.arange(n), 2)
    # previous_big = 0
    # potential_candidates = []
    # candidate_maps = {}
    # for big, small in pair_generator:
    #     pair_indices = {}
    #     if big == previous_big:
    #         size, dist = calculate_size_and_distance(boxes[big], boxes[small], radii[big], radii[small])
    #         pair_indices[big, small] = get_pair_candidates(size, dist)
    #     else:
    #         previous_big = big
    #         intersection = None
    #         for i, index_arr in enumerate(pair_indices.values()):
    #             if index_arr is None: continue
    #             if intersection is None:
    #                 intersection = index[..., 0][index_arr]
    #             else:
    #                 intersection = np.intersect1d(intersection, index[..., 0][index_arr])
    #         potential_candidates.append(intersection)
    #         candidate_maps.update(pair_indices)
    #         pair_indices.clear()
    #         print('###### found ', str(intersection.shape[0]), ' potential candidates for crater ', str(big))
    #
    # length = np.zeros(len(potential_candidates))
    # for i, c in enumerate(potential_candidates):
    #     length[i] = c.shape[0]
    # # sort by least potential candidates
    # sort = np.argsort(length)
    # potential_others = []
    # for count, s in enumerate(sort):
    #     others = []
    #     for i, crater_index in enumerate(potential_candidates[s]):
    #         others.append(index[..., 1][search_big_crater(crater_index)])
    #     if len(potential_others) > 1:
    #         for x in range(0, count):
    #             for o in others:
    #                 for y in potential_others[x]:
    #                     o = np.intersect1d(o, y)
    #     potential_others.append(others)
    # print('bdvsfd')


def add_to_db(key, data):
    # add possible crater candidates to database if there are any.
    # this is done via intersection of new crater candidates with
    # already existing ones (if there is new ones and already
    # existing ones)
    entry = database.get(key, None)
    if entry is None or len(entry) == 0:
        votes = add_votes(np.zeros(78287, dtype=np.uint8), data)
        database[key] = votes
    elif data is not None:
        database[key] = add_votes(database[key], data)


def add_votes(target, new):
    target[new] += 1
    return target


def check_db():
    found = False
    for k, v in database.items():
        sort = np.argsort(v)[::-1]
        print('highest probability IDs for crater ', k, ' are:')
        for i in range(5):
            print(sort[i], ' (coords: ', get_lunar_coordinates(sort[i]), ' ) with ', v[sort[i]], ' votes')
    return found


def get_pair_candidates(r, r_lower, r_upper, d, d_lower, d_upper):
    # if r < min_rel_radius: print('relative radius too small: ', str(r))
    # if d > max_rel_distance: print('relative distance too large: ', str(d))
    results_size = search_radius(r, r_lower, r_upper)
    results_dist = search_dist(d, d_lower, d_upper)
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


def get_lunar_coordinates(crater):
    return [crater_db.loc[crater_db.index[crater], 'lat'], crater_db.loc[crater_db.index[crater], 'lon']]


def search_radius(r, r_lower, r_upper):
    y_a = r_lower
    y_b = r_upper
    # if y_a < min_rel_radius:
    #     print('minimum relative radius exceeded:  ', str(y_a / r))
    #     y_a = min_rel_radius
    # # maximum relative size is always below, 1.0 meaning values in database don't exceed 1*size_precision_factor
    # if y_b > size_precision_factor:
    #     print('maximum relative radius exceeded:  ', str(y_b / r))
    #     y_b = size_precision_factor
    return k_vector.k_vector_search(True, y_a, y_b, pairs[..., 0], k_vector_radius, k_sort_radius, q_radius,
                                    m_radius)


def search_dist(d, d_lower, d_upper):
    y_a = d_lower
    y_b = d_upper
    # if y_b > max_rel_distance:
    #     print('maximum relative distance exceeded:  ', str(y_b / d))
    #     y_b = max_rel_distance
    return k_vector.k_vector_search(True, y_a, y_b, pairs[..., 1], k_vector_distance, k_sort_distance, q_distance,
                                    m_distance)


def calculate_size_and_distance(big_box, small_box, big_radius, small_radius):
    rel_radius = small_radius / big_radius
    tol_small = calculate_radius_tol(small_radius)
    tol_big = calculate_radius_tol(big_radius)
    # calculate upper and lower rel_radius bounds
    rel_radius_upper = small_radius * (1 + tolerance_radius * tol_small) / (
            big_radius * (1 - tolerance_radius * tol_big))
    rel_radius_lower = small_radius * (1 - tolerance_radius * tol_small) / (
            big_radius * (1 + tolerance_radius * tol_big))
    big_box_center = calculate_center(big_box)
    small_box_center = calculate_center(small_box)
    # print('big center : ', str(big_box_center), ' small center : ', str(small_box_center))

    # calculate upper and lower rel_distance bounds
    rel_distance = cdist(np.array([big_box_center]), np.array([small_box_center]))[0][0]
    rel_distance_upper = rel_distance / (big_radius * (1 - tolerance_radius))
    rel_distance_lower = rel_distance / (big_radius * (1 + tolerance_radius))
    rel_distance /= big_radius
    rel_radius *= size_precision_factor
    rel_radius_upper *= size_precision_factor
    rel_radius_lower *= size_precision_factor
    rel_distance *= dist_precision_factor
    rel_distance_upper *= dist_precision_factor
    rel_distance_lower *= dist_precision_factor
    # print('relative_size: ', str(rel_radius), ' relative_distance: ', str(rel_distance))
    return rel_radius, rel_radius_lower, rel_radius_upper, rel_distance, rel_distance_lower, rel_distance_upper


def calculate_radius_tol(r):
    tol = 1 + ((w_max / r)**2 / radius_fact**1.8)
    return tol


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

# def search_big_craters(big):
#     union = None
#     for b in big:
#         if union is not None:
#             union = np.union1d(union, search_big_crater(b))
#         else:
#             union = search_big_crater(b)
#     return union
#
#
# def search_small_craters(small):
#     union = None
#     for s in small:
#         if union is not None:
#             union = np.union1d(union, search_small_crater(s))
#         else:
#             union = search_big_crater(s)
#     return union
#
#
# def search_big_crater(big):
#     y_a = big - 0.5
#     y_b = big + 0.5
#     return k_vector.k_vector_search(True, y_a, y_b, index[..., 0], k_integers[0, ...], k_integers[1, ...], q_big,
#                                     m_big)
#
#
# def search_small_crater(small):
#     y_a = small - 0.5
#     y_b = small + 0.5
#     return k_vector.k_vector_search(True, y_a, y_b, index[..., 1], k_integers[2, ...], k_integers[3, ...], q_small,
#                                     m_small)
#
