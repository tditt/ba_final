import numpy as np
from itertools import combinations as combos
import pairs_db as pdb
import util

centers = None
w_max = None
w_min = None
radii = None
r_rel_tol = None
database = {}


def extract_coordinates(boxes, wmax, wmin, rtol, intersect_mode):
    global centers, w_max, w_min, radii, r_rel_tol, database
    database = {}
    r_rel_tol = rtol
    w_max = wmax
    w_min = wmin
    n = boxes.shape[0]
    radii = np.zeros(n)
    center_points = np.zeros((n, 2))
    # approximate all radii in pixels
    for i, box in enumerate(boxes):
        radii[i] = util.calculate_radius(box)
        center_points[i] = util.calculate_center(box)
    centers = center_points
    # get sorting vector
    sort = radii.argsort()[::-1]
    # sort dat shit
    radii = radii[sort]
    tuple_generator = combos(np.arange(n), 2)
    for count, (ci, cj) in enumerate(tuple_generator):
        # get pair-database indices of pair candidates
        if intersect_mode:
            ci_candidates, cj_candidates = search_pair(ci, cj, True)
            add_to_db(ci, ci_candidates)
            add_to_db(cj, cj_candidates)
        else:
            ci_d, ci_r, cj_d, cj_r = search_pair(ci, cj, False)
            add_to_db(ci, ci_d)
            add_to_db(ci, ci_r)
            add_to_db(cj, cj_d)
            add_to_db(cj, cj_r)
    check_db()
    return do_final_validation()


def search_pair(ci, cj, intersect_mode=True):
    # ci_tol = util.calculate_dynamic_r_rel_tol(radii[ci], w_max, w_min)
    # cj_tol = util.calculate_dynamic_r_rel_tol(radii[cj], w_max, w_min)
    ci_tol = .6
    cj_tol = .9
    r_upper = radii[cj] * (1 + (r_rel_tol * cj_tol)) / (radii[ci] * (1 - (r_rel_tol * ci_tol)))
    r_lower = radii[cj] * (1 - (r_rel_tol * cj_tol)) / (radii[ci] * (1 + (r_rel_tol * ci_tol)))
    d = util.calculate_distance(centers[ci], centers[cj])
    d_upper = d / (radii[ci] * (1 - (r_rel_tol * ci_tol)))
    d_lower = d / (radii[ci] * (1 + (r_rel_tol * ci_tol)))
    pair_ids = pdb.search_d_rel(d_lower, d_upper)
    ci_candidates_d = pdb.get_ci_by_pair_ids(pair_ids)
    cj_candidates_d = pdb.get_cj_by_pair_ids(pair_ids)
    pair_ids = pdb.search_r_rel(r_lower, r_upper)
    ci_candidates_r = pdb.get_ci_by_pair_ids(pair_ids)
    cj_candidates_r = pdb.get_cj_by_pair_ids(pair_ids)
    if intersect_mode:
        ci_candidates = np.intersect1d(ci_candidates_d, ci_candidates_r)
        cj_candidates = np.intersect1d(cj_candidates_d, cj_candidates_r)
        return ci_candidates, cj_candidates
    return ci_candidates_d, ci_candidates_r, cj_candidates_d, cj_candidates_r


def do_final_validation():
    indices = []
    n = len(database)

    minimum_candidates = 9999999999
    for v in database.values():
        sort_by_most_votes = np.argsort(v)[::-1]
        indices.append(sort_by_most_votes)
        num_candidates = np.where(v != 0)[0].shape[0]
        if minimum_candidates > num_candidates: minimum_candidates = num_candidates
    tuple_generator = combos(np.arange(n), 2)
    tuples = []
    for t in tuple_generator: tuples.append(t)
    final_votes = np.zeros(n)
    for a, b in tuples[::-1]:
        final_votes = validate_pair(0, final_votes, indices, a, b)
    sort = np.argsort(final_votes)[::-1]
    results = {}
    for i in sort:
        crater_id = indices[i][0]
        coords = pdb.get_lunar_coordinates(indices[i][0])
        radius = pdb.get_real_radius(indices[i][0])
        votes = final_votes[i]
        results[crater_id] = {'coords': coords, 'radius': radius, 'votes': votes}
        print("crater ", i, ": ", coords, ' with ', votes, ' votes!')
    return results


def validate_pair(start, votes, indices, ci, cj):
    top_ci = indices[ci][start]
    top_cj = indices[cj][start]
    candidates_ci, candidates_cj = search_pair(ci, cj, intersect_mode=True)
    if top_ci in candidates_ci and top_cj in candidates_cj:
        votes[ci] += 1
        votes[cj] += 1
    return votes


def add_to_db(key, data):
    # add possible crater candidates to database if there are any.
    # this is done via intersection of new crater candidates with
    # already existing ones (if there is new ones and already
    # existing ones)
    entry = database.get(key, None)
    if entry is None:
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
            print(sort[i], ' (coords: ', pdb.get_lunar_coordinates(sort[i]), ' ) with ', v[sort[i]], ' votes')
    return found
