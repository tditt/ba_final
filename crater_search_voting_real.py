import numpy as np
from itertools import combinations as combos
import pairs_db as pdb
import util

dtol = 0
rtol = 0
database = {}
debug_mode = True
radius_candidates = {}
centers = None


def extract_coordinates(boxes, kmpp, r_tol, d_tol):
    global radius_candidates
    global dtol
    global rtol
    global centers
    global database
    database = {}
    dtol = d_tol
    rtol = r_tol
    n = boxes.shape[0]
    radii = np.zeros(n)
    center_points = np.zeros((n, 2))

    # calculate all radii and center points
    for i, box in enumerate(boxes):
        radii[i] = util.calculate_radius(box) * kmpp
        center_points[i] = util.calculate_center(box)
    # get sorting vector
    sort = radii.argsort()[::-1]
    # sort dat shit
    radii = radii[sort]
    center_points = center_points[sort]
    centers = center_points
    # get the radius candidates
    for c, r in enumerate(radii): radius_candidates[c] = pdb.get_craters_by_real_radius(r, r_tol)
    tuples = list(combos(np.arange(n), 2))
    t = len(tuples)
    for count, (ci, cj) in enumerate(tuples):
        util.print_progress(count, t, 'Pairs loop', 'complete')
        ci_candidates, cj_candidates = search_pair(ci, cj, True)
        add_to_db(ci, ci_candidates)
        add_to_db(cj, cj_candidates)
        # add_to_db(ci, radius_candidates[ci])
        # add_to_db(cj, radius_candidates[cj])
    #check_db()
    return do_final_validation()


def get_most_same_votes(final_votes):
    sort = np.argsort(final_votes)[::-1]
    highest = final_votes[sort[0]]
    if highest == 0: return 0
    highest_count = 0
    count = 0
    last = highest
    for i in sort:
        if final_votes[i] == last:
            count += 1
        else:
            last = final_votes[i]
            if count > highest_count: highest_count = count
            count = 1
    return highest_count


def search_pair(ci, cj, intersect_mode=True):
    d_appr = util.calculate_distance(centers[ci], centers[cj])
    d_upper = d_appr * (1 + dtol)
    d_lower = d_appr * (1 - dtol)
    pair_ids = pdb.search_d_appr(d_lower, d_upper)
    ci_candidates = pdb.get_ci_by_pair_ids(pair_ids)
    cj_candidates = pdb.get_cj_by_pair_ids(pair_ids)
    if intersect_mode:
        ci_candidates = np.intersect1d(ci_candidates, radius_candidates[ci])
        cj_candidates = np.intersect1d(cj_candidates, radius_candidates[cj])
    return ci_candidates, cj_candidates


def do_final_validation():
    indices = []
    n = len(database)
    minimum_candidates = 9999999999
    for v in database.values():
        sort_by_most_votes = np.argsort(v)[::-1]
        indices.append(sort_by_most_votes)
        num_candidates = np.where(v != 0)[0].shape[0]
        if minimum_candidates > num_candidates: minimum_candidates = num_candidates
    tuples = list(combos(np.arange(n), 2))
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


def validate_pair(start, votes, indices, a, b):
    top_a = indices[a][start]
    top_b = indices[b][start]
    a_candidates, b_candidates = search_pair(a, b)
    if top_a in a_candidates and top_b in b_candidates:
        votes[a] += 1
        votes[b] += 1
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
