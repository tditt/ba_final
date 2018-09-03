from itertools import combinations as combos

import numpy as np
from scipy.stats import mode

import cluster
import pairs_db as pdb
import util
from crater_search_util import add_votes_to_db, sort_indices_by_votes

dtol = 0
rtol = 0
vote_db = {}
debug_mode = True
relative_search = True
radius_candidates = {}
centers = None
kmpp = None
w_max = None
w_min = None
radii = None
r_rel_tol = None
verbose = False


def geometric_voting_abs(boxes, km_pp, r_tol, d_tol, wmax, verb=False):
    global radius_candidates, rtol, dtol, centers, vote_db, kmpp, w_max, relative_search, verbose
    relative_search = False
    vote_db = {}
    radius_candidates = {}
    dtol = d_tol
    rtol = r_tol
    w_max = wmax
    kmpp = km_pp
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
    center_points *= kmpp
    centers = center_points
    # get the radius candidates
    for c, r in enumerate(radii):
        radius_candidates[c] = pdb.get_craters_by_real_radius(r, r_tol)
    tuples = list(combos(np.arange(n), 2))
    t = len(tuples)
    intersect = False
    for count, (ci, cj) in enumerate(tuples):
        util.print_progress(count, t, 'Pairs loop', 'complete')
        ci_candidates, cj_candidates = search_pair(ci, cj, False)
        vote_db = add_votes_to_db(vote_db, ci, ci_candidates)
        vote_db = add_votes_to_db(vote_db, cj, cj_candidates)
        if not intersect:
            vote_db = add_votes_to_db(vote_db, ci, radius_candidates[ci])
            vote_db = add_votes_to_db(vote_db, cj, radius_candidates[cj])
    # check_db()
    return do_final_validation()


def geometric_voting_rel(boxes, wmax, wmin, rtol, intersect_mode, verb=False):
    global centers, w_max, w_min, radii, r_rel_tol, vote_db, relative_search, verbose
    relative_search = True
    vote_db = {}
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
            add_votes_to_db(vote_db, ci, ci_candidates)
            add_votes_to_db(vote_db, cj, cj_candidates)
        else:
            ci_d, ci_r, cj_d, cj_r = search_pair(ci, cj, False)
            add_votes_to_db(vote_db, ci, ci_d)
            add_votes_to_db(vote_db, ci, ci_r)
            add_votes_to_db(vote_db, cj, cj_d)
            add_votes_to_db(vote_db, cj, cj_r)

    return do_final_validation(), vote_db


def search_pair(ci, cj, intersect_mode=True):
    if relative_search:
        return pdb.search_pair_rel(ci, cj, radii, centers, r_rel_tol, w_min, w_max, intersect_mode)
    return pdb.search_pair_abs(ci, cj, radius_candidates, centers, dtol, intersect_mode)


def do_final_validation():
    n = len(vote_db)
    indices = sort_indices_by_votes(vote_db)
    tuples = list(combos(np.arange(n), 2))
    final_votes = np.zeros(n)
    for ci, cj in tuples[::-1]:
        final_votes = validate_pair(0, final_votes, indices, ci, cj)
    sort = np.argsort(final_votes)[::-1]
    top_scorers = {}

    top20_first_round = []
    for k in range(n):
        top = []
        for l in range(20):
            c_id = indices[k][l]
            coords = pdb.get_lunar_coordinates(c_id)
            radius = pdb.get_real_radius(c_id)
            votes = vote_db[k][c_id]
            top.append({'coords': coords, 'radius': radius, 'votes': votes})
        top20_first_round.append(top)

    clustering_candidates = np.zeros((n, 2))
    for e, i in enumerate(sort):
        crater_id = indices[i][0]
        coords = pdb.get_lunar_coordinates(indices[i][0])
        radius = pdb.get_real_radius(indices[i][0])
        votes = final_votes[i]
        clustering_candidates[e] = coords
        top_scorers[crater_id] = {'coords': coords, 'radius': radius, 'votes': votes}
        print("crater ", i, ": ", coords, ' with ', votes, ' votes!')

    if verbose: pdb.check_vote_db(vote_db)
    if not relative_search and n != 0:
        # try to find clusters
        eps = 1.2 * kmpp * w_max / 1737.5
        dbscan_results, cls = cluster.perform_clustering(clustering_candidates, eps)
        clusters = []
        for i in mode(dbscan_results.labels_[np.where(dbscan_results.labels_ != -1)]):
            clusters.append(np.where(dbscan_results.labels_ == i))
        for i in clusters:
            print(clustering_candidates[i])

    return top_scorers, top20_first_round


def validate_pair(start, votes, indices, ci, cj):
    top_ci = indices[ci][start]
    top_cj = indices[cj][start]
    candidates_ci, candidates_cj = search_pair(ci, cj)
    if top_ci in candidates_ci and top_cj in candidates_cj:
        votes[ci] += 1
        votes[cj] += 1
    return votes
