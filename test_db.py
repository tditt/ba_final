from math import sqrt

import numpy as np
import pandas as pd

import pairs_db as pdb

db_basic_path = 'crater_database/'
hdf = pd.HDFStore(db_basic_path + 'crater_db.h5', 'r')
crater_db = hdf.get('/db')


def search_crater(lat, lon):
    lat_tol = 180 * .005
    lon_tol = 360 * .005
    lat_min = lat - lat_tol
    lat_max = lat + lat_tol
    lon_min = lon - lon_tol
    lon_max = lon + lon_tol
    results = crater_db[((crater_db.lat >= lat_min) & (crater_db.lat <= lat_max) & (crater_db.lon >= lon_min) & (
            crater_db.lon <= lon_max) & (crater_db.radius >= 4))]
    return results


def perform_search_tests():
    w_min = 16
    w_max = 832
    r_rel_lower = 16 / 832
    r_rel_upper = 1
    d_rel_upper = 2 * sqrt(2) * ((w_max / w_min) - (w_min / 2))
    d_rel_lower = 0
    random_r_rel = np.random.uniform(r_rel_lower, r_rel_upper, [10000])
    random_d_rel = np.random.uniform(d_rel_lower, d_rel_upper, [10000])
    random_r_abs = np.random.uniform(4, 100, [10000])
    random_d_abs = np.random.uniform(10, 2000, [10000])
    tol_r = 0.01
    tol_d = 0.01
    n = 10000
    n_r = 0
    for r in random_r_rel:
        lower, upper = get_search_range(r, tol_r)
        n_r += pdb.search_r_rel(lower, upper).shape[0]
    n_r /= n
    n_d = 0
    for d in random_d_rel:
        lower, upper = get_search_range(d, tol_d)
        n_d += pdb.search_d_rel(lower, upper).shape[0]
    n_d /= n
    n_r_abs = 0
    for r in random_r_abs:
        n_r_abs += pdb.get_craters_by_real_radius(r, tol_r).shape[0]
    n_r_abs /= n
    n_d_abs = 0
    for d in random_d_abs:
        lower, upper = get_search_range(d, tol_d)
        n_d_abs += pdb.search_d_appr(lower, upper).shape[0]
    n_d_abs /= n
    print('average search results for... relative radius: ', n_r, '| relative distance: ', n_d, ' | absolute radius : ',
          n_r_abs, ' | absolute distance : ', n_d_abs)


def get_search_range(value, tolerance):
    lower = value * (1 - tolerance)
    upper = value * (1 + tolerance)
    return lower, upper

