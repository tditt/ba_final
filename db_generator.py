import csv
import ogr
import time
import os
import pandas as pd
import h5py
import numpy as np
from operator import itemgetter
from math import sqrt, degrees, sin, pi, fabs
from itertools import combinations as combos
import k_vector
from util import approximate_visual_distance, calculate_pair_distances

db_basic_path = 'crater_database/'
file_crater_db = db_basic_path + 'crater_db.h5'
file_pairs_db = db_basic_path + 'pairs_db.h5'
file_pairs_index_db = db_basic_path + 'pairs_index.h5'
file_pairs_data_db = db_basic_path + 'pairs_data.h5'
file_k_floats = db_basic_path + 'k_floats.h5'
file_k_ints = db_basic_path + 'k_integers.h5'


# shape_file_lin = '/home/timo/code/util/resources/LU78287GT_GIS/LU78287GT_Moon2000.shp'
# shape_file_win = os.getcwd() + '\\resources\\LU78287GT_GIS\\LU78287GT_Moon2000.shp'
# driver = ogr.GetDriverByName('ESRI Shapefile')
# shape_dataset = driver.Open(shape_file_win, 0)
# shape_crater_layer = shape_dataset.GetLayer()


def get_craters_from_shapefile(crater_layer):
    craters = []
    for crater in crater_layer:
        radius_km = float(crater.GetField("Radius_km"))
        radius_deg = float(crater.GetField("Radius_deg"))
        lat = float(crater.GetField("Lat"))
        lon = float(crater.GetField("Lon_E"))
        name = str(crater.GetField("Name")).partition(' r:')[0]
        craters.append([lon, lat, radius_deg, radius_km, name])
    craters_sorted = sorted(craters, key=itemgetter(2))

    return craters_sorted


def generate_basic_db():
    hdf = pd.HDFStore(file_crater_db)
    craters = get_craters_from_shapefile()
    length = len(craters)
    # craters = get_craters_sorted()
    # with h5py.File("mytestfile.hdf5", "w") as f:
    #     dset = f.create_dataset("mydataset", (100,), dtype='d4')

    dict = {'lat': np.zeros(length), 'lon': np.zeros(length), 'radius': np.zeros(length)}
    for i, crater in enumerate(craters):
        dict['lat'][i] = crater[1]
        dict['lon'][i] = crater[0]
        dict['radius'][i] = crater[3]
    df = pd.DataFrame(dict, columns=['lat', 'lon', 'radius'])
    hdf.put('db', df, format='table', data_columns=True)
    hdf.close()


def generate_pairs_db(idx_start):
    start = time.time()
    hdf = pd.HDFStore(file_crater_db, 'r')
    db = hdf.get('/db')
    number_of_craters = db.shape[0]
    pair_generator = combos(np.arange(number_of_craters), 2)
    count = 0
    count_valid_pairs = 0
    # w_min is the width of smallest detection window
    w_min = 16
    # w_max is the width of the square detection image
    w_max = 832
    max_hav_distance = .35 * circumference_moon
    max_rel_distance = 2 * sqrt(2) * (w_max / w_min - w_min / 2)
    min_rel_radius = w_min / w_max
    iterations_to_report = 100000
    # create massive numpy arrays yo
    index = np.zeros(shape=(500000000, 2), dtype=np.uint32)
    pairs = np.zeros(shape=(500000000, 3), dtype=np.uint16)

    for i, (small, big) in enumerate(pair_generator):
        # python combo generator ensures that integers small < big.
        # db is sorted in ascending mode, so a low index is a small crater,
        # hence big_radius is always larger or equal to small_radius
        small_radius = approximate_visual_distance(db.loc[db.index[small], 'radius'])
        big_radius = approximate_visual_distance(db.loc[db.index[big], 'radius'])
        big_coords = [db.loc[db.index[big], 'lat'], db.loc[db.index[big], 'lon']]
        small_coords = [db.loc[db.index[small], 'lat'], db.loc[db.index[small], 'lon']]
        rel_radius = small_radius / big_radius
        # if ratio is too small craters cannot be in the same detection window
        if rel_radius < min_rel_radius: continue
        rel_dist, hav_dist, approx_dist = calculate_pair_distances(big_coords, small_coords, big_radius)
        # if real distance is more than 40% of the moon circumference, they pretty much can't be in same image
        if hav_dist > max_hav_distance: continue
        # if rel_distance between the two craters is more than max_rel_distance, they can't be in same image
        if rel_dist > max_rel_distance: continue
        # if we made it to here, it's a valid pair for db entry
        index[count_valid_pairs][0] = big
        index[count_valid_pairs][1] = small
        # normalize floating point values to increase precision and convert to int
        pairs[count_valid_pairs][0] = int(5000 * rel_radius)
        pairs[count_valid_pairs][1] = int(500 * rel_dist)
        pairs[count_valid_pairs][2] = int(10 * approx_dist)

        count += 1
        count_valid_pairs += 1
        if count >= iterations_to_report:
            print(i, ' iterations', ', ', count_valid_pairs, ' valid pairs')
            print('last valid pair : [(', big, ', ', small, '): rel_size: ', rel_radius, '; rel_dist: ', rel_dist,
                  '; approx_visual_dist: ', hav_dist, ' km]')
            print('time elapsed: ', time.time() - start)
            count = 0
    print('####################### FINISHED! ', count_valid_pairs, ' VALID PAIRS!')
    print('time elapsed: ', time.time() - start)
    cropped_index = index[:count_valid_pairs]
    cropped_pairs = pairs[:count_valid_pairs]
    file_index = h5py.File(file_pairs_index_db, 'w')
    file_index.create_dataset('pairs_index', data=cropped_index)
    file_index.close()
    file_pairs = h5py.File(file_pairs_data_db, 'w')
    file_pairs.create_dataset('pairs_data', data=cropped_pairs)
    file_pairs.close()


def generate_k_vector_db():
    # index = np.array(h5py.File(db_basic_path + 'pairs_index.h5', 'r')['pairs_index'])
    pairs = np.array(h5py.File(db_basic_path + 'pairs_data.h5', 'r')['pairs_data'])
    # k_to_h5(k_vector.construct_k_vector(index[..., 0]), 'pairs_index_big')
    # k_to_h5(k_vector.construct_k_vector(index[..., 1]), 'pairs_index_small')
    # k_to_h5(k_vector.construct_k_vector(pairs[..., 0]), 'pairs_data_radius')
    # k_to_h5(k_vector.construct_k_vector(pairs[..., 1]), 'pairs_data_distance'
    k_to_h5(k_vector.construct_k_vector(pairs[..., 2]), 'pairs_data_real_distance')


def k_to_h5(arr, name):
    k, sort, q, m = arr
    file_k = h5py.File(db_basic_path + name + '_k.h5', 'w')
    file_k.create_dataset(name + '_k', data=np.array([k, sort], dtype=np.uint32))
    file_k.close()
    file_qm = h5py.File(db_basic_path + name + '_qm.h5', 'w')
    file_qm.create_dataset(name + '_qm', data=[q, m])
    file_qm.close()


def append_to_hdf(df_to_append, path_to_target):
    hdf = pd.HDFStore(path_to_target)
    try:
        db = hdf.get('/db')
        db = pd.concat([db, df_to_append])
        hdf.put('db', db, format='table', data_columns=True)
    except:
        hdf.put('db', df_to_append, format='table', data_columns=True)
    hdf.close()


def generate_basic_db_csv():
    craters = get_craters_from_shapefile()
    with open('db.csv', 'w', newline='', encoding='utf-8') as db:
        db_writer = csv.writer(db, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        db_writer.writerows(craters)


def crop_db(idx=55885):
    hdf = pd.HDFStore(db_basic_path + 'crater_db.h5', 'r')
    crater_db = hdf.get('/db')
    index = h5py.File(db_basic_path + 'pairs_index.h5', 'r')['pairs_index']
    pairs = h5py.File(db_basic_path + 'pairs_data.h5', 'r')['pairs_data']
    new_path = 'cropped/'
    index = np.array(h5py.File(db_basic_path + 'pairs_index.h5', 'r')['pairs_index'])
    pairs = np.array(h5py.File(db_basic_path + 'pairs_data.h5', 'r')['pairs_data'])
    n_index = index.shape[0]
    n_pairs = index.shape[0]
    print(n_index, n_pairs)
    validpairs = []
    for i in range(n_pairs):
        if index[i, 0] < idx or index[i, 1] < idx: continue
        validpairs.append(i)

    b = np.array(validpairs, dtype=np.uint32)
    cropped_index = np.array(index[b], dtype=np.uint32)
    cropped_pairs = np.array(pairs[b], dtype=np.uint16)
    file_k = h5py.File(new_path + 'pairs_index.h5', 'w')
    file_k.create_dataset('pairs_index', data=cropped_index, dtype=np.uint32)
    file_k.close()
    file_k = h5py.File(new_path + 'pairs_data.h5', 'w')
    file_k.create_dataset('pairs_data', data=cropped_pairs, dtype=np.uint16)
    file_k.close()
