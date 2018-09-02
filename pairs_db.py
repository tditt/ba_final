import numpy as np
import h5py
import pandas as pd
import k_vector


def read_hdf_to_ram(db_path, idx, target_arr, filename):
    datafile = h5py.File(db_path + filename + '.h5', 'r')
    if idx is None:
        datafile[filename].read_direct(target_arr)
    else:
        datafile[filename].read_direct(target_arr, np.s_[idx])
    datafile.close()


# some definitions and array initializations
db_basic_path = 'crater_database/'
file_name_k_radius = 'pairs_data_radius_k'
file_name_k_distance = 'pairs_data_distance_k'
file_name_qm_radius = 'pairs_data_radius_qm'
file_name_qm_distance = 'pairs_data_distance_qm'
file_name_k_real_distance = 'pairs_data_real_distance_k'
file_name_qm_real_distance = 'pairs_data_real_distance_qm'
file_name_index = 'pairs_index'
file_name_data = 'pairs_data'
hdf = pd.HDFStore(db_basic_path + 'crater_db.h5', 'r')
crater_db = hdf.get('/db')
pairs_len = h5py.File(db_basic_path + 'pairs_data_radius_k.h5', 'r')['pairs_data_radius_k'][0].shape[0]
k_vector_radius = np.zeros(pairs_len, dtype=np.uint32)
k_sort_radius = np.zeros(pairs_len, dtype=np.uint32)
qm_radius = np.zeros(2, dtype=np.float64)
k_vector_distance = np.zeros(pairs_len, dtype=np.uint32)
k_sort_distance = np.zeros(pairs_len, dtype=np.uint32)
qm_distance = np.zeros(2, dtype=np.float64)
k_vector_real_distance = np.zeros(pairs_len, dtype=np.uint32)
k_sort_real_distance = np.zeros(pairs_len, dtype=np.uint32)
qm_real_distance = np.zeros(2, dtype=np.float64)
index = np.zeros((pairs_len, 2), dtype=np.uint32)
pairs = np.zeros((pairs_len, 3), dtype=np.uint16)

# load the whole database directly into RAM as numpy arrays
print('loading database into RAM...')
read_hdf_to_ram(db_basic_path, 0, k_vector_radius, file_name_k_radius)
read_hdf_to_ram(db_basic_path, 1, k_sort_radius, file_name_k_radius)
read_hdf_to_ram(db_basic_path, 0, k_vector_distance, file_name_k_distance)
read_hdf_to_ram(db_basic_path, 1, k_sort_distance, file_name_k_distance)
read_hdf_to_ram(db_basic_path, 0, k_vector_real_distance, file_name_k_real_distance)
read_hdf_to_ram(db_basic_path, 1, k_sort_real_distance, file_name_k_real_distance)
read_hdf_to_ram(db_basic_path, None, qm_radius, file_name_qm_radius)
read_hdf_to_ram(db_basic_path, None, qm_distance, file_name_qm_distance)
read_hdf_to_ram(db_basic_path, None, qm_real_distance, file_name_qm_real_distance)
read_hdf_to_ram(db_basic_path, None, index, file_name_index)
read_hdf_to_ram(db_basic_path, None, pairs, file_name_data)
q_radius = qm_radius[0]
m_radius = qm_radius[1]
q_distance = qm_distance[0]
m_distance = qm_distance[1]
q_real_distance = qm_real_distance[0]
m_real_distance = qm_real_distance[1]
print('... done!')
print('number of pairs in database: ', pairs_len)

# some parameters
r_rel_normalization_factor = 5000
d_rel_normalizaition_factor = 500
d_appr_normalization_factor = 10


def get_real_radius(crater):
    return crater_db.loc[crater_db.index[crater], 'radius']


def get_craters_by_real_radius(radius, tolerance):
    upper = radius * (1 + tolerance)
    lower = radius * (1 - tolerance)
    return np.array(crater_db[(crater_db.radius <= upper) & (crater_db.radius >= lower)].index)


def get_lunar_coordinates(crater):
    return [crater_db.loc[crater_db.index[crater], 'lat'], crater_db.loc[crater_db.index[crater], 'lon']]


def search_r_rel(r_lower, r_upper):
    y_a = r_lower * r_rel_normalization_factor
    y_b = r_upper * r_rel_normalization_factor
    return k_vector.k_vector_search(True, y_a, y_b, pairs[..., 0], k_vector_radius, k_sort_radius, q_radius,
                                    m_radius)


def search_d_rel(d_lower, d_upper):
    y_a = d_lower * d_rel_normalizaition_factor
    y_b = d_upper * d_rel_normalizaition_factor
    return k_vector.k_vector_search(True, y_a, y_b, pairs[..., 1], k_vector_distance, k_sort_distance, q_distance,
                                    m_distance)


def search_d_appr(d_lower, d_upper):
    y_a = d_lower * d_appr_normalization_factor
    y_b = d_upper * d_appr_normalization_factor
    return k_vector.k_vector_search(True, y_a, y_b, pairs[..., 2], k_vector_real_distance, k_sort_real_distance,
                                    q_real_distance,
                                    m_real_distance)


def get_ci_by_pair_ids(pair_ids):
    return index[..., 0][pair_ids]


def get_cj_by_pair_ids(pair_ids):
    return index[..., 1][pair_ids],
