import pandas as pd
import numpy as np
import h5py

db_basic_path = 'crater_db_final/'
hdf = pd.HDFStore(db_basic_path + 'crater_db.h5', 'r')
crater_db = hdf.get('/db')


# test = np.random.randint(0, k_integers.shape[1], 1000)
# for number in test:
#     print(str(k_integers[5, number]))
#

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


x = np.array([1, 1, 1, 2, 3, 4])
y = np.array([2, 1, 4, 6]) 
xy, x_ind, y_ind = np.intersect1d(x, y, return_indices=True, assume_unique=True)