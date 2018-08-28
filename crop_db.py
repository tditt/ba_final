import pandas as pd
import numpy as np
import h5py

db_basic_path = ''
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
    if index[i, 0] < 55885 or index[i, 1] < 55885: continue
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
