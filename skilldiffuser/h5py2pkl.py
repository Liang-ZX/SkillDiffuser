import pdb

import h5py
import numpy as np
import pickle
import pandas as pd
import os

root_path = '/home/zxliang/dataset/lorel/may_08_sawyer_50k'
f = h5py.File(os.path.join(root_path, 'data.hdf5'),'r')
# f.visit(lambda x: print(x))

data = dict()

df = pd.read_table(os.path.join(root_path, "labels.csv"), sep=",")
langs = df["Text Description"].str.strip().to_numpy().reshape(-1)
langs = np.array(['' if x is np.isnan else x for x in langs])
filtr1 = np.array([int(("nothing" in l) or ("nan" in l) or ("wave" in l)) for l in langs])
filtr = filtr1 == 0
data['langs'] = langs[filtr]

for group in f.keys():
    for key in f[group].keys():
        print(group, key)
        data[key] = f[group][key][:][filtr]

with open(os.path.join(root_path, "prep_data3.pkl"),"wb") as fo:
    pickle.dump(data, fo, protocol=4)
