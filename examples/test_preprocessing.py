import numpy as np 
import matplotlib.pyplot as plt
from sklearn import datasets
import xarray as xr
import datetime
from time import time

try:
	import clif
except:
	import sys
	sys.path.append("../")
	# from eof import fingerprints
	import clif


# define the data directory of E3SM historical decks
QOI = "QRL" 
H_NUM = str(1) # Deck ensemble to choose

import os, sys
HOME_DIR = os.getenv("HOME")
DATA_DIR = os.path.join(HOME_DIR,"Research/e3sm_data/fingerprint/H{0}/24x48/monthly/12yr".format(H_NUM))
print(DATA_DIR)

# get file with QoI
import glob
QOI_FILE_PATH = glob.glob(DATA_DIR + "/*{0}*".format(QOI))

# use xarray to open the data set and use dask to import as chunks
ds = xr.open_mfdataset(QOI_FILE_PATH, chunks={'time': 1})
assert hasattr(ds,QOI), "Xarray dataset does NOT contain {0} variable.".format(QOI)

data = ds[QOI]

# using polyfit native to xarray (much slower)
lindetrend = clif.preprocessing.linear_detrending()
data_new = lindetrend.fit_transform(data)


# # ########################################## CLASS TESTS
# clipT = clif.preprocessing.clip(dims=['lat','lon'],bounds=[(-60,60),(150,np.inf)])
# data_new = clipT.fit_transform(data)
# print(data_new.values.shape)

# mt = clif.preprocessing.marginalize(coords=['lon'],lat_lon_weighted=True,lat_lon_weights=ds.area)
# data_new = mt.fit_transform(data)

# data = ds[QOI]
# ds_new = clif.preprocessing.remove_cyclical_trends( ds, QOI, cycle='month',new_variable_suffix='', use_groupby=True)
# data1 = ds_new[QOI+'_']

# T1 = clif.preprocessing.seasonal_detrending(cycle='month')
# T1.fit(data)
# data_new = T1.transform(data)
# assert np.nansum(data_new.values - data1.values) == 0

# # Test2 
# T1 = clif.preprocessing.seasonal_detrending(cycle='month')
# data_new = T1.fit_transform(data)
# assert np.nansum(data_new.values - data1.values) == 0


# ds_new = clif.preprocessing.remove_cyclical_trends( ds, QOI, cycle='month',new_variable_suffix='', use_groupby=False)
# data2 = ds_new[QOI+'_']

# # make sure both data sets are the same
# assert np.nansum(data1.values - data2.values) == 0


