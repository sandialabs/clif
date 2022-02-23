import numpy as np 
import matplotlib.pyplot as plt
from sklearn import datasets
import xarray as xr
import datetime

try:
	import clif
except:
	import sys
	sys.path.append("../")
	# from eof import fingerprints
	import clif


# define the data directory of E3SM historical decks
QOI = "AEROD_v" 
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

# preprocess the data set using clif
X,xr_data = clif.preprocessing.construct_data_matrix(dataset=ds,variable=QOI,row_coord=['time'],col_coord=['lat'],detrend='month',return_np_array_only=False)

# add linear detrending for numpy data matrix before fingerprinting
from scipy import signal
X_detrend = signal.detrend(X,axis=0)
X = X_detrend.copy()

# Now we can begin calculating the EOFs
# obtain fingerprints
n_components = 8
fp = clif.fingerprints(n_eofs=n_components,varimax=True)
fp.fit(X)

# extract pca fingerprints and convergence diagnostics
eofs_pca = fp.eofs_
explained_variance_ratio = fp.explained_variance_ratio_
eof_time_series = fp.projections_

# convert cftime series to datetime for plotting with matplotlib
times = ds.indexes['time'].to_datetimeindex()

# add trend lines to eofs
pinatubo_event = datetime.datetime(1991,6,15)

# plot eof's with trend lines before and after event
# import nc_time_axis # to allow plotting of cftime datetime using matplotlib
fig, axes = plt.subplots(3,2,figsize=(10,8))
fig.suptitle("EOF scores using PCA",fontsize=20)
for i,ax in enumerate(axes.flatten()):
    eof_ts = eof_time_series[:,i]
    (times_before, trend_before),(times_after, trend_after) =  clif.preprocessing.add_trend_lines_to_eof(times,eof_ts,change_point=pinatubo_event)
    ax.plot(times,eof_ts,label='PC score {0}'.format(i+1),color='C{0}'.format(i),alpha=.6)
    ax.plot(times_before,trend_before,'--',color='C{0}'.format(i),alpha=.7)
    ax.plot(times_after,trend_after,':',color='C{0}'.format(i),alpha=.7)
    ax.axvline(pinatubo_event,color='k',linestyle='--',alpha=.5)
    ax.legend(fancybox=True)
    ax.grid(True)

