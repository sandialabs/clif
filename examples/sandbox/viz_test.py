from calendar import month
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import xarray as xr
import datetime, os, sys
from tqdm import tqdm
from time import time

try:
    import clif
except:
    import sys

    sys.path.append("../")
    # from eof import fingerprints
    import clif

# define the data directory of E3SM historical decks
QOI = "T"
# define source of H* directories
SOURCE_DIR = os.path.join(os.getenv("HOME"), "Research", "e3sm_data/fingerprint/")

##################################################################
## Gather all H1 through H5 deck data
##################################################################
QOI_FILE_PATH = []
for hi in list(range(1, 5 + 1)):
    DATA_DIR = os.path.join(SOURCE_DIR, "H{0}/24x48/monthly/12yr".format(hi))
    # get file with QoI
    import glob

    QOI_FILE_PATH += glob.glob(DATA_DIR + "/{0}*".format(QOI))


# use xarray to open the data set and use dask to import as chunks
deck_runs = {}
for ii, fi in enumerate(QOI_FILE_PATH):
    ds_temp = xr.open_dataset(fi)  # add chunks={"time": 1} for large data files
    deck_runs["H{0}".format(ii + 1)] = ds_temp

# Average over all deck data and combine to a single data set
deck_combined = xr.concat([val for key, val in deck_runs.items()], "x")
ds = deck_combined.mean(dim="x")
lat_lon_weights = ds.area
data = ds["T"]

################################################################
# Pipelining Transforms
################################################################

clipT = clif.preprocessing.ClipTransform(
    dims=["lat", "plev"], bounds=[(-60.0, 60.0), (5000.0, np.inf)]
)
monthlydetrend = clif.preprocessing.SeasonalAnomalyTransform(cycle="month")
intoutT = clif.preprocessing.MarginalizeOutTransform(
    dims=["lat", "lon"], lat_lon_weights=lat_lon_weights_new
)
lindetrendT = clif.preprocessing.LinearDetrendTransform()
flattenT = clif.preprocessing.FlattenData(dims=["plev"])
transformT = clif.preprocessing.Transpose(dims=["time", "plev"])

from sklearn.pipeline import Pipeline

# Pipelining with the EOF transform at the end
fp = clif.fingerprints(n_eofs=8, varimax=False)
pipe_w_eof = Pipeline(
    steps=[
        ("clip", clipT),
        ("anom", monthlydetrend),
        ("marginalize", intoutT),
        ("detrend", lindetrendT),
        ("flatten", flattenT),
        ("transpose", transformT),
        ("fingerprint", fp),
    ]
)
pipe_w_eof.fit(data)
eofs_ = pipe_w_eof.named_steps["fingerprint"].eofs_
evr_ = pipe_w_eof.named_steps["fingerprint"].explained_variance_ratio_

