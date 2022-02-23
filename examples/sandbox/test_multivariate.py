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
# Pipelining
################################################################

clipT = clif.preprocessing.ClipTransform(
    dims=["lat", "plev"], bounds=[(-60.0, 60.0), (5000.0, np.inf)]
)
monthlydetrend = clif.preprocessing.SeasonalAnomalyTransform(cycle="month")
intoutT = clif.preprocessing.MarginalizeOutTransform(dims=["lat", "lon"])
lindetrendT = clif.preprocessing.LinearDetrendTransform()
flattenT = clif.preprocessing.FlattenData(dims=["plev"])
transformT = clif.preprocessing.Transpose(dims=["time", "plev"])
scaleT = clif.preprocessing.ScalerTransform(scale_type="variance")

from sklearn.pipeline import Pipeline

pipe = Pipeline(
    steps=[
        ("clip", clipT),
        ("anom", monthlydetrend),
        ("marginalize", intoutT),
        ("detrend", lindetrendT),
        ("flatten", flattenT),
        ("transpose", transformT),
        ("scale", scaleT),
    ]
)

data_new = pipe.fit_transform(data)

# single variable selector test
colselectT = clif.preprocessing.SingleVariableSelector(variable="T")

pipe2 = Pipeline(
    steps=[
        ("selection", colselectT),
        ("clip", clipT),
        ("anom", monthlydetrend),
        ("marginalize", intoutT),
        ("detrend", lindetrendT),
        ("flatten", flattenT),
        ("transpose", transformT),
        ("scale", scaleT),
    ]
)

data_new2 = pipe2.fit_transform(ds)

# test column selector with python dictionary of variables
from sklearn.datasets import make_low_rank_matrix

rn = np.random.RandomState(234)
A = make_low_rank_matrix(
    n_samples=144, n_features=37, effective_rank=8, random_state=rn
)
B = make_low_rank_matrix(
    n_samples=144, n_features=24, effective_rank=6, random_state=rn
)

C = make_low_rank_matrix(
    n_samples=144, n_features=48, effective_rank=2, random_state=rn
)

A_xr = xr.DataArray(
    A,
    dims=["time", "plev"],
    coords={"time": np.linspace(1, 144, 144), "plev": np.linspace(0, 10000, 37)},
)
B_xr = xr.DataArray(
    B,
    dims=["time", "lat"],
    coords={"time": np.linspace(1, 144, 144), "lat": np.linspace(-80, 80, 24)},
)
C_xr = xr.DataArray(
    C,
    dims=["time", "lon"],
    coords={"time": np.linspace(1, 144, 144), "lon": np.linspace(-180, 180, 48)},
)

# create a transform for each data product separately
ds = xr.Dataset({"A": A_xr, "B": B_xr, "C": C_xr})

from clif.preprocessing import *

pipeA = Pipeline(
    steps=[
        ("colselect", SingleVariableSelector(variable="A", inverse=True)),
        ("scale", ScalerTransform()),
    ]
)

pipeB = Pipeline(
    steps=[
        ("colselect", SingleVariableSelector(variable="B", inverse=True)),
        ("scale", ScalerTransform()),
    ]
)

pipeC = Pipeline(
    steps=[
        ("colselect", SingleVariableSelector(variable="C", inverse=True)),
        ("scale", ScalerTransform()),
    ]
)

Ahat = 1 + 0 * pipeA.fit_transform(ds)
Bhat = 1.74 + 0 * pipeB.fit_transform(ds)
Chat = 3.14 + 0 * pipeC.fit_transform(ds)
dataarrays = [Ahat, Bhat, Chat]

# Combine all the data arrays
concatT = CombineDataArrays()
Z_concat = concatT.fit_transform(dataarrays)
Z_split = concatT.inverse_transform(Z_concat)

# alternate to stacking
da_dict = {"A": Ahat, "B": Bhat, "C": Chat}
stackT = StackDataArrays()
stacked_data = stackT.fit_transform(da_dict)
unstacked_data = stackT.inverse_transform(stacked_data)
