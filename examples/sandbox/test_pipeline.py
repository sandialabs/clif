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


# ######################################################################
# ## Begin fingerprinting and plotting EOF time-series scores
# ######################################################################
# # Now we can begin calculating the EOFs
# # obtain fingerprints
# n_components = 8
# fp = clif.fingerprints(n_eofs=n_components, varimax=False)
# fp.fit(data_new)

# # extract pca fingerprints and convergence diagnostics
# eofs_pca = fp.eofs_
# explained_variance_ratio = fp.explained_variance_ratio_
# eof_time_series = fp.projections_
# print(
#     "Explained variance ratios for first {0} components:\n".format(n_components),
#     explained_variance_ratio,
# )

# # i conver tcftime series to datetime for plotting with matplotlib
# times = data.indexes["time"].to_datetimeindex(unsafe=True)

# # add trend lines to eofs
# pinatubo_event = datetime.datetime(1991, 6, 15)

# # plot eof's with trend lines before and after event
# # import nc_time_axis # to allow plotting of cftime datetime using matplotlib
# fig, axes = plt.subplots(3, 2, figsize=(10, 8))
# fig.suptitle("EOF scores for {0} using PCA".format(QOI), fontsize=20)
# for i, ax in enumerate(axes.flatten()):
#     eof_ts = eof_time_series[:, i]
#     ax.plot(
#         times,
#         eof_ts,
#         label="PC score {0}".format(i + 1),
#         color="C{0}".format(i),
#         alpha=0.6,
#     )
#     ax.axvline(pinatubo_event, color="k", linestyle="--", alpha=0.5)
#     ax.legend(fancybox=True)
#     ax.grid(True)

# plt.show()
