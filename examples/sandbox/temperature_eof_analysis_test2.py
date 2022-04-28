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

    sys.path.append("../../")
    # from eof import fingerprints
    import clif

# define the data directory of E3SM historical decks
QOI = "T"
ERA5_QOI = "ta"
PI_QOI = "T"
# define source of H* directories
SOURCE_DIR = os.path.join(os.getenv("HOME"), "Research", "e3sm_data/fingerprint/")
ERA5_SOURCE_DIR = os.path.join(
    os.getenv("HOME"), "Research", "e3sm_data/fingerprint/ERA5"
)
PI_SOURCE_DIR = os.path.join(os.getenv("HOME"), "Research", "e3sm_data/fingerprint/PI")

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
    ds_temp = xr.open_dataset(
        fi, chunks={"time": 1}
    )  # add chunks={"time": 1} for large data files
    deck_runs["H{0}".format(ii + 1)] = ds_temp

# Average over all deck data and combine to a single data set
deck_combined = xr.concat([val for key, val in deck_runs.items()], "x")
ds = deck_combined.mean(dim="x")
lat_lon_weights = ds.area
data_all = ds[QOI]

data_before = data_all.sel(time=slice("1985-01-01", "1991-06-20"))
mu_before = data_before.groupby("time.month").mean("time")
data_after = data_all.sel(time=slice("1991-06-20", "1997-12-31"))

################
# Import ERA5 and PI data
################
ERA5_FILE = os.path.join(ERA5_SOURCE_DIR, "ta_24x48_198501_199612.nc")
era5_data = xr.open_dataset(ERA5_FILE, chunks={"time": 1})[ERA5_QOI]

PI_FILE = os.path.join(PI_SOURCE_DIR, "T_020001_027912.nc")
pi_data = xr.open_dataset(PI_FILE, chunks={"time": 1})[PI_QOI]

# split up pre-industrial data into groups of 12 years
nyears = 12
pi_data_split = []
for i in range(int(pi_data.shape[0] / 12 / 12)):
    pi_data_temp = pi_data[-12 * nyears * (i + 1) :][: 12 * nyears]
    pi_data_split.append(pi_data_temp)

# data = data_before.copy()
data = data_all.copy()

##################################################################
## Preprocess data using new API
##################################################################

clipT = clif.preprocessing.ClipTransform(dims=["lat"], bounds=[(-60.0, 60.0)])
lat_lon_weights_clipped = clipT.fit_transform(lat_lon_weights)

monthlydetrend = clif.preprocessing.SeasonalAnomalyTransform(cycle="month")

intoutT = clif.preprocessing.MarginalizeOutTransform(
    dims=["lat", "lon"], lat_lon_weights=lat_lon_weights
)

lindetrendT = clif.preprocessing.LinearDetrendTransform()

transformT = clif.preprocessing.Transpose(dims=["time", "plev"])

from sklearn.pipeline import Pipeline

pipe = Pipeline(
    steps=[
        # ("clip", clipT),
        ("anom", monthlydetrend),
        ("marginalize", intoutT),
        # ("detrend", lindetrendT),
        ("transpose", transformT),
    ]
)

# Transform average DECK data to obtain EOF
data_H_avg_trfm = pipe.fit_transform(data)

data_H_trfm_all = []
for key, val in deck_runs.items():
    print(key)
    data_H_temp = deck_runs[key][QOI]
    data_H_temp_trfm = pipe.fit_transform(data_H_temp)
    data_H_trfm_all.append(data_H_temp_trfm)

# transform the era 5 data
print("Transforming era and pre-industrial data sets...")
data_era5_trfm = pipe.fit_transform(era5_data)

pi_data_split_t = []
for pi_data_i in pi_data_split:
    data_pi_trfm_temp = pipe.fit_transform(pi_data_i)
    pi_data_split_t.append(data_pi_trfm_temp)

######################################################################
## Begin fingerprinting and plotting EOF time-series scores
######################################################################
# Now we can begin calculating the EOFs
# obtain fingerprints
n_components = 2
fp = clif.fingerprints(n_eofs=n_components, varimax=False)

# Fit EOF to deck avg data
print("Computing fingerprints...")
fp.fit(data_H_avg_trfm)

# extract pca fingerprints and convergence diagnostics
eofs_pca = fp.eofs_

# plot eofs:
fig, ax = plt.subplots(1, figsize=[10, 4])
ax.plot(data["plev"].values, eofs_pca[0])
ax.grid(True)
ax.set_xscale("log")
ax.set_xlabel("hPa")
ax.set_ylabel("Scaled \n principal \ncomponent", rotation=0, labelpad=26)
ax.invert_xaxis()

print("Transforming data...")
explained_variance_ratio = fp.explained_variance_ratio_
eof_time_series_H_avg = fp.transform(data_H_avg_trfm.values)
eof_time_series_era5 = fp.transform(data_era5_trfm.values)

eof_time_series_H_data = []
for data_H_temp in data_H_trfm_all:
    eof_time_series_temp = fp.transform(data_H_temp.values)
    eof_time_series_H_data.append(eof_time_series_temp)

eof_time_series_pi_data = []
data_pi_avg = np.mean(
    np.array([data_pi_temp.values for data_pi_temp in pi_data_split_t]), axis=0
)
eof_time_series_pi_avg = fp.transform(data_pi_avg)
for data_pi_temp in pi_data_split_t:
    eof_time_series_temp = fp.transform(data_pi_temp.values)
    eof_time_series_pi_data.append(eof_time_series_temp)

print(
    "Explained variance ratios for first {0} components:\n".format(n_components),
    explained_variance_ratio,
)

# i conver tcftime series to datetime for plotting with matplotlib
times = data_all.indexes["time"].to_datetimeindex(unsafe=True)

# add trend lines to eofs
pinatubo_event = datetime.datetime(1991, 6, 15)

# plot eof's with trend lines before and after event
# import nc_time_axis # to allow plotting of cftime datetime using matplotlib
fig, axes = plt.subplots(2, int(n_components / 2), figsize=(11, 5))
fig.suptitle("EOF scores/ loadings", fontsize=20)
for i, ax in enumerate(axes.flatten()):
    eof_ts = eof_time_series_H_avg[:, i]
    eof_ts_H_deck = np.array(
        [eof_time_series_H_data[ii][:, i] for ii in range(len(data_H_trfm_all))]
    ).T
    eof_ts_PI = np.array(
        [
            eof_time_series_pi_data[ii][:, i]
            for ii in range(len(eof_time_series_pi_data))
        ]
    ).T
    eof_ts_H_quantiles = np.quantile(eof_ts_H_deck, [0.025, 0.975], axis=1)
    eof_ts_PI_quantiles = np.quantile(eof_ts_PI, [0.025, 0.975], axis=1)
    ax.plot(
        times,
        eof_ts,
        label="PC score {0}".format(i + 1),
        color=f"C{i}",
        linewidth=2,
        alpha=0.8,
    )
    ax.fill_between(
        x=times,
        y1=eof_ts_H_quantiles[0],
        y2=eof_ts_H_quantiles[1],
        color="C{0}".format(i),
        alpha=0.2,
    )
    ax.plot(
        times,
        eof_time_series_pi_avg[:, i],
        color=f"C{i+1}",
        alpha=0.7,
        linestyle="-",
        linewidth=1,
        label="pre-industrial score",
    )
    ax.fill_between(
        x=times,
        y1=eof_ts_PI_quantiles[0],
        y2=eof_ts_PI_quantiles[1],
        color="C{0}".format(i + 1),
        alpha=0.2,
    )
    ax.plot(
        times,
        eof_time_series_era5[:, i],
        color=f"C{i}",
        alpha=0.7,
        linestyle="-",
        linewidth=1,
        label="ERA5 score",
    )
    ax.axvline(pinatubo_event, color="k", linestyle="--", alpha=0.5)
    ax.legend(fancybox=True)
    ax.grid(True)

##################
# Plot trend analysis
##################
t = np.arange(len(times)) / 12.0
eof_ts = eof_time_series_H_avg[:, 0].copy()

eof_ts_PI = np.array(
    [eof_time_series_pi_data[ii][:, i] for ii in range(len(eof_time_series_pi_data))]
).T

from sklearn.linear_model import LinearRegression


def compute_trend_coef(X, y, return_pred=True):
    if X.ndim == 1:
        X = X[:, np.newaxis]
    linreg = LinearRegression(fit_intercept=True)
    linreg.fit(X, y)
    if return_pred:
        return linreg.coef_, linreg.predict(X)
    return linreg.coef_


linreg = LinearRegression()
tot_years = 12
pi_samples = eof_ts_PI.shape[1]
beta_L = np.zeros(tot_years)
beta_L_pi = np.zeros((tot_years, pi_samples))
for L in range(tot_years):
    L_index = range(12 * (L + 1))
    t_L = t[L_index]
    y_L = eof_ts[L_index]
    beta_temp, y_pred = compute_trend_coef(t_L, y_L)
    beta_L[L] = beta_temp

for j in range(pi_samples):
    eof_ts_temp = eof_ts_PI[:, j]
    for L in range(tot_years):
        L_index = range(12 * (L + 1))
        t_L = t[L_index]
        y_L = eof_ts_temp[L_index]
        beta_temp, y_pred = compute_trend_coef(t_L, y_L)
        beta_L_pi[L, j] = beta_temp

N_L = beta_L_pi.var(axis=1)
fig, ax = plt.subplots(1, 1, figsize=[8, 5])
ax.plot(range(1986, 1996), beta_L[1:-1] / N_L[1:-1])
ax.set_xlabel("year")
ax.set_ylabel(r"$\beta(y)/N(y)$")
ax.grid(True)
