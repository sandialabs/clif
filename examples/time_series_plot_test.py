import xarray as xr
import dask
import os, sys
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import numpy as np
from sklearn.pipeline import Pipeline

import dask

dask.config.set({"array.slicing.split_large_chunks": False})

try:
    import clif
except:
    sys.path.append("../")
    # from eof import fingerprints
    import clif

# import visualization tools and functions
import clif.visualization as cviz
import clif.preprocessing as cpp
import autocorr


def monthly_fft_analysis():
    DATA_DIR = "../../e3sm_data/fingerprint/"
    freq = "monthly"
    Ts = xr.open_dataarray(os.path.join(DATA_DIR, f"T_{freq}.nc"), chunks={"time": 1})

    time_index = Ts.indexes["time"]
    time_diff = time_index[1:] - time_index[:-1]
    avg_dt_hours = np.mean([dti / np.timedelta64(1, "h") for dti in time_diff])

    sampling_freq_per_year = 365.25 / (avg_dt_hours / 24)

    x = time_index.to_datetimeindex(unsafe=True)
    y = Ts.values
    sampling_rate = int(sampling_freq_per_year)  # number of sampler per unit=year
    yhat = np.fft.fft(y)[1:]  # remove the mean, i.e. c0
    # plt.plot(yhat.real, yhat.imag, "ro", alpha=0.2)

    nhalf = int(np.floor(len(y) / 2))
    power = np.abs(yhat[:nhalf]) ** 2
    maxfreq = sampling_rate / 2  # shannon nyquist
    freq = np.arange(1, nhalf + 1) / (nhalf + 1) * maxfreq
    period = 1.0 / freq
    plt.plot(freq, power)
    plt.xlabel("Cycles/Year")
    # plt.xlim([0, period[50]])
    plt.ylim([0, power.max()])
    plt.grid(True)


# def hourly_fft_analysis():
DATA_DIR = "../../e3sm_data/fingerprint/"
freq = "hourly"
Ts = xr.open_dataarray(os.path.join(DATA_DIR, f"T_{freq}.nc"))[:]

fourier = clif.FourierTimeSeriesAnalysis(base_unit="month")
fourier.fit(data=Ts)
# fig, ax = fourier.plot_power_spectrum(xaxis="frequency", logscale=False)
y_filtered = fourier.transform(period_cutoff=1.1)
# y_filtered = Ts.copy()

# autocorrelation analysis
# y = Ts.values.copy()
yt = y_filtered.values
ds = 1.0 / fourier.sampling_freq_per_unit_
tlag_per_base_unit = ds * np.arange(len(yt))
acor_func = autocorr.function_1d(yt)
acor = autocorr.integrated_time(yt, c=5, tol=50, quiet=False)

acorT = clif.AutocorrelationAnalysis(base_unit="month")
acorT.fit(y_filtered)
acorT.plot("T", show_integrated_acor=False)

raise SystemExit(0)
time_index = Ts.indexes["time"]
time_diff = time_index[1:] - time_index[:-1]
avg_dt_hours = np.mean([dti / np.timedelta64(1, "h") for dti in time_diff])

# sampling_freq_per_unit = 365.25 / (avg_dt_hours / 24)
sampling_freq_per_unit = 30.5 / (avg_dt_hours / 24)  # monthly unit

x = time_index  # .to_datetimeindex(unsafe=True)
y = Ts.values
sampling_rate = int(sampling_freq_per_unit)  # number of sampler per unit=year
yhat = np.fft.fft(y)  # remove the mean, i.e. c0
# yhat_c = np.fft.ifftshift(yhat)
# if len(y) % 2 == 0:
#     argcenter = int(np.floor(len(y) / 2))
# else:
#     argcenter = int(np.floor((len(y) + 1) / 2))  # np.where(yhat_c == yhat[0])[0][0]

yhat0 = yhat[1:]

nhalf = int(np.floor(len(y) / 2))
power = np.abs(yhat0[:nhalf]) ** 2
maxfreq = sampling_rate / 2  # shannon nyquist
freq = np.arange(1, nhalf + 1) / (nhalf + 1) * maxfreq
period = 1.0 / freq

raise SystemExit(0)
# filter the signal
thresh_period = 0.9  # filter out any period < thresh
thresh_freq = 1.0 / thresh_period  # filter out any freq > 1./thresh_period

freq2 = np.fft.fftfreq(len(y), d=1.0 / sampling_rate)
yhat_filt = yhat.copy()
yhat_filt[np.where(np.abs(freq2) >= thresh_freq)] = 0.0

# dictionary of freq vs power
d = dict(zip(np.abs(freq2), yhat))
# plt.plot(d.keys(), np.abs(list(d.values())) ** 2)
# raise SystemExit(0)

y_filt = np.fft.ifft(yhat_filt).real

plt.plot(x, y, "r", alpha=0.5)
plt.plot(x, y_filt, "--b", alpha=0.6)
raise SystemExit(0)

plt.plot(period, power)
plt.xlabel("Months/Cycle")
# plt.xlabel("Cycles/Year")
# plt.xlim([0, period[50]])
plt.ylim([0, power.max()])
plt.grid(True)


def sunspot():
    # fft analysis for sunspot data to test
    Ts = np.loadtxt("sunspot.dat")
    x, y = Ts[:, 0], Ts[:, 1]
    yhat = np.fft.fft(y)[1:]  # remove the mean, i.e. c0
    # plt.plot(yhat.real, yhat.imag, "ro", alpha=0.2)

    sampling_rate = 12  # 1 sample per unit=year
    n = len(y)
    nhalf = int(np.floor(n / 2))
    power = np.abs(yhat[:nhalf]) ** 2
    maxfreq = sampling_rate / 2  # shannon nyquist
    freq = np.arange(1, nhalf + 1) / (nhalf + 1) * maxfreq
    # plt.plot(1.0 / freq, power)
    # plt.xlim([0, 50])
    # plt.ylim([0, power.max()])
    # plt.grid(True)

    # let's try filtering out the signal
    f = np.fft.fftfreq(len(y), d=1.0 / sampling_rate)
