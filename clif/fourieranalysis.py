import xarray
import cftime
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
from . import autocorr


class FourierTimeSeriesAnalysis:
    """Fourier analysis of time series

    Parameters
    ----------
    base_unit: {'day', 'month', 'year'}, default='month'
        base unit used to calculate the sampling rate and frequency

    Examples
    --------
    >>>fourier = FourierTimeSeriesAnalysis(base_unit="month")
    >>>fourier.fit(data=Ts)
    >>>fourier.plot_power_spectrum(xaxis="period")
    >>>y_filtered = fourier.transform(period_cutoff=1.1)
    """

    def __init__(self, base_unit="year"):
        self.base_unit = base_unit

    def _compute_sampling_rate(self, data):
        # compute sampling rate
        time_index_ = data.indexes["time"]
        time_diff_ = time_index_[1:] - time_index_[:-1]
        avg_dt_hours_ = np.mean([dti / np.timedelta64(1, "h") for dti in time_diff_])

        if self.base_unit == "month":
            avg_days_per_month = 30.5
            self.sampling_freq_per_unit_ = avg_days_per_month / (
                avg_dt_hours_ / 24.0
            )  # monthly unit
        if self.base_unit == "year":
            avg_days_per_year = 365.25
            self.sampling_freq_per_unit_ = avg_days_per_year / (
                avg_dt_hours_ / 24.0
            )  # monthly unit
        if self.base_unit == "day":
            avg_days_per_day = 1.0
            self.sampling_freq_per_unit_ = avg_days_per_day / (
                avg_dt_hours_ / 24.0
            )  # monthly unit
        return

    def fit(self, data):
        # data must be time series data, i.e. have a time index
        assert isinstance(
            data, xarray.DataArray
        ), "Data must be a data array with time index"
        assert "time" in data.dims, "time must be a dimension in the data array"

        # compute sampling frequence per base_unit (month or year)
        self._compute_sampling_rate(data)

        if isinstance(data.indexes["time"][0], cftime._cftime.DatetimeNoLeap):
            x = data.indexes["time"].to_datetimeindex(unsafe=True)
        else:
            x = data.indexes["time"]
        self.t_ = x.copy()
        y = data.values

        # compute fft
        self.yhat_ = np.fft.fft(y)
        self.freq_full_ = np.fft.fftfreq(len(y), d=1.0 / self.sampling_freq_per_unit_)

        # power spectrum
        self.maxfreq_ = self.sampling_freq_per_unit_ / 2  # shannon nyquist
        self.compute_power_spectrum()

    def transform(self, freq_cutoff=None, period_cutoff=None):
        # simple bandwidth limiter, i.e. cut off frequences > threshold (or 1./freq > threshold if period is given)
        if freq_cutoff is not None:
            thresh_freq = freq_cutoff
        if period_cutoff is not None:
            thresh_freq = 1.0 / period_cutoff

        yhat_filt = self.yhat_.copy()
        yhat_filt[np.where(np.abs(self.freq_full_) >= thresh_freq)] = 0.0
        y_filt = np.fft.ifft(yhat_filt).real
        y_filt_xr = xarray.DataArray(y_filt, dims=["time"], coords={"time": self.t_})
        return y_filt_xr

    def compute_power_spectrum(self):
        freq_index = np.where(self.freq_full_ > 0)
        self.freq_ = self.freq_full_[freq_index]
        self.period_ = 1.0 / self.freq_
        self.power_ = np.abs(self.yhat_[freq_index]) ** 2

    def plot_power_spectrum(
        self, xaxis="frequency", logscale=False, xtickmultiple=2, xlim=None
    ):
        fig, ax = plt.subplots(figsize=(8.5, 4.0))
        if xaxis == "frequency":
            x = self.freq_
            xlabel = f"cycles/{self.base_unit}"
        elif xaxis == "period":
            x = self.period_
            xlabel = f"{self.base_unit}/cycle"
        y = self.power_
        # plot
        ax.plot(x, y)
        # set x-axis properties
        ax.set_xlabel(xlabel)
        if xlim is not None:
            ax.set_xlim([x.min(), xlim])
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(xtickmultiple))
        # set y-axis properties
        ax.set_ylabel("power")
        if logscale:
            ax.set_yscale("log")
        else:
            ax.set_ylim([0, y.max()])
        # misc axis properties
        ax.grid(True, alpha=0.5)
        return fig, ax


class AutocorrelationAnalysis(FourierTimeSeriesAnalysis):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def fit(self, data):
        self._compute_sampling_rate(data)
        yt = data.values
        self.ds_ = 1.0 / self.sampling_freq_per_unit_
        self.tlag_per_base_unit_ = self.ds_ * np.arange(len(yt))
        # compute autocorrelation function
        self.acor_func_ = autocorr.function_1d(yt)
        # get location where autocorrelation drops to zero
        self.tlag0_ = (self.tlag_per_base_unit_[self.acor_func_ < 0][:2]).mean()
        # Compute integrated autocorrelation time
        try:
            self.acor_integrated_ = autocorr.integrated_time(
                yt, c=5, tol=50, quiet=False
            )
        except:
            self.acor_integrated_ = self.ds_ * autocorr.integrated_time(
                yt, c=5, tol=0, quiet=False
            )
        self.acor_integrated_ = self.tlag_per_base_unit_[int(self.acor_integrated_[0])]

    def plot(self, ylabel="", xtickmultiple=2, show_integrated_acor=False):
        fig, ax = plt.subplots()
        ax.plot(self.tlag_per_base_unit_, self.acor_func_)
        ax.set_xlabel(self.base_unit + "s")

        # x axis properties
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(xtickmultiple))
        # set y-axis properties
        ax.set_ylabel(ylabel)
        # misc axis properties
        ax.grid(True, alpha=0.5)
        ax.axvline(x=self.tlag0_, alpha=0.5, linestyle="dashed")
        if show_integrated_acor:
            ax.axvline(
                x=self.acor_integrated_, alpha=0.5, color="r", linestyle="dashed"
            )
        return fig, ax
