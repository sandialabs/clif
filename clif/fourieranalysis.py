import xarray
import cftime
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker


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

    def compute_sampling_rate(self, data):
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
        self.compute_sampling_rate(data)

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
        return y_filt

    def compute_power_spectrum(self):
        freq_index = np.where(self.freq_full_ > 0)
        self.freq_ = self.freq_full_[freq_index]
        self.period_ = 1.0 / self.freq_
        self.power_ = np.abs(self.yhat_[freq_index]) ** 2

    def plot_power_spectrum(self, xaxis="frequency", logscale=False):
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
        ax.set_xlim([x.min(), x.max() / 2])
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(2))
        # set y-axis properties
        ax.set_ylabel("power")
        ax.set_ylim([0, y.max()])
        if logscale:
            ax.set_yscale("log")
        # misc axis properties
        ax.grid(True, alpha=0.5)
        return fig, ax
