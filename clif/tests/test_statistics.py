from sklearn.metrics import mean_squared_error
from sklearn import datasets
import clif
import numpy as np
import os
import statsmodels.api as sm
import time
import unittest
import warnings
import xarray as xr

relpath = clif.__file__[:-11]  # ignore the __init__.py specification
print("relpath:", relpath)


class TestStationarityTests(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.n = 1000
        # stocastic_data[:, 0] is a stocastic gaussian process. This should always be stationary.
        self.stocastic_data = np.random.randn(self.n, 3)
        # stocastic_data[:, 1] is (y_t = y_{t-1} * error) (error representing by data[:, 0])
        #   This represents a heteroscadistic process that *should* fail the test but it currently does not. The test does not test for scadisticity right now.
        # stocastic_data[:, 1] is (y_t = y_{t-1} + error) (error representing by data[:, 0])
        #   This represents a seasonal process, which is nonstationary.
        for t in range(self.n):
            self.stocastic_data[t, 1] = (
                self.stocastic_data[t - 1, 1] * self.stocastic_data[t, 0]
            )
            self.stocastic_data[t, 2] = (
                self.stocastic_data[t - 1, 2] + self.stocastic_data[t, 0]
            )
        # trend_series is a linearly increasing function representing a simple changing mean.
        #   It is difference stationary, so the ADF test return stationary and the KPSS test return nonstationary.
        #   Due to the difference in the results from ADF test and KPSS test, it can be inferred that the series is trend stationary and not strict stationary.
        self.trend_series = np.arange(0, self.n)
        # This dataset is useful because it is trend stationary, so it should make the ADF test return nonstationary and the KPSS test return stationary.
        #   Due to the difference in the results from ADF test and KPSS test, it can be inferred that the series is trend stationary and not strict stationary.
        self.sun_data = sm.datasets.sunspots.load_pandas().data["SUNACTIVITY"].values

    def test_stationarity_tests(self):
        trend_series_result = clif.statistics.stationarity(
            self.trend_series, 0.01, verbosity=0
        )
        assert (
            trend_series_result[0] == True and trend_series_result[1] == False
        ), "Test on linear trend is incorrect."

        stationary_data_result = clif.statistics.stationarity(
            self.stocastic_data[:, 0], 0.01, verbosity=0
        )
        assert (
            stationary_data_result[0] == True and stationary_data_result[1] == True
        ), "Test on stationary data is incorrect."

        # Not implemented:
        # heteroscedastic_result = clif.statistics.stationarity(
        #     stocastic_data[:, 1], 0.01, verbosity=0
        # )
        # assert (
        #     heteroscedastic_result[0] == True
        #     and heteroscedastic_result[1] == True
        #     and heteroscedastic_result[2] == False
        # ), "Test on heteroscedastic data is incorrect"

        seasonal_series_result = clif.statistics.stationarity(
            self.stocastic_data[:, 2], 0.01, verbosity=0
        )
        assert (
            seasonal_series_result[0] == False and seasonal_series_result[1] == False
        ), "Test on seasonal data is incorrect."

        sun_data_result = clif.statistics.stationarity(self.sun_data, 0.01, verbosity=0)
        assert (
            sun_data_result[0] == False and sun_data_result[1] == True
        ), "Test on trend-only stationary data is incorrect."
