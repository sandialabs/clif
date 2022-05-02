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

relpath, _ = os.path.split(clif.__file__)  # ignore the /__init__.py specification
print("\nrelative path:", relpath)


class TestStationarityTests(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.n = 1000
        self.alpha = 0.01
        self.rn = np.random.RandomState(2342)
        # stocastic_data[:, 0] is a stocastic gaussian process. This should always be stationary.
        self.stochastic_data = self.rn.randn(self.n, 3)
        # stochastic_data[:, 1] is (y_t = y_{t-1} * error) (error representing by data[:, 0])
        #   This represents a heteroscadistic process that *should* fail the test but it currently does not. The test does not test for scadisticity right now.
        # stochastic_data[:, 1] is (y_t = y_{t-1} + error) (error representing by data[:, 0])
        #   This represents a seasonal process, which is nonstationary.
        for t in range(self.n):
            self.stochastic_data[t, 1] = (
                self.stochastic_data[t - 1, 1] * self.stochastic_data[t, 0]
            )
            self.stochastic_data[t, 2] = (
                self.stochastic_data[t - 1, 2] + self.stochastic_data[t, 0]
            )
        # trend_series is a linearly increasing function representing a simple changing mean.
        #   It is difference stationary, so the ADF test return stationary and the KPSS test return nonstationary.
        #   Due to the difference in the results from ADF test and KPSS test, it can be inferred that the series is trend stationary and not strict stationary.
        self.trend_series = np.arange(0, self.n)
        # This dataset is useful because it is trend stationary, so it should make the ADF test return nonstationary and the KPSS test return stationary.
        #   Due to the difference in the results from ADF test and KPSS test, it can be inferred that the series is trend stationary and not strict stationary.
        self.sun_data = sm.datasets.sunspots.load_pandas().data["SUNACTIVITY"].values

    def test_stationarity_for_simple_linear_trend(self):
        time_series = self.trend_series
        alpha = self.alpha
        stest_adfuller = clif.statistics.StationarityTest(tests="adfuller", alpha=alpha)
        stest_adfuller.fit(time_series)
        assert (
            stest_adfuller.is_stationary is True
        ), "Unit root is present, i.e. series is non-stationary."
        stest_kpss = clif.statistics.StationarityTest(tests="kpss", alpha=alpha)
        stest_kpss.fit(time_series)
        assert (
            stest_kpss.is_stationary is False
        ), "linear time-series should not be stationary using the kpss test."

    def test_stationarity_for_simple_random_linear_trend(self):
        time_series = 1 + 0.1 * np.sort(self.rn.rand(self.n))
        alpha = self.alpha
        stest_adfuller = clif.statistics.StationarityTest(tests="adfuller", alpha=alpha)
        stest_adfuller.fit(time_series)
        assert (
            stest_adfuller.is_stationary is False
        ), "linear time-series should be stationary using the adfuller test."
        stest_kpss = clif.statistics.StationarityTest(tests="kpss", alpha=alpha)
        stest_kpss.fit(time_series)
        assert (
            stest_kpss.is_stationary is False
        ), "linear time-series should not be stationary using the kpss test."

    def test_stationarity_for_simple_random_process(self):
        time_series = self.stochastic_data[:, 0]
        alpha = self.alpha
        stest_adfuller = clif.statistics.StationarityTest(tests="adfuller", alpha=alpha)
        stest_adfuller.fit(time_series)
        assert (
            stest_adfuller.is_stationary is True
        ), "random time-series should be stationary using the adfuller test."
        stest_kpss = clif.statistics.StationarityTest(tests="kpss", alpha=alpha)
        stest_kpss.fit(time_series)
        assert (
            stest_kpss.is_stationary is True
        ), "random time-series should be stationary using the kpss test."

    def test_stationarity_for_sun_spot_data(self):
        time_series = self.sun_data
        alpha = self.alpha
        stest_adfuller = clif.statistics.StationarityTest(tests="adfuller", alpha=alpha)
        stest_adfuller.fit(time_series)
        assert (
            stest_adfuller.is_stationary is False
        ), "sun spot time-series should be not be stationary using the adfuller test."
        stest_kpss = clif.statistics.StationarityTest(tests="kpss", alpha=alpha)
        stest_kpss.fit(time_series)
        assert (
            stest_kpss.is_stationary is True
        ), "sun spot time-series should be stationary using the kpss test."

    def test_stationarity_for_additive_stochastic_process(self):
        time_series = self.stochastic_data[:, 2]
        alpha = self.alpha
        stest_adfuller = clif.statistics.StationarityTest(tests="adfuller", alpha=alpha)
        stest_adfuller.fit(time_series)
        assert (
            stest_adfuller.is_stationary is False
        ), "stochastic seasonal time-series should be not be stationary using the adfuller test."
        stest_kpss = clif.statistics.StationarityTest(tests="kpss", alpha=alpha)
        stest_kpss.fit(time_series)
        assert (
            stest_kpss.is_stationary is False
        ), "stochastic seasonal time-series should not be stationary using the kpss test."

    # def test_stationarity_for_heteraskedastic_time_series(self):
    #     # Not implemented:
    #     # heteroscedastic_result = clif.statistics.StationarityTest(
    #     #     stocastic_data[:, 1], 0.01, verbosity=0
    #     # )
    #     # assert (
    #     #     heteroscedastic_result[0] == True
    #     #     and heteroscedastic_result[1] == True
    #     #     and heteroscedastic_result[2] == False
    #     # ), "Test on heteroscedastic data is incorrect"
