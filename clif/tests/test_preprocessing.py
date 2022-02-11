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


def mse(a, b):
    return mean_squared_error(a, b, squared=False)


class TestPreprocessing(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        # load data sets
        nc_file = relpath + "/tests/data/t2m_1991_monthly.nc"
        nc_file2 = relpath + "/tests/data/AEROD_v_198501_199612.nc"
        self.xarray_dataset = xr.open_dataset(nc_file)
        self.ds = xr.open_mfdataset(nc_file2, chunks={"time": 1})

    def test_removing_monthly_cylcal_trend(self):
        # default constructor
        from clif.preprocessing import remove_cyclical_trends

        data = self.xarray_dataset
        data_new = remove_cyclical_trends(
            data=data, variable="t2m", cycle="month", new_variable_suffix="denoised"
        )
        assert "t2m" in data_new.variables, "Trend removed did not keep original data."
        assert "t2m_denoised" in data_new.variables, "Trend removed did not work."
        return

    def test_removing_yearly_cylcal_trend(self):
        # default constructor
        from clif.preprocessing import remove_cyclical_trends

        data = self.xarray_dataset
        data_new = remove_cyclical_trends(
            data=data, variable="t2m", cycle="year", new_variable_suffix="denoised"
        )
        assert "t2m" in data_new.variables, "Trend removed did not keep original data."
        assert "t2m_denoised" in data_new.variables, "Trend removed did not work."
        return

    def test_removing_hourly_cylcal_trend(self):
        # default constructor
        from clif.preprocessing import remove_cyclical_trends

        data = self.xarray_dataset
        data_new = remove_cyclical_trends(
            data=data, variable="t2m", cycle="hour", new_variable_suffix="denoised"
        )
        assert "t2m" in data_new.variables, "Trend removed did not keep original data."
        assert "t2m_denoised" in data_new.variables, "Trend removed did not work."
        return

    def test_removing_daily_cylcal_trend(self):
        # default constructor
        from clif.preprocessing import remove_cyclical_trends

        data = self.xarray_dataset
        data_new = remove_cyclical_trends(
            data=data, variable="t2m", cycle="day", new_variable_suffix="denoised"
        )
        assert "t2m" in data_new.variables, "Trend removed did not keep original data."
        assert "t2m_denoised" in data_new.variables, "Trend removed did not work."
        return

    def test_check_if_data_shared(self):
        # default constructor
        from clif.preprocessing import remove_cyclical_trends

        data = self.xarray_dataset
        data_new = remove_cyclical_trends(
            data=data, variable="t2m", cycle="day", new_variable_suffix="denoised"
        )
        data_new["t2m"] *= 0.0
        assert (
            np.sum(data["t2m"]).values > 0
        ), "Variables in old xarray dataset are being shared with new ones. Could cause issues."
        return


class TestPreprocessing2(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        # load data sets
        nc_file2 = relpath + "/tests/data/AEROD_v_198501_199612.nc"
        self.ds = xr.open_mfdataset(nc_file2, chunks={"time": 1})
        self.QOI = "AEROD_v"
        assert hasattr(
            self.ds, self.QOI
        ), "Xarray dataset does NOT contain {0} variable.".format(QOI)

    def test_remove_cyclical_trends_using_groupby_and_xarray_dataset_input(self):
        ds, QOI, data = self.ds, self.QOI, self.ds[self.QOI]
        ds_new = clif.preprocessing.remove_cyclical_trends(
            ds, QOI, cycle="month", new_variable_suffix="", use_groupby=True
        )
        data1 = ds_new[QOI + "_"]

        ds_new = clif.preprocessing.remove_cyclical_trends(
            ds, QOI, cycle="month", new_variable_suffix="", use_groupby=False
        )
        data2 = ds_new[QOI + "_"]

        # make sure both data sets are the same
        assert (
            np.nansum(data1.values - data2.values) == 0
        ), "Groupby cyclical trends by month is not working the same as the old for loop method."

    def test_remove_cyclical_trends_using_groupby_and_xarray_dataarray_input(self):
        ds, QOI, data = self.ds, self.QOI, self.ds[self.QOI]
        ds_new = clif.preprocessing.remove_cyclical_trends(
            ds, QOI, cycle="month", new_variable_suffix="", use_groupby=True
        )
        data1 = ds_new[QOI + "_"]

        data_new = clif.preprocessing.remove_cyclical_trends(
            data, cycle="month", new_variable_suffix="", use_groupby=True
        )

        # make sure both data sets are the same
        assert (
            np.nansum(data1.values - data_new.values) == 0
        ), "Groupby cyclical trends by month is not working the same as the old for loop method."


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
        trend_series_result = clif.preprocessing_new.StationarityTesting.test_stationarity(
            self.trend_series, 0.01, verbosity=0
        )
        assert (
            trend_series_result[0] == True and trend_series_result[1] == False
        ), "Test on linear trend is incorrect."

        stationary_data_result = clif.preprocessing_new.StationarityTesting.test_stationarity(
            self.stocastic_data[:, 0], 0.01, verbosity=0
        )
        assert (
            stationary_data_result[0] == True and stationary_data_result[1] == True
        ), "Test on stationary data is incorrect."

        # Not implemented:
        # heteroscedastic_result = clif.preprocessing_new.StationarityTesting.test_stationarity(
        #     stocastic_data[:, 1], 0.01, verbosity=0
        # )
        # assert (
        #     heteroscedastic_result[0] == True
        #     and heteroscedastic_result[1] == True
        #     and heteroscedastic_result[2] == False
        # ), "Test on heteroscedastic data is incorrect"

        seasonal_series_result = clif.preprocessing_new.StationarityTesting.test_stationarity(
            self.stocastic_data[:, 2], 0.01, verbosity=0
        )
        assert (
            seasonal_series_result[0] == False and seasonal_series_result[1] == False
        ), "Test on seasonal data is incorrect."

        sun_data_result = clif.preprocessing_new.StationarityTesting.test_stationarity(
            self.sun_data, 0.01, verbosity=0
        )
        assert (
            sun_data_result[0] == False and sun_data_result[1] == True
        ), "Test on trend-only stationary data is incorrect."

