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


def mse(a, b):
    return mean_squared_error(a, b, squared=False)


class TestPreprocessing(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        # load data sets
        nc_file = relpath + "/tests/data/t2m_1991_monthly.nc"
        nc_file2 = relpath + "/tests/data/AEROD_v_198501_199612.nc"
        self.xarray_dataset = xr.open_dataset(nc_file)
        self.ds = xr.open_dataset(nc_file2)

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
        self.ds = xr.open_dataset(nc_file2)
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
