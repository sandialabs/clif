from ast import Assert
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import xarray as xr
import datetime
from time import time
import unittest
import clif
import os, sys

relpath = clif.__file__[:-11]  # ignore the __init__.py specification
print("relpath:", relpath)


class TestSeasonalDetrending(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        # load data sets
        AERO_FILE = os.path.join(relpath, "tests/data/AEROD_v_198501_199612.nc")
        QRL_FILE = os.path.join(relpath, "tests/data/QRL.nc")
        T_FILE = os.path.join(relpath, "tests/data/Temp.nc")
        AREA_FILE = os.path.join(relpath, "tests/data/area_weights.nc")
        self.ds_AERO = xr.open_dataset(AERO_FILE, chunks={"time": 1})
        self.ds_QRL = xr.open_dataset(QRL_FILE, chunks={"time": 1})
        self.ds_T = xr.open_dataset(T_FILE, chunks={"time": 1})
        self.area_weights = xr.open_dataset(AREA_FILE)["area"]

    def test_seasonal_anomaly_detrending_fails_for_xarray_datasets(self):
        sat = clif.preprocessing.SeasonalAnomalyTransform(cycle="month")
        data = self.ds_AERO["AEROD_v"]
        with self.assertRaises(AssertionError):
            sat.fit(self.ds_AERO)

    def test_seasonal_anomaly_detrending_init_works_for_xarray_dataarrays(self):
        sat = clif.preprocessing.SeasonalAnomalyTransform(cycle="month")
        data = self.ds_AERO["AEROD_v"]
        sat.fit(data)
        sat.transform(data)

    def test_seasonal_anomaly_detrending_must_call_fit_before_transform(self):
        sat = clif.preprocessing.SeasonalAnomalyTransform(cycle="month")
        data = self.ds_AERO["AEROD_v"]
        with self.assertRaises(AssertionError):
            sat.transform(data)

    def test_seasonal_anomaly_detrending_fit_transform(self):
        sat = clif.preprocessing.SeasonalAnomalyTransform(cycle="month")
        data = self.ds_AERO["AEROD_v"]
        data_new = sat.fit_transform(data)

    def test_seasonal_anomaly_detrending_groupby_is_working(self):
        sat1 = clif.preprocessing.SeasonalAnomalyTransform(cycle="month")
        data1 = self.ds_AERO["AEROD_v"]
        data1_T = sat1.fit_transform(data1)
        ds2 = clif.preprocessing.remove_cyclical_trends(
            data=self.ds_AERO,
            variable="AEROD_v",
            cycle="month",
            new_variable_suffix="",
            use_groupby=False,
        )
        data2_T = ds2["AEROD_v_"]
        error = np.nansum((data1_T.values - data2_T.values) ** 2)
        assert (
            error == 0.0
        ), "Seasonal detrending not doing what it is supposed to be doing. "

    def test_seasonal_anomaly_detrending_groupby_is_working_w_T(self):
        sat1 = clif.preprocessing.SeasonalAnomalyTransform(cycle="month")
        data1 = self.ds_T["T"]
        data1_T = sat1.fit_transform(data1)
        ds2 = clif.preprocessing.remove_cyclical_trends(
            data=self.ds_T,
            variable="T",
            cycle="month",
            new_variable_suffix="",
            use_groupby=False,
        )
        data2_T = ds2["T_"]
        error = np.sum((data1_T.values - data2_T.values) ** 2)
        assert (
            error == 0.0
        ), "Seasonal detrending not doing what it is supposed to be doing. "


class TestMarginalizeTransform(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        # load data sets
        AERO_FILE = os.path.join(relpath, "tests/data/AEROD_v_198501_199612.nc")
        QRL_FILE = os.path.join(relpath, "tests/data/QRL.nc")
        T_FILE = os.path.join(relpath, "tests/data/Temp.nc")
        AREA_FILE = os.path.join(relpath, "tests/data/area_weights.nc")
        self.ds_AERO = xr.open_dataset(AERO_FILE, chunks={"time": 1})
        self.ds_QRL = xr.open_dataset(QRL_FILE, chunks={"time": 1})
        self.ds_T = xr.open_dataset(T_FILE, chunks={"time": 1})
        self.area_weights = xr.open_dataset(AREA_FILE)["area"]

    def test_marginalize_fails_for_xarray_datasets(self):
        mot = clif.preprocessing.MarginalizeOutTransform(coords=["lon"])
        data = self.ds_T["T"]
        with self.assertRaises(AssertionError):
            mot.fit(self.ds_T)

    def test_marginalize_out_unweighted_make_sure_coord_is_removed(self):
        mot = clif.preprocessing.MarginalizeOutTransform(coords=["lon"])
        data = self.ds_T["T"]
        data_transformed = mot.fit_transform(data)
        print(data_transformed.dims)
        with self.assertRaises(KeyError):
            for key in data.dims:
                data_transformed[key]

    def test_marginalize_out_unweighted_check_lon(self):
        mot = clif.preprocessing.MarginalizeOutTransform(coords=["lon"])
        data = self.ds_T["T"]
        data_transformed = mot.fit_transform(data)
        error = data_transformed.values - data.mean(dim="lon").values
        assert (
            np.sum(error**2) == 0
        ), "marginalization over longitude, unweighted not working."

    def test_marginalize_out_unweighted_check_lat(self):
        mot = clif.preprocessing.MarginalizeOutTransform(coords=["lat"])
        data = self.ds_T["T"]
        data_transformed = mot.fit_transform(data)
        error = data_transformed.values - data.mean(dim="lat").values
        assert (
            np.sum(error**2) == 0
        ), "marginalization over latitude, unweighted not working."

    def test_marginalize_out_unweighted_check_lat_lon(self):
        mot = clif.preprocessing.MarginalizeOutTransform(coords=["lat", "lon"])
        data = self.ds_T["T"]
        data_transformed = mot.fit_transform(data)
        error = data_transformed.values - data.mean(dim=["lat", "lon"]).values
        relerror = np.sum(error**2) / np.sum(
            data.mean(dim=["lat", "lon"]).values ** 2
        )
        assert (
            relerror <= 1e-14
        ), "Check method. We integrate out each dimension at a time."

    def test_marginalize_out_weighted_check_lat_lon_without_normalization(self):
        data = self.ds_T["T"]
        area_weight = self.area_weights
        area_weight_norm = area_weight.copy(deep=True)
        area_weight /= np.sum(area_weight)
        mot = clif.preprocessing.MarginalizeOutTransform(
            coords=["lat", "lon"], lat_lon_weights=area_weight
        )
        data_transformed = mot.fit_transform(data)
        ref = (data * area_weight_norm).sum(dim=["lat", "lon"])
        error = np.sum((data_transformed.values - ref.values) ** 2)
        relerror = error / np.sum(ref.values**2)
        assert (
            relerror <= 1e-14
        ), "Check method. We integrate out each dimension at a time."

    def test_marginalize_out_weighted_check_lat_lon(self):
        data = self.ds_T["T"]
        area_weight = self.area_weights
        area_weight /= np.sum(area_weight)
        mot = clif.preprocessing.MarginalizeOutTransform(
            coords=["lat", "lon"], lat_lon_weights=area_weight
        )
        data_transformed = mot.fit_transform(data)
        ref = (data * area_weight).sum(dim=["lat", "lon"])
        error = np.sum((data_transformed.values - ref.values) ** 2)
        relerror = error / np.sum(ref.values**2)
        assert (
            relerror <= 1e-14
        ), "Check method. We integrate out each dimension at a time."

    def test_marginalize_out_weighted_check_lat_only(self):
        data = self.ds_T["T"]
        area_weight = self.area_weights
        area_weight /= np.sum(area_weight)
        area_weight_lat = area_weight.mean(dim=["lon"])
        area_weight_lat /= np.sum(area_weight_lat)
        mot = clif.preprocessing.MarginalizeOutTransform(
            coords=["lat"], lat_lon_weights=area_weight
        )
        data_t = mot.fit_transform(data)
        ref = (data * area_weight_lat).sum(dim=["lat"])
        error = np.sum((data_t.values - ref.values) ** 2)
        relerror = error / np.sum(ref.values**2)
        assert (
            relerror <= 1e-14
        ), "Marginalizating out latitude with weights not working as expected."

    def test_marginalize_out_weighted_check_lon_only(self):
        """Should just be equivalent to mean even if weighted"""
        data = self.ds_T["T"]
        area_weight = self.area_weights
        area_weight /= np.sum(area_weight)
        area_weight_lon = area_weight.mean(dim=["lat"])
        area_weight_lon /= np.sum(area_weight_lon)
        mot = clif.preprocessing.MarginalizeOutTransform(
            coords=["lon"], lat_lon_weights=area_weight
        )
        data_t = mot.fit_transform(data)
        ref = data.mean(dim=["lon"])
        error = np.sum((data_t.values - ref.values) ** 2)
        relerror = error / np.sum(ref.values**2)
        print(relerror)
        assert (
            relerror <= 1e-14
        ), "Marginalizating out latitude with weights not working as expected."

    def test_marginalize_out_weights_dims_must_be_length_2(self):
        """Should just be equivalent to mean even if weighted"""
        data = self.ds_T["T"]
        area_weight = self.area_weights
        area_weight /= np.sum(area_weight)
        area_weight_lon = area_weight.mean(dim=["lat"])
        area_weight_lon /= np.sum(area_weight_lon)
        mot = clif.preprocessing.MarginalizeOutTransform(
            coords=["lon"], lat_lon_weights=area_weight_lon
        )
        with self.assertRaises(AssertionError):
            mot.fit(data)

    def test_marginalize_out_weights_dims_must_be_subset_of_data_dims(self):
        """Should just be equivalent to mean even if weighted"""
        data = self.ds_T["T"]
        area_weight = self.area_weights
        area_weight = area_weight.rename({"lat": "latitude", "lon": "longitude"})
        mot = clif.preprocessing.MarginalizeOutTransform(
            coords=["lon"], lat_lon_weights=area_weight
        )
        with self.assertRaises(AssertionError):
            mot.fit(data)


# # using polyfit native to xarray (much slower)
# lindetrend = clif.preprocessing.linear_detrending()
# data_new = lindetrend.fit_transform(data)


# # ########################################## CLASS TESTS
# clipT = clif.preprocessing.clip(dims=['lat','lon'],bounds=[(-60,60),(150,np.inf)])
# data_new = clipT.fit_transform(data)
# print(data_new.values.shape)

# mt = clif.preprocessing.marginalize(coords=['lon'],lat_lon_weighted=True,lat_lon_weights=ds.area)
# data_new = mt.fit_transform(data)
