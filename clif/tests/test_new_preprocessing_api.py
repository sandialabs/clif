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

relpath, _ = os.path.split(clif.__file__)  # ignore the /__init__.py specification
print("\nrelative path:", relpath)


class TestSeasonalDetrending(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        # load data sets
        AERO_FILE = os.path.join(relpath, "tests/data/AEROD_v_198501_199612.nc")
        QRL_FILE = os.path.join(relpath, "tests/data/QRL.nc")
        T_FILE = os.path.join(relpath, "tests/data/Temp.nc")
        AREA_FILE = os.path.join(relpath, "tests/data/area_weights.nc")
        self.ds_AERO = xr.open_dataset(AERO_FILE)
        self.ds_QRL = xr.open_dataset(QRL_FILE)
        self.ds_T = xr.open_dataset(T_FILE)
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
        self.ds_AERO = xr.open_dataset(AERO_FILE)
        self.ds_QRL = xr.open_dataset(QRL_FILE)
        self.ds_T = xr.open_dataset(T_FILE)
        self.area_weights = xr.open_dataset(AREA_FILE)["area"]

    def test_marginalize_fails_for_xarray_datasets(self):
        mot = clif.preprocessing.MarginalizeOutTransform(dims=["lon"])
        data = self.ds_T["T"]
        with self.assertRaises(AssertionError):
            mot.fit(self.ds_T)

    def test_marginalize_out_unweighted_make_sure_coord_is_removed(self):
        mot = clif.preprocessing.MarginalizeOutTransform(dims=["lon"])
        data = self.ds_T["T"]
        data_transformed = mot.fit_transform(data)
        print(data_transformed.dims)
        with self.assertRaises(KeyError):
            for key in data.dims:
                data_transformed[key]

    def test_marginalize_out_unweighted_check_lon(self):
        mot = clif.preprocessing.MarginalizeOutTransform(dims=["lon"])
        data = self.ds_T["T"]
        data_transformed = mot.fit_transform(data)
        error = data_transformed.values - data.mean(dim="lon").values
        assert (
            np.sum(error ** 2) == 0
        ), "marginalization over longitude, unweighted not working."

    def test_marginalize_out_unweighted_check_lat(self):
        mot = clif.preprocessing.MarginalizeOutTransform(dims=["lat"])
        data = self.ds_T["T"]
        data_transformed = mot.fit_transform(data)
        error = data_transformed.values - data.mean(dim="lat").values
        assert (
            np.sum(error ** 2) == 0
        ), "marginalization over latitude, unweighted not working."

    def test_marginalize_out_unweighted_check_lat_lon(self):
        mot = clif.preprocessing.MarginalizeOutTransform(dims=["lat", "lon"])
        data = self.ds_T["T"]
        data_transformed = mot.fit_transform(data)
        error = data_transformed.values - data.mean(dim=["lat", "lon"]).values
        relerror = np.sum(error ** 2) / np.sum(
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
            dims=["lat", "lon"], lat_lon_weights=area_weight
        )
        data_transformed = mot.fit_transform(data)
        ref = (data * area_weight_norm).sum(dim=["lat", "lon"])
        error = np.sum((data_transformed.values - ref.values) ** 2)
        relerror = error / np.sum(ref.values ** 2)
        assert (
            relerror <= 1e-14
        ), "Check method. We integrate out each dimension at a time."

    def test_marginalize_out_weighted_check_lat_lon(self):
        data = self.ds_T["T"]
        area_weight = self.area_weights
        area_weight /= np.sum(area_weight)
        mot = clif.preprocessing.MarginalizeOutTransform(
            dims=["lat", "lon"], lat_lon_weights=area_weight
        )
        data_transformed = mot.fit_transform(data)
        ref = (data * area_weight).sum(dim=["lat", "lon"])
        error = np.sum((data_transformed.values - ref.values) ** 2)
        relerror = error / np.sum(ref.values ** 2)
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
            dims=["lat"], lat_lon_weights=area_weight
        )
        data_t = mot.fit_transform(data)
        ref = (data * area_weight_lat).sum(dim=["lat"])
        error = np.sum((data_t.values - ref.values) ** 2)
        relerror = error / np.sum(ref.values ** 2)
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
            dims=["lon"], lat_lon_weights=area_weight
        )
        data_t = mot.fit_transform(data)
        ref = data.mean(dim=["lon"])
        error = np.sum((data_t.values - ref.values) ** 2)
        relerror = error / np.sum(ref.values ** 2)
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
            dims=["lon"], lat_lon_weights=area_weight_lon
        )
        with self.assertRaises(AssertionError):
            mot.fit(data)

    def test_marginalize_out_weights_dims_must_be_subset_of_data_dims(self):
        """Should just be equivalent to mean even if weighted"""
        data = self.ds_T["T"]
        area_weight = self.area_weights
        area_weight = area_weight.rename({"lat": "latitude", "lon": "longitude"})
        mot = clif.preprocessing.MarginalizeOutTransform(
            dims=["lon"], lat_lon_weights=area_weight
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

# mt = clif.preprocessing.marginalize(dims=['lon'],lat_lon_weighted=True,lat_lon_weights=ds.area)
# data_new = mt.fit_transform(data)


class TestEOFSnapshot(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        # load data sets
        T_FILE = os.path.join(relpath, "tests/data/regression_test/Temperature.nc")
        T_FILE_REF = os.path.join(
            relpath, "tests/data/regression_test/Temperature_transformed.nc"
        )
        AREA_FILE = os.path.join(
            relpath, "tests/data/regression_test/lat_lon_weights.nc"
        )
        self.T = xr.open_dataarray(T_FILE)
        self.T_ref = xr.open_dataarray(T_FILE_REF)
        self.area_weights = xr.open_dataarray(AREA_FILE)

    def test_full_preprocessing_transforms(self):
        data = self.T
        data_transformed = self.T_ref
        lat_lon_weights = self.area_weights

        # clip latitude only for use in lat lon weighting
        clipLatT = clif.preprocessing.ClipTransform(
            dims=["lat"], bounds=[(-60.0, 60.0)]
        )
        lat_lon_weights_new = clipLatT.fit_transform(lat_lon_weights)

        # First clip the data
        clipT = clif.preprocessing.ClipTransform(
            dims=["lat", "plev"], bounds=[(-60.0, 60.0), (5000.0, np.inf)]
        )
        data_new = clipT.fit_transform(data)
        data_new_shape_true = (18, 15, 8, 24)
        assert (
            data_new.shape == data_new_shape_true
        ), "Clip transformed may have changed expected behavior."

        # detrend by month
        monthlydetrend = clif.preprocessing.SeasonalAnomalyTransform(cycle="month")
        data_new = monthlydetrend.fit_transform(data_new)

        # marginalize out lat and lon variables
        intoutT = clif.preprocessing.MarginalizeOutTransform(
            dims=["lat", "lon"], lat_lon_weights=lat_lon_weights_new
        )
        data_new = intoutT.fit_transform(data_new)
        assert (
            data_new.shape == data_new_shape_true[:2]
        ), "Marginal transform may have changed."

        # linear detrend by time
        lindetrendT = clif.preprocessing.LinearDetrendTransform()
        data_new = lindetrendT.fit_transform(data_new)

        # flatten data for EOF analysis
        flattenT = clif.preprocessing.FlattenData(dims=["plev"])
        data_new = flattenT.fit_transform(data_new)

        # return data in specific order using the Transpose transform
        transformT = clif.preprocessing.Transpose(dims=["time", "plev"])
        data_new = transformT.fit_transform(data_new)
        error = np.sum((data_new.values - data_transformed.values) ** 2)
        assert (
            error <= 1e-16
        ), "Transform operations may have changed so snapshot tests do not align. Doesn't mean it's wrong though."

    def test_full_eof_analysis_with_preprocessing_transforms(self):
        data = self.T
        data_transformed = self.T_ref
        lat_lon_weights = self.area_weights

        X = data_transformed.values

        # Now we can begin calculating the EOFs
        # obtain fingerprints
        n_components = 8
        fp = clif.fingerprints(n_eofs=n_components, varimax=False)
        fp.fit(X)

        # extract pca fingerprints and convergence diagnostics
        eofs_pca = fp.eofs_
        explained_variance_ratio = fp.explained_variance_ratio_

        evr_ref = np.array(
            [
                7.79464367e-01,
                1.81064479e-01,
                2.09833878e-02,
                1.07164463e-02,
                5.28720247e-03,
                1.81034729e-03,
                2.94594213e-04,
                2.19518543e-04,
            ]
        )

        error = np.sqrt(
            np.sum((explained_variance_ratio - evr_ref) ** 2)
            / np.sum((explained_variance_ratio) ** 2)
        )
        print(error)
        assert (
            error <= 1e-9
        ), "explained variance ratio relative error may have changed for regression test. "
