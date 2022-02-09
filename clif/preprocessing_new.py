from tqdm import tqdm
import numpy as np
import xarray
from abc import ABC, abstractmethod
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss



class TransformerMixin(ABC):
    """
	Base class for preprocessing transforms
	"""

    @abstractmethod
    def fit(self, data):
        pass

    @abstractmethod
    def transform(self, data):
        pass

    def fit_transform(self, data, **fit_params):
        """
        Runs the fit and transform methods in one call
        """
        return self.fit(data, **fit_params).transform(data)


class AnomalyTransform(TransformerMixin):
    """Removes cyclical trends from xarray time-series data"""

    def __init__(self, cycle="month"):
        """
        Parameters
        ----------
        cycle: str (default = 'month')
                string representing the seasonal detrending resolution, e.g., month, day, hour, year, etc.
        """
        self.cycle = cycle

    def fit(self, data):
        """
        Parameters
        ----------
        data: xarray.DataArray
                data array object holding the time-series data
        """
        assert isinstance(
            data, xarray.DataArray
        ), "Input must be an xarray DataArray object."
        # Compute the mean by grouping times
        self.mu_by_group_ = data.groupby("time." + self.cycle).mean(dim="time")
        return self

    def transform(self, data):
        """
        Parameters
        ----------
        data: xarray.DataArray
                data array object holding the time-series data
        """
        assert hasattr(self, "mu_by_group_"), "Must run self.fit() first."
        data_new = data.groupby("time." + self.cycle) - self.mu_by_group_
        return data_new

    def inverse_transform(self, data):
        assert hasattr(self, "mu_by_group_"), "Must run self.fit() first."
        return data.groupby("time." + self.cycle) + self.mu_by_group_


class ClipTransform(TransformerMixin):
    def __init__(self, dims, bounds, drop=True):
        self.dims = dims
        self.bounds = bounds
        self.drop = drop

    def _check(self, data):
        # check to make sure dimensions are indeed in the data array
        assert isinstance(data, xarray.DataArray), "Input must be a data array"
        assert isinstance(
            self.dims, list
        ), "dimensions must be provided as a list of strings"
        assert len(self.dims) == len(
            self.bounds
        ), "List of dimensions and bounds must match in length."
        for dim in self.dims:
            assert hasattr(data, dim), "DataArray does not have {0} dimension".format(
                dim
            )

    def fit(self, data):
        # check data to make sure dimensions are in there
        self._check(data)
        bnds_index = [None] * len(self.dims)
        for i, bnds in enumerate(self.bounds):
            if isinstance(bnds, (int, float, complex)):
                bnds_index[i] = data[self.dims[i]] == bnds
            elif isinstance(bnds, tuple):
                bnds_index[i] = (data[self.dims[i]] >= bnds[0]) & (
                    data[self.dims[i]] < bnds[1]
                )
            else:
                raise TypeError("bounds must be a list of numbers or tuples of size 2.")
        self.mask = bnds_index[0]
        if len(bnds_index) > 1:
            for i in range(1, len(self.dims) + 1):
                self.mask = self.mask & bnds_index[1]
        return self

    def transform(self, data):
        data_new = data.where(self.mask, drop=self.drop)
        return data_new


class MarginalizeTransform(TransformerMixin):
    def __init__(self, coords, lat_lon_weighted=False, lat_lon_weights=None):
        self.coords = coords
        self.lat_lon_weighted = lat_lon_weighted
        self.lat_lon_weights = lat_lon_weights

    def fit(self, data):
        # get lat lon weights if possible
        if self.lat_lon_weighted is True:
            assert (
                self.lat_lon_weights is not None
            ), "If weighted, you must provide the weights!"
            # get normalized weights
            self.weight_dims = self.lat_lon_weights.dims
            weights = self.lat_lon_weights.copy()
            weights /= np.sum(weights)  # normalize

            # weight dims are lat and lon, but not necessarily by that name
            weights_d1 = weights.mean(
                dim=self.weight_dims[1]
            )  # latitude weights (non-uniform)
            weights_d1 /= np.sum(weights_d1)
            weights_d2 = weights.mean(
                dim=self.weight_dims[0]
            )  # longitude weights (uniform)
            weights_d2 /= np.sum(weights_d2)
            self.weight_dict = {
                self.weight_dims[0]: weights_d1,
                self.weight_dims[1]: weights_d2,
            }
        return self

    def transform(self, data):
        # marginalize data array over marginal_coords
        for c in self.coords:
            if self.lat_lon_weighted is False:
                # no need for weighted sums is lat_lon_weighting is True
                data = data.mean(dim=c)
            else:
                # if lat_lon_weighting (default False), then only weight the latitude before summing over
                if c in self.weight_dims:
                    # must use area weighting if marginalizing over latitude
                    data = (data * self.weight_dict[c]).sum(dim=c)
                else:
                    # uniform average weighting for all other coordinates, including longitude
                    data = data.mean(dim=c)
        return data


class FlattenSpatialData(TransformerMixin):
    def fit(self, data):
        """Gets the list of spatial dimensions, i.e. the given DataArray's dimensions without time.
        """
        self.dims = []
        for dim in data.dims:
            if dim != "time":
                self.dims.append(dim)
        return self

    def transform(self, data):
        """Flattens the given DataArray's spatial dimensions into a matrix.

        Args:
            data (xarray.DataArray): The xarray DataArray to flatten.

        Returns:
            xarray.DataArray: The flattened DataArray.
        """
        return data.stack(dim=self.dims)


class LinearDetrendTransform(TransformerMixin):
    def __init__(self, degree=1, dim="time"):
        self.degree = degree
        self.dim = "time"

    def fit(self, data):
        """For each time series, learn a best fit line via least squares
		"""
        reg = data.polyfit(dim="time", deg=1, full=True)
        self.coeff = reg.polyfit_coefficients
        self.lines = xarray.polyval(coord=data["time"], coeffs=self.coeff)
        return self

    def transform(self, data):
        assert hasattr(self, "lines"), "Must run fit first!"
        return data - self.lines

    def inverse_transform(self, data):
        return data + self.lines


class StationarityTesting:
    """A class containing functions for determining the stationarity of a given time series.

    Stationarity means that the statistical properties of a time series i.e. mean, variance and covariance do not change over time.
    Many statistical models require the series to be stationary to make effective and precise predictions.

    Two statistical tests are used to check the stationarity of a time series - Augmented Dickey Fuller (“ADF”) test and Kwiatkowski-Phillips-Schmidt-Shin (“KPSS”) test.
    """

    def test_stationarity(time_series, p_value_threshold, verbosity=1):
        """Tests whether the given time series is stationary with respect to the given p-value threshold.

        This uses both the Augmented Dickey Fuller (“ADF”) test and Kwiatkowski-Phillips-Schmidt-Shin (“KPSS”) test to determine stationarity. These are hypothsis tests in which the null and alternate hypotheses are opposites.
        If the ADF test fails to reject the null hypothesis, this may provide evidence that the series is non-stationary.
        If the KPSS test fails to reject the null hypothesis, this may provide evidence that the series is stationary.

        There are four cases of stationarity in this test:
        Case 1: Both tests conclude that the series is not stationary - The series is not stationary
        Case 2: Both tests conclude that the series is stationary - The series is stationary
        Case 3: KPSS indicates stationarity and ADF indicates non-stationarity - The series is trend stationary. Trend needs to be removed to make series strict stationary. The detrended series is checked for stationarity.
        Case 4: KPSS indicates non-stationarity and ADF indicates stationarity - The series is difference stationary. Differencing is to be used to make series stationary. The differenced series is checked for stationarity.


        Args:
            time_series (numpy:array_like, 1d): The data to be tested.
            p_value_threshold (float): The hypothesis test threshold for the returned p-values.
            verbosity (int) (default=1): Determines the level of verbosity. 1 explains the results and 2 print the ADF and KPSS p-values.

        Returns:
            tuple: A pair of booleans (adf_stationary, kpss_stationary) corresponding to whether the ADF test returns stationary or the KPSS test returns stationary, respectively.
        """

        adf_p_val = adfuller(time_series, autolag="AIC")[1]
        kpss_p_val = kpss(time_series, regression="c", nlags="auto")[1]

        # adf_reject means stationary
        # kpss_reject means not stationary
        adf_reject = adf_p_val <= p_value_threshold
        kpss_reject = kpss_p_val <= p_value_threshold
        adf_stationary = adf_reject
        kpss_stationary = ~kpss_reject

        if verbosity > 0:
            if verbosity > 1:
                print("ADF p-value:", adf_p_val)
                print("KPSS p-value:", kpss_p_val)
            if adf_stationary and kpss_stationary:
                print("Series is stationary.")
            elif not adf_stationary and kpss_stationary:
                print(
                    "Series is trend stationary only. Trend needs to be removed to make the time series fully stationary."
                )
            elif adf_stationary and not kpss_stationary:
                print(
                    "Series is difference stationary only. Seasonal differencing can be used to make the time series fully stationary. "
                )
            else:
                print("Series is not stationary.")

        return adf_stationary, kpss_stationary


"""
To do: signal detrending and removal of nan variables. 
"""

