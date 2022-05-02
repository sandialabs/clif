from tqdm import tqdm
import numpy as np
import xarray
from abc import ABC, abstractmethod
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss

"""
To do: signal detrending and removal of nan variables. 
"""


class TransformerMixin(ABC):
    """Abstract base class for all transformers

    Templated base class for all transformers used herein. Users must define a fit and transform method only and the rest is done by the base class.

    Methods
    -------
    fit:
        Perform calculations for the preprocessing, e.g. mean computation.
    transform:
        transform the data
    """

    @abstractmethod
    def fit(self, data, y=None, **fit_params):
        """User defined fit function (required).

        Parameters
        ----------
        data : xarray.DataArray

        Returns
        -------
        self: object
            Returns self oject

        """
        pass

    @abstractmethod
    def transform(self, data):
        """User defined transform function (required).

        Parameters
        ----------
        data : xarray.DataArray

        Returns
        -------
        data_new: xarray.DataArray
            returns new data array possibly of different size and dimensions.

        """
        pass

    def fit_transform(self, data, y=None, **fit_params):
        """
        Runs the fit and transform methods in one call (not required)
        """
        return self.fit(data, **fit_params).transform(data)


class SeasonalAnomalyTransform(TransformerMixin):
    """Removes cyclical trends from xarray time-series data

    Parameters
    ----------
    cycle: {'day', 'month', 'year'}, default='month'
        De-trending cycle resolution used to mean center the data.

    Methods
    -------
    fit:
        Perform calculations for the preprocessing, e.g. mean computation.
    transform:
        transform the data

    Examples
    --------
    >>> import xarray
    >>> from clif.preprocessing import SeasonalAnomalyTransform
    >>> X = xarray.open_dataarray('Temperature.nc')
    >>> anomalyT = SeasonalAnomalyTransform(cycle='month')
    >>> anomalyT.fit(X)
    >>> X_transformed = anomalyT.transform(X)

    """

    def __init__(self, cycle="month", group_mean=None):
        """
        Parameters
        ----------
        cycle: str (default = 'month')
                string representing the seasonal detrending resolution, e.g., month, day, hour, year, etc.
        """
        self.cycle = cycle
        self.group_mean = group_mean

    def fit(self, data, y=None, **fit_params):
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
        if self.group_mean is None:
            self.mu_by_group_ = data.groupby("time." + self.cycle).mean(dim="time")
        else:
            self.mu_by_group_ = self.group_mean.copy()
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
    """Clips a dimension according to a specific range or value from xarray time series data

    Transform to extract slices or subsets of the data

    Methods
    -------
    fit:
        Perform calculations for the preprocessing, e.g. mean computation.
    transform:
        transform the data

    Parameters
    ----------
    dims: list(str)
        List of dimensions to clip or slice, e.g., dims=['lat','lon']

    bounds: list(tuples)
        List of 2-length tuples corresponding to each dimension, e.g., bounds=[(-60,60),(0,180)]

    Examples
    --------
    >>> import xarray
    >>> import numpy as np
    >>> from clif.preprocessing import ClipTransform
    >>> X = xarray.open_dataarray('Temperature.nc')
    >>> clipT = ClipTransform(dims=['lat','plev'],bounds=[(-60,60),(5000,np.inf)])
    >>> X_transformed = clipT.fit_transform(X)
    """

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

    def fit(self, data, y=None, **fit_params):
        """
        Parameters
        ----------
        data: xarray.DataArray
                data array object holding the time-series data
        """
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
        """
        Parameters
        ----------
        data: xarray.DataArray
                data array object holding the time-series data
        """
        data_new = data.where(self.mask, drop=self.drop)
        return data_new


class MarginalizeOutTransform(TransformerMixin):
    """Integrate out or marginalize over dimensions to average over effect

    Parameters
    ----------
    coords: list(str)
        list of strings corresponding to the dimensions

    lat_lon_weights: None (default) or xarray.DataArray
        weights corresponding to lat/ lon grid. Must match the input data array coordinates.

    Methods
    -------
    fit:
        Perform calculations for the preprocessing, e.g. mean computation.
    transform:
        transform the data

    Examples
    --------
    >>> import xarray
    >>> import numpy as np
    >>> from clif.preprocessing import MarginalizeOutTransform
    >>> X = xarray.open_dataarray('Temperature.nc')
    >>> marginalizeT = MarginalizeOutTransform(dims=['lat','lon']])
    >>> X_transformed = marginalizeT.fit_transform(X)
    """

    def __init__(self, dims, lat_lon_weights=None):
        self.coords = dims
        self.lat_lon_weights = lat_lon_weights

    def _check_lat_lon_weights(self, data, lat_lon_weights):
        assert set(lat_lon_weights.dims) <= set(
            data.dims
        ), "Area weight dimensions are not a subset of the data dimensions"
        assert (
            len(set(data.dims).intersection(set(lat_lon_weights.dims))) == 2
        ), "For now the lat lon weights must be 2 dimensional lat by lon. We internally marginalize to find the lon and lat weights respectively. No need to do it before hand."
        for dim, size in lat_lon_weights.sizes.items():
            assert (
                data.sizes[dim] == size
            ), "lat and lon area weights must be the same size as the data lat lon sizes. names must also match."

    def fit_lat_only_weights(self, data, y=None, **fit_params):
        """For fitting only the latitude weights. Not integrated with fit_transform


        Parameters
        ----------
        data : _type_
            _description_
        y : _type_, optional
            _description_, by default None
        """
        if self.lat_lon_weights is not None:
            # self._check_lat_lon_weights(data, self.lat_lon_weights)
            # get normalized weights
            self.weight_dims = self.lat_lon_weights.dims
            weights = self.lat_lon_weights.copy()
            weights /= np.sum(weights)  # normalize

            # only one dimension
            self.weight_dict = {
                self.weight_dims[0]: weights,
            }

    def fit(self, data, y=None, **fit_params):
        assert isinstance(
            data, xarray.DataArray
        ), "Input must be an xarray DataArray object."
        # get lat lon weights if possible
        if self.lat_lon_weights is not None:
            # check lat lon weights
            self._check_lat_lon_weights(data, self.lat_lon_weights)
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
            if self.lat_lon_weights is None:
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


class Transpose(TransformerMixin):
    """Simple wrapper to transpose the data array

    This transform allows the user to return the data array in any order of dimensions that is specified.

    Parameters
    ----------
    dims: list(str)
        list of strings corresponding to the dimensions that you want returned, in that particular order.

    Methods
    -------
    fit:
        Perform calculations for the preprocessing, e.g. mean computation.
    transform:
        transform the data

    Examples
    --------
    >>> import xarray
    >>> import numpy as np
    >>> from clif.preprocessing import Transpose
    >>> X = xarray.open_dataarray('Temperature.nc')
    >>> transpose = Transpose(dims=['time','plev','lat','lon']])
    >>> X_transformed = transpose.fit_transform(X)
    """

    def __init__(self, dims):
        self.dims = dims  # order of dimensions you want to return

    def fit(self, data, y=None, **fit_params):
        assert len(data.dims) == len(
            self.dims
        ), "Order of dimensions must be the same as the total dimensions."
        return self

    def transform(self, data):
        return data.transpose(*self.dims)


class FlattenData(TransformerMixin):
    """Flatten or stack dimensions together

    Stacking or flattening of user-specified dimensions. The stacked dimension name is the concatenated name of the dimensions.

    Parameters
    ----------
    dims: list(str)
        Dimensions to stack or flatten together.

    Methods
    -------
    fit:
        Perform calculations for the preprocessing, e.g. mean computation.
    transform:
        transform the data

    Examples
    --------
    >>> import xarray
    >>> import numpy as np
    >>> from clif.preprocessing import FlattenData
    >>> X = xarray.open_dataarray('Temperature.nc')
    >>> flattenT = FlattenData(dims=['lat','lon']])
    >>> X_transformed = flattenT.fit_transform(X)
    """

    def __init__(self, dims):
        """Flatten data"""
        self.dims = dims  # must be a list

    def fit(self, data, y=None, **fit_params):
        """Gets the list of spatial dimensions, i.e. the given DataArray's dimensions without time."""
        self.new_dim_name = "_".join(self.dims)
        return self

    def transform(self, data):
        """Flattens the given DataArray's spatial dimensions into a matrix.

        Parameters
        ----------
        data : xarray.DataArray
            The xarray DataArray to flatten.

        Returns
        -------
        xarray.DataArray
            The flattened DataArray.
        """
        if len(self.dims) == 1:
            return data
        else:
            return data.stack({self.new_dim_name: self.dims})


class LinearDetrendTransform(TransformerMixin):
    """Remove linear trend for every grid point

    For every grid point we compute a linear time-series trend and subtract from the signal.

    Parameters
    ----------
    degree: int, default=1
        Degree of polynomial used in de-trending. Each grid point is fit with a monomial.

    dim: str, default='time'
        Dimension of the x-axis for computing the linear trend, i.e., dependent variable.

    Methods
    -------
    fit:
        Perform calculations for the preprocessing, e.g. mean computation.
    transform:
        transform the data

    Examples
    --------
    >>> import xarray
    >>> import numpy as np
    >>> from clif.preprocessing import LinearDetrendTransform
    >>> X = xarray.open_dataarray('Temperature.nc')
    >>> lindetrendT = LinearDetrendTransform()
    >>> X_transformed = lindetrendT.fit_transform(X)
    """

    def __init__(self, degree=1, dim="time"):
        self.degree = degree
        self.dim = "time"

    def fit(self, data, y=None, **fit_params):
        """For each time series, learn a best fit line via least squares"""
        reg = data.polyfit(dim="time", deg=1, full=True)
        self.coeff = reg.polyfit_coefficients
        self.lines = xarray.polyval(coord=data["time"], coeffs=self.coeff)
        return self

    def transform(self, data):
        assert hasattr(self, "lines"), "Must run fit first!"
        return data - self.lines

    def inverse_transform(self, data):
        return data + self.lines


# ********* Moved StationarityTesting to statistics.py module


class ScalerTransform(TransformerMixin):
    """Scaling transforms for data array object

    This transform scales the user-specified dimensions using different strategies. By default, the dimensions are none so that the scaling is done across all dimensions using a single scalar value. The three different options are scaling by the variance/ standard deviation, scaling to the unit cube (minmax), and standard scaler, which mean centers and scales by the standard deviation. All transforms are invertible.

    Parameters
    ----------
    dims : list(str), default = None
        List of dimensions. If None, all dimensions are combined to perform scaling.

    scale_type = {"variance", "minmax", "standard", "fixed"}, default = "variance"
        Type of scaling. Fixed means user specifies the mean and variance to scale the data.

    Methods
    -------
    fit:
        Perform calculations for the preprocessing, e.g. mean computation.
    transform:
        transform the data

    Examples
    --------
    >>> import xarray
    >>> import numpy as np
    >>> from clif.preprocessing import ScalerTransform
    >>> X = xarray.open_dataarray('Temperature.nc')
    >>> scaleT = ScalerTransform()
    >>> X_transformed = scaleT.fit_transform(X)
    """

    def __init__(self, dims=None, scale_type="variance", mu=0.0, var=1.0):
        self.scale_type = scale_type
        self.dims = dims
        self.mu = mu
        self.var = var

    def fit(self, data, y=None, **fit_params):
        if self.dims is None:
            self.dims = data.dims
        if self.scale_type == "variance":
            self._fit_variance(data)
        elif self.scale_type == "minmax":
            self._fit_minmax(data)
        elif self.scale_type == "standard":
            self._fit_standard(data)
        elif self.scale_type == "fixed":
            self._fit_fixed(data)
        else:
            return NotImplementedError
        return self

    def _fit_fixed(self, data):
        self.mu_, self.sigma_ = self.mu, np.sqrt(self.var)

    def _fit_standard(self, data):
        """Mean center and scale by stnd deviation"""
        self.mu_ = data.mean(dim=self.dims)
        self.sigma_ = np.sqrt(data.var(dim=self.dims))

    def _fit_variance(self, data):
        """Rescale by variance"""
        self.sigma_ = np.sqrt(data.var(dim=self.dims))
        self.mu_ = 0.0

    def _fit_minmax(self, data):
        """Rescale to [0,1]"""
        m = data.min(dim=self.dims)
        M = data.max(dim=self.dims)
        self.mu_ = m.copy()
        self.sigma_ = M - m

    def transform(self, data):
        data_new = (data - self.mu_) / self.sigma_
        return data_new

    def inverse_transform(self, data):
        data_original = self.sigma_ * data + self.mu_
        return data_original


class SingleVariableSelector(TransformerMixin):
    """Select a single "Column" or variable form a data set input

    This is a transformer wrapping of selecting a single variable from an xarray data set. This is useful if you want to perform feature unions or combining different transforms in a single operation. This essentially selects a particular variable from the data set object.

    Parameters
    ----------
    variable : str
        Must specify a single variable that is included in the list of variables of the dataset.

    Methods
    -------
    fit:
        Perform calculations for the preprocessing, e.g. mean computation.
    transform:
        transform the data

    Examples
    --------
    >>> import xarray
    >>> import numpy as np
    >>> from clif.preprocessing import SingleVariableSelector
    >>> ds = xarray.open_dataset("some_dataset_file.nc")
    >>> colselectT = SingleVariableSelector(variable="T")
    >>> data = colselectT.fit_transform(ds)
    """

    def __init__(self, variable, inverse=False):
        self.variable = variable
        self.inverse = inverse  # store ds for inversion

    def fit(self, dataset, y=None, **fit_params):
        assert self.variable in dataset.variables, "Variable is not in data set. "
        self.ds_original = dataset.copy(deep=True)
        return self

    def transform(self, dataset):
        data = dataset[self.variable]
        assert isinstance(
            data, xarray.DataArray
        ), "Must be a single variable in order to return a data array object. "
        return data

    def inverse_transform(self, data):
        """Gives us back the data set"""
        assert hasattr(
            self, "ds_original"
        ), "Choose inverse=True to save the original dataset. "
        self.ds_original[self.variable] = data
        return self.ds_original


class CombineDataArrays(TransformerMixin):
    """Combines or stacks data horizontally assuming that the first dimension of all arrays are the same

    Parameters
    ----------

    """

    def __init__(self, keep_meta=True):
        self.keep_meta = keep_meta  # store data arrays

    def _check_array_sizes(self, dataarrays):
        for data in dataarrays:
            assert data.ndim == 2, "For now, each data array must be two dimensional"
        row_sizes = [data.shape[0] for data in dataarrays]
        assert all(
            [rs == row_sizes[0] for rs in row_sizes]
        ), "The first dimension of each data array must be the same, since we stack relative to this axis."

    def fit(self, dataarrays):
        self.ndata = len(dataarrays)
        self._check_array_sizes(dataarrays)
        col_sizes = np.array([data.shape[1] for data in dataarrays])

        split_indices = [None] * (len(col_sizes) - 1)
        col_cumsum = np.cumsum(col_sizes)
        for i, split in enumerate(split_indices):
            split_indices[i] = int(col_cumsum[i])
        self.split_indices = split_indices

        if self.keep_meta:
            self.dataarrays_ = [data.copy(deep=True) for data in dataarrays]
            for data_ in self.dataarrays_:
                data_.values *= 0
        return self

    def transform(self, dataarrays):
        # convert to numpy arrays
        data_nparrays = [data.values for data in dataarrays]
        concatenated_arrays = np.concatenate(data_nparrays, axis=1)
        return concatenated_arrays

    def split(self, concatenated_array, return_np=False):
        split_arrays = np.split(
            concatenated_array, indices_or_sections=self.split_indices, axis=1
        )
        # make a local copy
        dataarrays_local_ = [data_.copy(deep=True) for data_ in self.dataarrays_]
        for i, data_local_ in enumerate(dataarrays_local_):
            data_local_.values = split_arrays[i]
        if return_np:
            return split_arrays
        else:
            return dataarrays_local_

    def inverse_transform(self, concatenated_array, return_np=False):
        return self.split(concatenated_array, return_np=return_np)


class StackDataArrays(TransformerMixin):
    """Alternative to Combine data arrays using the stack feature in xarray. Avoid having to convert to numpy arrays, but may be more inefficient.

    Parameters
    ----------
    TransformerMixin : _type_
        _description_
    """

    def __init__(self, dim="time", new_dim_name="combined"):
        self.dim = dim
        self.new_dim_name = new_dim_name

    def fit(self, dataarrays, y=None, **fit_params):
        # alternate to stacking
        ds = xarray.Dataset(dataarrays)
        self.data_vars = set(ds.keys())
        self.coords_dict = {}
        self.dims_dict = {}
        for vars in self.data_vars:
            self.coords_dict[vars] = ds[vars].coords
            self.dims_dict[vars] = ds[vars].dims
        return self

    def transform(self, dataarrays):
        ds = xarray.Dataset(dataarrays)
        data_stacked = ds.to_stacked_array(self.new_dim_name, [self.dim])
        return data_stacked

    def inverse_transform(self, dataarray):
        dim_name = self.new_dim_name
        data_unstacked = dataarray.to_unstacked_dataset(dim=dim_name)
        data_arrays_dict = {}
        for vars in self.data_vars:
            data_temp = data_unstacked[vars].dropna(dim=dim_name)
            data_temp = data_temp.drop_vars(dim_name)  # remove multindex
            data_temp = data_temp.rename({dim_name: self.dims_dict[vars][1]})
            data_temp = data_temp.assign_coords(self.coords_dict[vars])
            data_arrays_dict[vars] = data_temp
        # data_unstacked = xarray.Dataset(data_arrays_dict)
        return data_arrays_dict


class MultiObjectiveUnion(TransformerMixin):
    """Combine variables in a data set

    At this point, this class has potential memory issues given that we keep a copy of the dataset internally.

    Parameters
    ----------
    TransformerMixin : _type_
        _description_
    """

    def __init__(self, transformer_list, dim=None, n_jobs=None, inverse=True):
        self.transformer_list = transformer_list
        self.n_jobs = n_jobs
        self.inverse = inverse
        self.dim = dim

    def fit(self, dataset, y=None, **fit_params):
        if self.inverse:
            self.ds_ = dataset.copy(deep=True)
        if self.dim is None:
            self.dim = list(self.ds_.dims)[0]
        self.results = []
        self.variables = []
        for ii, transformer in enumerate(self.transformer_list):
            self.variables.append(transformer[0].variable)
            res_temp = transformer.fit_transform(dataset)
            self.results.append(res_temp)
            self.transformer_list[ii] = transformer  # update transformer list
        self._check_transform_results(self.results)

        return self

    def _check_transform_results(self, transform_results):
        size_along_dim = [t.sizes[self.dim] for t in transform_results]
        assert all([si == size_along_dim[0] for si in size_along_dim])

    def transform(self, dataset):
        self.res_ = [ri for ri in self.results]
        res_np = [ri.values for ri in self.results]
        col_sizes = [rii.shape[1] for rii in res_np]
        self.results_np = np.concatenate(res_np, axis=1)
        self.split_indices = [None] * (len(res_np) - 1)
        for i, split in enumerate(self.split_indices):
            self.split_indices[i] = int(np.cumsum(col_sizes[: i + 1]))
        return self.results_np

    def inverse_transform(self, data_np):
        assert hasattr(self, "ds_"), "Must store the orignal dataset"
        assert hasattr(self, "split_indices"), "must run the transform function"
        data_np_split = np.split(
            data_np, indices_or_sections=self.split_indices, axis=1
        )
        self.data_np_split = data_np_split
        for j, data_np in enumerate(data_np_split):
            self.res_[j].values = data_np
            ds_temp = self.transformer_list[j].inverse_transform(self.res_[j])
            # self.ds_ = self.transformer_list[-1][0].ds_original.copy(deep=True)
            self.ds_[self.variables[j]] = ds_temp[self.variables[j]]
        return self.ds_
