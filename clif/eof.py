import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import xarray as xr
from typing import List, Tuple, Optional, Union, Any
from numpy.typing import NDArray

try:
    import cartopy.crs as ccrs
    from cartopy.util import add_cyclic_point
except:
    print("Cartopy is not installed so EOF plotting may not work.")


class fingerprints:
    """Class for fingerprinting and computing Empirical Orthogonal Functions (EOFs) from data

    Given a 2d data set we perform PCA and return the empirical orthogonal functions. The data set should be in form (n_samples, latitude x longitude), so that the second dimension is a combination of all latitude and longitude points.

    Parameters
    ----------
    n_eofs : int, default = 2
        Number of empirical orthogonal functions to return, i.e. number of principal components

    varimax : bool, default = False
        Perform Varimax rotation after PCA is performed to obtain sparse entries

    sort_by_lon : bool, default = False
        Sort EOFs by the longitude of their centroids after computing

    reverse_lon_sort : bool, default = True
        If sort_by_lon is True, whether to sort in descending order (west to east, True)
        or ascending order (east to west, False)

    standardize_eofs : bool, default = False
        Whether to standardize the signs of EOFs so that the maximum absolute value
        in each EOF is positive. If True, flips the sign of any EOF where the
        element with the largest magnitude is negative, and also flips the
        corresponding projections to maintain mathematical consistency.

    lat : NDArray, optional, default = None
        Latitude values to use for centroid calculation when sorting

    lon : NDArray, optional, default = None
        Longitude values to use for centroid calculation when sorting

    method : str, default = "pca"
        Method used for EOF calculation

    method_opts : dict, default = {"whiten": True, "svd_solver": "arpack"}
        Additional options for the method

    verbose : bool, default = False
        Whether to print additional information

    Attributes
    ----------
    eofs_ : NDArray
        The computed EOF patterns (n_eofs, n_features)

    projections_ : NDArray
        Projections of the input data onto the EOFs (n_samples, n_eofs)

    explained_variance_ : NDArray
        The amount of variance explained by each EOF

    explained_variance_ratio_ : NDArray
        The percentage of variance explained by each EOF

    cumulative_explained_variance_ratio_ : NDArray
        Cumulative sum of explained variance ratios

    Methods
    -------
    fit:
        Perform the SVD/ PCA for the given data matrix.
    transform:
        Project the data onto the components/ EOFs
    sort_eofs_by_longitude:
        Sort EOFs by the longitude of their centroids
    plot_field:
        Plot a given EOF as a 2D field on a latitude-longitude grid

    Examples
    --------
    >>> import xarray
    >>> import numpy as np
    >>> from clif import fingerprints
    >>> X = xarray.open_dataarray('Temperature.nc')
    >>> lat = X.lat.values
    >>> lon = X.lon.values
    >>> fp = fingerprints(n_eofs=8, whiten=True, varimax=False, sort_by_lon=True, lat=lat, lon=lon)
    >>> fp.fit(X)
    >>> print(fp.eofs_, fp.projections_, fp.explained_variance_ratio_)

    Notes
    -----
    For now, the PCA will use an automatic solver, with default parameters in sklearn's PCA. In general, we allow the user to specify a different method.

    To do
    -----
    * add option for using xarray as input
    * add option for multiple outputs
    * add option for factor analysis and other methods
    * implement varimax procedure into code using the factor analysis toolbox in sklearn
    """

    def __init__(
        self,
        n_eofs: int = 2,
        varimax: bool = False,
        sort_by_lon: bool = False,
        reverse_lon_sort: bool = True,
        standardize_eofs: bool = False,
        lat: Optional[NDArray] = None,
        lon: Optional[NDArray] = None,
        method: str = "pca",
        method_opts: dict = {"whiten": True, "svd_solver": "arpack"},
        center_method: str = "max_abs",
        verbose: bool = False,
    ) -> None:
        """Initialize the fingerprints class with the specified parameters.

        Parameters
        ----------
        n_eofs : int, default = 2
            Number of empirical orthogonal functions to return, i.e. number of principal components
        varimax : bool, default = False
            Perform Varimax rotation after PCA is performed to obtain sparse entries
        sort_by_lon : bool, default = False
            Sort EOFs by the longitude of their centers after computing
        reverse_lon_sort : bool, default = True
            If sort_by_lon is True, whether to sort in descending order (west to east, True)
            or ascending order (east to west, False)
        standardize_eofs : bool, default = False
            Whether to standardize the signs of EOFs so that the maximum absolute value
            in each EOF is positive. If True, flips the sign of any EOF where the
            element with the largest magnitude is negative, and also flips the
            corresponding projections to maintain mathematical consistency.
        lat : Optional[NDArray], default = None
            Latitude values to use for center calculation when sorting
        lon : Optional[NDArray], default = None
            Longitude values to use for center calculation when sorting
        method : str, default = "pca"
            Method used for EOF calculation
        method_opts : dict, default = {"whiten": True, "svd_solver": "arpack"}
            Additional options for the method
        center_method : str, default = "centroid"
            Method to use for finding EOF centers for longitude sorting.
            Options: "centroid" (weighted average) or "max_abs" (maximum absolute value)
        verbose : bool, default = False
            Whether to print additional information
        """
        self.n_eofs = n_eofs
        self.method_opts = method_opts
        self.varimax = varimax
        self.method = method
        self.sort_by_lon = sort_by_lon
        self.reverse_lon_sort = reverse_lon_sort
        self.standardize_eofs = standardize_eofs
        self.lat = lat
        self.lon = lon
        self.center_method = center_method
        self.centers = None
        self.verbose = verbose

    def fit_transform(self, X: NDArray, y: Optional[NDArray] = None, **fit_params) -> NDArray:
        """Fit the model to data and transform in one step.

        Equivalent to calling `fit(X)` followed by `transform(X)` but more efficient.

        Parameters
        ----------
        X : NDArray
            Input data matrix (n_samples, n_features)
        y : Optional[NDArray], default = None
            Ignored (for scikit-learn compatibility)
        **fit_params : dict
            Additional parameters passed to the fitting method

        Returns
        -------
        NDArray
            Transformed data (projections of X onto EOFs)
        """
        self.fit(X)
        return self.projections_

    def fit(self, X: NDArray, y: Optional[NDArray] = None, **fit_params) -> "fingerprints":
        """Perform EOF decomposition for the input data.

        Computes the EOFs, their projections, and explained variance ratios.
        If sort_by_lon is True and lat/lon are provided, sorts the EOFs by longitude.

        Parameters
        ----------
        X : NDArray
            Input data matrix (n_samples, n_features)
        y : Optional[NDArray], default = None
            Ignored (for scikit-learn compatibility)
        **fit_params : dict
            Additional parameters passed to the fitting method

        Returns
        -------
        fingerprints
            The fitted estimator (self)

        Notes
        -----
        If X is an xarray.DataArray, it will be converted to a numpy array.
        The input must be 2-dimensional.
        """
        # allow input as xarray dataarray object type
        if type(X) == xr.DataArray:
            assert X.ndim == 2, "data array object must only have 2 dimensions otherwise EOF calculation is ambiguous."
            X = X.values  # convert to numpy array
        # perform a check on the data
        assert X.ndim == 2, "Input data matrix is not 2 dimensional."
        self.n_samples, self.n_dim = X.shape
        if self.method == "pca":
            self._fit_pca(X)
        else:
            raise ValueError(f"Method '{self.method}' is not implemented. Currently only 'pca' is implemented.")

        # Sort by longitude if requested and lat/lon are provided
        if self.sort_by_lon:
            if self.lat is not None and self.lon is not None:
                if self.verbose:
                    print(f"Sorting EOFs by longitude (reverse={self.reverse_lon_sort})")
                self.sort_eofs_by_longitude(self.lat, self.lon, self.reverse_lon_sort, self.verbose)
            else:
                print("Warning: sort_by_lon=True but lat or lon not provided. Skipping sorting.")

    def _fit_pca(self, X: NDArray) -> None:
        """Perform PCA on the input data.

        Uses scikit-learn's PCA implementation with the specified options.
        Stores the results in the class attributes.

        Parameters
        ----------
        X : NDArray
            Input data matrix (n_samples, n_features)

        Notes
        -----
        This method is called by fit() when method="pca".
        If varimax=True, _fit_varimax() will be called after this method.
        """
        pca = PCA(n_components=self.n_eofs, **self.method_opts)
        self.pca = pca
        # get projection coefficients (n_samples x n_eofs)
        self.U_ = pca.fit_transform(X)
        self.projections_ = self.U_
        # get EOFs (each row is an EOF n_eofs x n_dim)
        self.eofs_ = pca.components_
        self.V_ = pca.components_
        # get singular values (square root of variances)
        self.S_ = pca.singular_values_
        # explained variance ratio for convergence plotting
        self.explained_variance_ = pca.explained_variance_
        self.explained_variance_ratio_ = pca.explained_variance_ratio_
        self.cumulative_explained_variance_ratio_ = np.cumsum(pca.explained_variance_ratio_)
        # compute total variance by reconstruction from sklearn's pca
        self.total_variance_ = (pca.singular_values_[0] ** 2 / self.n_samples) / pca.explained_variance_ratio_[0]

        # For regular PCA (non-varimax), standardize signs if requested
        if self.standardize_eofs and not self.varimax:
            self._standardize_eof_signs()

        # Compute varimax rotation if requested
        if self.varimax:
            self._fit_varimax(X)

            # For varimax-rotated EOFs, standardize signs if requested
            # This must be done after rotation since varimax affects the signs
            if self.standardize_eofs:
                self._standardize_eof_signs()

    def _fit_varimax(self, X: NDArray) -> None:
        """Apply Varimax rotation to the PCA results.

        Modifies the EOFs and projections to have a Varimax rotation,
        which tends to produce more spatially localized patterns.

        Parameters
        ----------
        X : NDArray
            Input data matrix (n_samples, n_features)

        Notes
        -----
        This method is called by _fit_pca() when varimax=True.
        It modifies eofs_, V_, U_, projections_, explained_variance_,
        and explained_variance_ratio_ in place.

        Note that Varimax rotation may change the signs of the EOF patterns.
        If standardize_eofs=True is specified, sign standardization will be
        applied after the rotation.
        """
        # fit using varimax AFTER we fit using PCA.
        # compute varimax rotation is True
        self.eofs_ = self._ortho_rotation(componentsT=self.eofs_.T)
        self.V_ = self.eofs_
        # also compute the explained variance
        X0 = X - np.mean(X, axis=0)
        # recompute the projections and overwrite that of PCA
        self.U_ = np.dot(X0, self.eofs_.T)
        self.projections_ = self.U_
        self.explained_variance_ = np.var(self.U_, axis=0)
        self.explained_variance_ratio_ = self.explained_variance_ / self.total_variance_
        if "whiten" in self.method_opts:
            if self.method_opts["whiten"] == True:
                sigma = np.sqrt(self.explained_variance_)
                # scale each column of the projection by the stdev if whiten == True (default)
                self.U_ = np.dot(self.U_, np.diag(1.0 / sigma))
                self.projections_ = self.U_
        else:
            pass  # use projection computed above

    def transform(self, X: NDArray) -> NDArray:
        """Project new data onto the EOFs.

        Parameters
        ----------
        X : NDArray
            Input data matrix (n_samples, n_features)

        Returns
        -------
        NDArray
            Projections of X onto the EOFs (n_samples, n_eofs)

        Notes
        -----
        This method centers X before projection.
        If whiten=True, the projections are scaled by the standard deviations.
        """
        X0 = X - np.mean(X, axis=0)
        projection_ = np.dot(X0, self.eofs_.T)
        if "whiten" in self.method_opts:
            if self.method_opts["whiten"] == True:
                sigma = np.sqrt(self.explained_variance_)
                projection_ = np.dot(projection_, np.diag(1.0 / sigma))
        return projection_

    def _ortho_rotation(self, componentsT: NDArray, method: str = "varimax", tol: float = 1e-8, max_iter: int = 1000) -> NDArray:
        """Perform orthogonal rotation of the components.

        Return rotated components (transpose).
        Here, componentsT are the transpose of the PCA components

        Parameters
        ----------
        componentsT : NDArray
            Transpose of the components matrix (n_features, n_eofs)
        method : str, default = "varimax"
            Rotation method to use. Currently only "varimax" is implemented.
        tol : float, default = 1e-8
            Tolerance for convergence
        max_iter : int, default = 1000
            Maximum number of iterations

        Returns
        -------
        NDArray
            Rotated components (n_eofs, n_features)

        Notes
        -----
        This method implements the iterative algorithm for Varimax rotation.
        The rotation matrix is stored in the rotation_matrix attribute.
        """
        assert componentsT.shape[1] == self.n_eofs, "Input shape must be the transpose of the pca components_ vector."
        nrow, ncol = componentsT.shape
        rotation_matrix = np.eye(ncol)
        var = 0

        for _ in range(max_iter):
            comp_rot = np.dot(componentsT, rotation_matrix)
            if method == "varimax":
                tmp = comp_rot * np.transpose((comp_rot**2).sum(axis=0) / nrow)
            elif method == "quartimax":
                tmp = 0
            u, s, v = np.linalg.svd(np.dot(componentsT.T, comp_rot**3 - tmp))
            rotation_matrix = np.dot(u, v)
            var_new = np.sum(s)
            if var != 0 and var_new < var * (1 + tol):
                break
            var = var_new

        self.rotation_matrix = rotation_matrix

        return np.dot(componentsT, rotation_matrix).T

    def _find_eof_centroids(self, eofs: List[NDArray], lat_grid: NDArray, lon_grid: NDArray) -> List[Tuple[float, float]]:
        """Find centroids of EOF patterns in lat/lon space.

        Calculates the weighted centroid of each EOF pattern,
        using the absolute values of the pattern as weights.

        Parameters
        ----------
        eofs : List[NDArray]
            List of EOF patterns, each reshaped to lat/lon grid (n_lat, n_lon)
        lat_grid : NDArray
            2D array of latitude values in a meshgrid (n_lat, n_lon)
        lon_grid : NDArray
            2D array of longitude values in a meshgrid (n_lat, n_lon)

        Returns
        -------
        List[Tuple[float, float]]
            List of (lat, lon) centroid coordinates for each EOF

        Notes
        -----
        The centroid is calculated as the weighted average of the grid coordinates,
        where the weights are the absolute values of the EOF pattern normalized to sum to 1.
        """
        centroids = []

        for i in range(len(eofs)):
            # Get the EOF pattern
            eof = eofs[i]

            # Use absolute values for centroid calculation
            eof_abs = np.abs(eof)

            # Normalize to use as weights
            weights = eof_abs / np.sum(eof_abs)

            # Calculate weighted centroid
            lat_centroid = np.sum(weights * lat_grid)
            lon_centroid = np.sum(weights * lon_grid)

            centroids.append((lat_centroid, lon_centroid))

        return centroids

    def _find_eof_centers_of_max(self, eofs: List[NDArray], lat_grid: NDArray, lon_grid: NDArray) -> List[Tuple[float, float]]:
        """Find centers of maximum absolute value in EOF patterns.

        Instead of calculating weighted centroids, this function identifies
        the location of the maximum absolute value in each EOF pattern.

        Parameters
        ----------
        eofs : List[NDArray]
            List of EOF patterns, each reshaped to lat/lon grid (n_lat, n_lon)
        lat_grid : NDArray
            2D array of latitude values in a meshgrid (n_lat, n_lon)
        lon_grid : NDArray
            2D array of longitude values in a meshgrid (n_lat, n_lon)

        Returns
        -------
        List[Tuple[float, float]]
            List of (lat, lon) coordinates of maximum absolute value for each EOF

        Notes
        -----
        For patterns with distinct positive and negative regions,
        this identifies the location of the strongest feature regardless of sign.
        """
        centers = []

        for i in range(len(eofs)):
            # Get the EOF pattern
            eof = eofs[i]

            # Find the location of maximum absolute value
            max_abs_idx = np.unravel_index(np.argmax(np.abs(eof)), eof.shape)

            # Get the lat/lon coordinates at that location
            lat_center = lat_grid[max_abs_idx]
            lon_center = lon_grid[max_abs_idx]

            centers.append((lat_center, lon_center))

        return centers

    def _find_eof_centers(self, eofs: List[NDArray], lat_grid: NDArray, lon_grid: NDArray) -> List[Tuple[float, float]]:
        """Find centers of EOF patterns based on the specified method.

        Dispatches to the appropriate method based on self.center_method.

        Parameters
        ----------
        eofs : List[NDArray]
            List of EOF patterns, each reshaped to lat/lon grid
        lat_grid : NDArray
            2D array of latitude values in a meshgrid
        lon_grid : NDArray
            2D array of longitude values in a meshgrid

        Returns
        -------
        List[Tuple[float, float]]
            List of (lat, lon) center coordinates for each EOF

        Raises
        ------
        ValueError
            If the specified center_method is not recognized
        """
        if self.center_method == "centroid":
            return self._find_eof_centroids(eofs, lat_grid, lon_grid)
        elif self.center_method == "max_abs":
            return self._find_eof_centers_of_max(eofs, lat_grid, lon_grid)
        else:
            raise ValueError(f"Unrecognized center method: {self.center_method}. " f"Supported methods are 'centroid' and 'max_abs'.")

    def sort_eofs_by_longitude(self, lat: NDArray, lon: NDArray, reverse: bool = True, verbose: bool = False) -> None:
        """Sort EOF patterns based on the longitude of their centers.

        Updates all relevant class attributes in-place to maintain the new order.
        The center coordinates are determined using the method specified in center_method.

        Parameters
        ----------
        lat : NDArray
            1D array of latitude values (n_lat)
        lon : NDArray
            1D array of longitude values (n_lon)
        reverse : bool, default = True
            If True, sort in descending order (west to east)
            If False, sort in ascending order (east to west)
        verbose : bool, default = False
            If True, print sorting information
        """
        # Create 2D grids of lat and lon values
        lon_grid, lat_grid = np.meshgrid(lon, lat)

        # Reshape all EOFs to lat/lon space
        eof_patterns = []
        for i in range(self.n_eofs):
            eof_pattern = np.reshape(self.eofs_[i], (len(lat), len(lon)))
            eof_patterns.append(eof_pattern)

        # Find centers for each EOF using the specified method
        self.centers = self._find_eof_centers(eof_patterns, lat_grid, lon_grid)

        # Create a list of (index, lon_center) tuples for sorting
        eof_lon_indices = [(i, self.centers[i][1]) for i in range(self.n_eofs)]

        # Sort by longitude (reverse=True for descending order)
        sorted_eof_indices = sorted(eof_lon_indices, key=lambda x: x[1], reverse=reverse)
        sorted_indices = [idx for idx, _ in sorted_eof_indices]

        if verbose:
            center_type = "centroids" if self.center_method == "centroid" else "maximum absolute values"
            print(f"EOF {center_type} (lat, lon):", self.centers)
            print("EOFs sorted by longitude:", sorted_indices)

        # Create sorted versions of EOFs
        sorted_eofs = np.array([self.eofs_[idx] for idx in sorted_indices])

        # Sort projections
        sorted_projections = np.array([self.projections_[:, idx] for idx in sorted_indices]).T

        # Sort explained variance ratios
        sorted_explained_variance_ratio = np.array([self.explained_variance_ratio_[idx] for idx in sorted_indices])
        sorted_explained_variance = np.array([self.explained_variance_[idx] for idx in sorted_indices])

        # Update class attributes with sorted values
        self.eofs_ = sorted_eofs
        self.V_ = sorted_eofs
        self.projections_ = sorted_projections
        self.U_ = sorted_projections
        self.explained_variance_ratio_ = sorted_explained_variance_ratio
        self.explained_variance_ = sorted_explained_variance
        self.cumulative_explained_variance_ratio_ = np.cumsum(sorted_explained_variance_ratio)

        # Sort centers as well to match the new order
        self.centers = [self.centers[idx] for idx in sorted_indices]

    def _standardize_eof_signs(self) -> None:
        """Standardize the signs of EOFs based on maximum absolute value.

        For each EOF, identifies the element with the largest absolute magnitude.
        If this element has a negative value, flips the sign of the entire EOF
        and its corresponding projections.

        This ensures a consistent sign convention where the element with the
        largest magnitude in each EOF is always positive.

        Notes
        -----
        This method modifies the following attributes in-place:
        - eofs_
        - V_
        - projections_
        - U_
        """
        for i in range(self.n_eofs):
            # Find element with the largest absolute value
            abs_values = np.abs(self.eofs_[i])
            max_abs_idx = np.argmax(abs_values)
            max_abs_value = self.eofs_[i][max_abs_idx]

            if self.verbose:
                print(f"EOF {i+1} maximum absolute value: {abs_values[max_abs_idx]:.6f}")

            # If maximum absolute value is negative, flip the sign
            if max_abs_value < 0:
                if self.verbose:
                    print(f"Standardizing EOF {i+1}: Flipping sign")

                # Create new arrays with flipped signs
                new_eof = -1.0 * self.eofs_[i].copy()
                self.eofs_[i] = new_eof
                self.V_[i] = new_eof.copy()

                # Also flip the corresponding projections
                if hasattr(self, "projections_"):
                    new_proj = -1.0 * self.projections_[:, i].copy()
                    self.projections_[:, i] = new_proj

                    if hasattr(self, "U_"):
                        self.U_[:, i] = new_proj.copy()

    def plot_field(
        self,
        eof_to_print: int,
        lats: List[float],
        lons: List[float],
        cmap: matplotlib.colors.Colormap = plt.get_cmap("cividis"),
        extend_cmap: str = "neither",
        ax: Optional[matplotlib.pyplot.axes] = None,
        title: str = "",
        grid: bool = False,
        colorbar_title: str = "",
        grid_kwargs: dict = {},
        colorbar_orientation: str = "vertical",
        colorbar_kwargs: dict = {},
    ) -> Tuple[matplotlib.pyplot.axes, matplotlib.pyplot.colorbar]:
        """Plot a given EOF as a 2D field on a latitude-longitude grid.

        Creates a contour plot of the EOF pattern using Cartopy for map projection.

        Parameters
        ----------
        eof_to_print : int
            The index of the EOF to plot
        lats : List[float]
            List of latitude values
        lons : List[float]
            List of longitude values
        cmap : matplotlib.colors.Colormap, default = plt.get_cmap("cividis")
            Colormap to use for the plot
        extend_cmap : str, default = "neither"
            How to extend the colormap for out-of-range values
            Options: "neither", "both", "min", "max"
        ax : Optional[matplotlib.pyplot.axes], default = None
            Axes to plot on. If None, a new figure and axes are created.
        title : str, default = ""
            Title for the plot
        grid : bool, default = False
            Whether to add a grid to the plot
        colorbar_title : str, default = ""
            Title for the colorbar
        grid_kwargs : dict, default = {}
            Additional arguments for grid lines
        colorbar_orientation : str, default = "vertical"
            Orientation of the colorbar. Options: "vertical", "horizontal"
        colorbar_kwargs : dict, default = {}
            Additional arguments for colorbar formatting

        Returns
        -------
        Tuple[matplotlib.pyplot.axes, matplotlib.pyplot.colorbar]
            The axes and colorbar objects

        Notes
        -----
        This method requires Cartopy to be installed.
        The EOF pattern is reshaped to the lat/lon grid before plotting.
        """
        EOF_recons = np.reshape(self.eofs_[eof_to_print], (len(lats), len(lons)))
        data = EOF_recons

        if not ax:
            f = plt.figure(figsize=(8, (data.shape[0] / float(data.shape[1])) * 8))
            ax = plt.axes(projection=ccrs.PlateCarree())

        data, lons = add_cyclic_point(data, coord=lons)
        pl = plt.contourf(lons, lats, data, cmap=cmap, extend=extend_cmap, transform=ccrs.PlateCarree())
        ax.coastlines()

        # Adjust colorbar based on orientation
        colorbar_params = {"label": colorbar_title}
        colorbar_params.update(colorbar_kwargs)

        if colorbar_orientation == "horizontal":
            _colorbar = plt.colorbar(pl, ax=ax, orientation="horizontal", **colorbar_params)
            # Optionally adjust colorbar position for horizontal orientation
            if "pad" not in colorbar_kwargs:
                _colorbar.ax.set_position([ax.get_position().x0, ax.get_position().y0 - 0.10, ax.get_position().width, 0.03])  # Move below the map  # Height of colorbar
        else:
            _colorbar = plt.colorbar(pl, ax=ax, **colorbar_params)

        if grid:
            if grid_kwargs:
                ax.gridlines(**grid_kwargs)
            else:
                ax.gridlines(
                    draw_labels=True,
                    dms=True,
                    x_inline=False,
                    y_inline=False,
                    linestyle="--",
                    color="black",
                )

        ax.set_title(title, fontsize=16)
        return ax, _colorbar
