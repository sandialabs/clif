import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import xarray as xr

try:
    import cartopy.crs as ccrs
    from cartopy.util import add_cyclic_point
except:
    print("Cartopy is not installed so EOF plotting may not work.")


class fingerprints:
    """Class for fingerprinting and computing Empircal Orthogonal Functions (EOFs) from data

    Given a 2d data set we perform PCA and return the empirical orthogonal functions. The data set should be in form (n_samples, latitude x longitude), so that the second dimension is a combination of all latitude and longitude points.

    Parameters
    ----------
    n_eofs	: 	str, default = 2

            Number of empirical orthogonal functions to return, i.e. number of principal components

    whiten	: 	bool, default = False

            Whether to normalize the projection coefficients of the EOFs.

    varimax	:	bool, default = False

            Perform Varimax rotation after PCA is perform to obtain sparse entries

    Methods
    -------
    fit:
        Perform the SVD/ PCA for the given data matrix.
    transform:
        Project the data onto the components/ EOFs

    Examples
    --------
    >>> import xarray
    >>> import numpy as np
    >>> from clif import fingerprints
    >>> X = xarray.open_dataarray('Temperature.nc')
    >>> fp = fingerprints(n_eofs=8,whiten=True,varimax=False)
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
        n_eofs=2,
        varimax=False,
        method="pca",
        method_opts={"whiten": True, "svd_solver": "arpack"},
    ):
        self.n_eofs = n_eofs
        self.method_opts = method_opts
        self.varimax = varimax
        self.method = method

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X)
        return self.projections_

    def fit(self, X, y=None, **fit_params):
        """Perform EOF decomposition for a numpy array

        To do
        -----
        * Allow for xarray data set (n,lat,lon)
        * allow for list of data X = [X1,X2,...] for multivariate fingerprinting
        """
        # allow input as xarray dataarray object type
        if type(X) == xr.DataArray:
            assert (
                X.ndim == 2
            ), "data array object must only have 2 dimensions otherwise EOF calculation is ambiguous."
            X = X.values  # convert to numpy array
        # perform a check on the data
        assert X.ndim == 2, "Input data matrix is not 2 dimensional."
        self.n_samples, self.n_dim = X.shape
        if self.method == "pca":
            self._fit_pca(X)

    def _fit_pca(self, X):
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
        self.cumulative_explained_variance_ratio_ = np.cumsum(
            pca.explained_variance_ratio_
        )
        # compute total variance by reconstruction from sklearn's pca
        self.total_variance_ = (
            pca.singular_values_[0] ** 2 / self.n_samples
        ) / pca.explained_variance_ratio_[0]

        # compute varimax rotation is True
        if self.varimax == True:
            # this will overwrite what pca does so be careful
            self._fit_varimax(X)

    def _fit_varimax(self, X):
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

    def transform(self, X):
        X0 = X - np.mean(X, axis=0)
        projection_ = np.dot(X0, self.eofs_.T)
        if "whiten" in self.method_opts:
            if self.method_opts["whiten"] == True:
                sigma = np.sqrt(self.explained_variance_)
                projection_ = np.dot(projection_, np.diag(1.0 / sigma))
        return projection_

    def _ortho_rotation(self, componentsT, method="varimax", tol=1e-8, max_iter=1000):
        """Return rotated components (transpose).
        Here, componentsT are the transpose of the PCA components
        """
        assert (
            componentsT.shape[1] == self.n_eofs
        ), "Input shape must be the transpose of the pca components_ vector."
        nrow, ncol = componentsT.shape
        rotation_matrix = np.eye(ncol)
        var = 0

        for _ in range(max_iter):
            comp_rot = np.dot(componentsT, rotation_matrix)
            if method == "varimax":
                tmp = comp_rot * np.transpose((comp_rot ** 2).sum(axis=0) / nrow)
            elif method == "quartimax":
                tmp = 0
            u, s, v = np.linalg.svd(np.dot(componentsT.T, comp_rot ** 3 - tmp))
            rotation_matrix = np.dot(u, v)
            var_new = np.sum(s)
            if var != 0 and var_new < var * (1 + tol):
                break
            var = var_new

        self.rotation_matrix = rotation_matrix

        return np.dot(componentsT, rotation_matrix).T

    def plot_field(
        self,
        eof_to_print: int,
        lats: list,
        lons: list,
        cmap: matplotlib.colors.Colormap = plt.get_cmap("cividis"),
        ax: matplotlib.pyplot.axes = None,
        title: str = "",
        grid: bool = False,
        colorbar_title: bool = "",
        grid_kwargs: dict = {},
    ) -> tuple:
        """Plots a given fingerprint's EOFs as a 2-dimensional field on a latitude by longitude grid.

        Parameters
        ----------
        eof_to_print : int
            The specific EOF to print in order of variance explained in the un-rotated set.
        lats : list
            List of latitude values.
        lons : list
            List of longitude values.
        cmap : matplotlib.pyplot.cmap, optional
            A given colormap, by default plt.get_cmap("jet")
        ax : matplotlib.pyplot.axes, optional
            Given matplotlib axes, by default None
        title : str, optional
            Title of plot, by default ""
        grid : bool, optional
            Select whether a lattitude by longtiude grid is plotted, by default False
        colorbar_title : str, optional
            Title of colorbar, by default ""
        grid_kwargs : dict, optional
            Alternative arguments for the grid layout, by default {}

        Returns
        -------
        tuple
            A tuple of the plot's figure, axes, and colorbar objects.
        """
        EOF_recons = np.reshape(self.eofs_[eof_to_print], (len(lats), len(lons)))
        data = EOF_recons  # [eof_to_print, :, :]

        if not ax:
            f = plt.figure(figsize=(8, (data.shape[0] / float(data.shape[1])) * 8))
            ax = plt.axes(projection=ccrs.PlateCarree())

        data, lons = add_cyclic_point(data, coord=lons)
        pl = plt.contourf(
            lons, lats, data, cmap=cmap, extend="both", transform=ccrs.PlateCarree()
        )
        ax.coastlines()
        _colorbar = plt.colorbar(pl, label=colorbar_title)
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

        return f, ax, _colorbar
