import clif
import unittest
import numpy as np
import warnings
from sklearn.metrics import mean_squared_error
from sklearn import datasets
import os
import time

relpath, _ = os.path.split(clif.__file__)  # ignore the /__init__.py specification
print("\nrelative path:", relpath)


def mse(a, b):
    return mean_squared_error(a, b, squared=False)


class TestFingerprintClass(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        # load data sets
        self.X = datasets.load_digits().data

    def test_number_of_eofs_in_init(self):
        # default constructor
        n_components = 8
        fp = clif.fingerprints(n_eofs=n_components)
        assert fp.n_eofs == 8, "Number of components not being set properly."
        return

    def test_pca_fit_producing_right_size(self):
        # default constructor
        n_components = 8
        fp = clif.fingerprints(n_eofs=n_components)
        fp.fit(self.X)
        n_samples, n_dim = self.X.shape
        assert fp.U_.shape == (
            n_samples,
            n_components,
        ), "Projection matrix is not the right size"
        assert fp.V_.shape == (
            n_components,
            n_dim,
        ), "Component matrix is not the right size"
        return

    def test_pca_fit_make_sure_eof_same_shape_as_component(self):
        # default constructor
        n_components = 8
        fp = clif.fingerprints(n_eofs=n_components)
        fp.fit(self.X)
        n_samples, n_dim = self.X.shape
        assert (
            fp.pca.components_.shape == fp.eofs_.shape
        ), "EOFs and PCA components must be the same shape"
        return

    def test_pca_fit_make_sure_eof_same_as_component(self):
        # default constructor
        n_components = 8
        fp = clif.fingerprints(n_eofs=n_components)
        fp.fit(self.X)
        n_samples, n_dim = self.X.shape
        assert (
            np.sum((fp.V_ - fp.eofs_) ** 2) <= 1e-16
        ), "EOFs should be n_dim by n_components"
        return

    def test_pca_fit(self):
        # default constructor
        n_components = 8
        fp = clif.fingerprints(n_eofs=n_components)
        fp.fit(self.X)
        assert hasattr(fp, "U_"), "Fit function is not producing a projection matrix."
        assert hasattr(fp, "S_"), "Fit function is not producing singular values."
        assert hasattr(fp, "V_"), "Fit function is not producing a component matrix."
        assert hasattr(fp, "eofs_"), "Fit function is not producing an eofs matrix."
        return

    def test_pca_explained_variance(self):
        n_samples, n_dim = self.X.shape
        # obtain fingerprints
        n_components = 8
        fp = clif.fingerprints(n_eofs=n_components, varimax=False)
        fp.fit(self.X)
        V_pca = fp.eofs_  # retrieve EOFs

        # calculate pca explained variance just to double check
        X0 = self.X - np.mean(self.X, axis=0)  # center data
        assert (
            X0.shape[1] == V_pca.shape[1]
        ), "eofs_ number of columns must be the same as X."
        U_pca = np.dot(X0, V_pca.T)  # project data
        pca_explained_variance = np.sum(U_pca**2, axis=0) / (n_samples - 1)
        relerror = np.linalg.norm(
            pca_explained_variance - fp.explained_variance_
        ) / np.linalg.norm(fp.explained_variance_)
        assert (
            relerror <= 1e-13
        ), "PCA explained variance is not being incorrectly computed."
        return

    def test_varimax_rotation_matrix_size(self):
        n_samples, n_dim = self.X.shape
        # obtain fingerprints
        n_components = 8
        fp = clif.fingerprints(n_eofs=n_components, varimax=True)
        fp.fit(self.X)
        V_varimax = fp.eofs_  # retrieve EOFs
        assert fp.rotation_matrix.shape == (
            n_components,
            n_components,
        ), "rotation matrix should a square matrix is size n_eofs x n_eofs. Make sure feed in the transpose of the pca.components_ into the _ortho_rotation function. "

    def test_pca_varimax_explained_variance(self):
        n_samples, n_dim = self.X.shape
        # obtain fingerprints
        n_components = 8
        fp = clif.fingerprints(n_eofs=n_components, varimax=True)
        fp.fit(self.X)
        V_varimax = fp.eofs_  # retrieve EOFs

        # calculate pca explained variance just to double check
        X0 = self.X - np.mean(self.X, axis=0)  # center data
        assert (
            X0.shape[1] == V_varimax.shape[1]
        ), "eofs_varimax_ number of columns must be the same as X."
        U_varimax = np.dot(X0, V_varimax.T)  # project data
        varimax_explained_variance = np.var(U_varimax, axis=0)
        relerror = np.linalg.norm(
            varimax_explained_variance - fp.explained_variance_
        ) / np.linalg.norm(fp.explained_variance_)
        assert (
            relerror <= 1e-13
        ), "PCA varimax explained variance is not being incorrectly computed."
        return

    def test_compare_pca_vs_varimax_explained_variance(self):
        n_samples, n_dim = self.X.shape
        # obtain fingerprints
        n_components = 8
        fp_varimax = clif.fingerprints(n_eofs=n_components, varimax=True)
        fp = clif.fingerprints(n_eofs=n_components, varimax=False)
        fp.fit(self.X)
        fp_varimax.fit(self.X)
        V_varimax = fp_varimax.eofs_  # retrieve EOFs
        V_pca = fp.eofs_  # EOFs without varimax

        # calculate pca explained variance just to double check
        X0 = self.X - np.mean(self.X, axis=0)  # center data
        U_pca = np.dot(X0, V_pca.T)  # project data
        pca_explained_variance = np.sum(U_pca**2, axis=0) / (n_samples - 1)
        U_varimax = np.dot(X0, V_varimax.T)  # project data
        varimax_explained_variance = np.var(U_varimax, axis=0)
        error = np.abs(
            np.sum(pca_explained_variance) - np.sum(varimax_explained_variance)
        )
        assert error < 1.0, "varimax should close to pca explained variance."
        return

    def test_compare_pca_vs_varimax_explained_variance_ratio(self):
        n_samples, n_dim = self.X.shape
        # obtain fingerprints
        n_components = 8
        fp_varimax = clif.fingerprints(n_eofs=n_components, varimax=True)
        fp = clif.fingerprints(n_eofs=n_components, varimax=False)
        fp.fit(self.X)
        fp_varimax.fit(self.X)

        error = np.linalg.norm(
            np.sum(fp_varimax.explained_variance_ratio_)
            - np.sum(fp.explained_variance_ratio_)
        )
        assert (
            error <= 1e-14
        ), "Explained variance ratio should be more or less the same between varimax and pca since the rotation is orthogonal."
        return


# import xarray as xr
# class TestFingerprintClassWithXarray(unittest.TestCase):
#   @classmethod
#   def setUpClass(self):
#       # load data sets
#       nc_file = relpath + '/tests/data/t2m_1991_monthly.nc'
#       self.xarray_dataset = xr.open_dataset(nc_file)
#   def test_xarray_input(self):
#       X = self.xarray_dataset
#       print(X)
#       print(X['t2m'])
