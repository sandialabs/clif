import clif
import unittest
import numpy as np
import warnings
from sklearn.metrics import mean_squared_error
from sklearn import datasets
import os
import time

relpath = clif.__file__[:-11] # ignore the __init__.py specification
print("relpath:",relpath)

def mse(a,b):
	return mean_squared_error(a,b,squared=False)

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
		assert fp.U_.shape == (n_samples,n_components), "Projection matrix is not the right size"
		assert fp.V_.shape == (n_components,n_dim), "Component matrix is not the right size"
		return
	def test_pca_fit_make_sure_eof_same_as_component(self):
		# default constructor
		n_components = 8
		fp = clif.fingerprints(n_eofs=n_components)
		fp.fit(self.X)
		n_samples, n_dim = self.X.shape
		assert np.sum((fp.V_ - fp.eofs_.T)**2) == 0, "EOFs should be n_dim by n_components"
		return
	def test_pca_fit(self):
		# default constructor
		n_components = 8
		fp = clif.fingerprints(n_eofs=n_components)
		fp.fit(self.X)
		assert hasattr(fp,'U_'), "Fit function is not producing a projection matrix."
		assert hasattr(fp,'S_'), "Fit function is not producing singular values."
		assert hasattr(fp,'V_'), "Fit function is not producing a component matrix."
		assert hasattr(fp,'eofs_'), "Fit function is not producing an eofs matrix."
		return