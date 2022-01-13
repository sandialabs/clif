import clif
import unittest
import numpy as np
import warnings
from sklearn.metrics import mean_squared_error
from sklearn import datasets
import os
import time
import xarray as xr

relpath = clif.__file__[:-11] # ignore the __init__.py specification
print("relpath:",relpath)

def mse(a,b):
	return mean_squared_error(a,b,squared=False)

class TestPreprocessing(unittest.TestCase):
	@classmethod
	def setUpClass(self):
		# load data sets
		nc_file = relpath + '/tests/data/t2m_1991_monthly.nc'
		self.xarray_dataset = xr.open_dataset(nc_file)
	def test_removing_monthly_cylcal_trend(self):
		# default constructor
		from clif.preprocessing import remove_cyclical_trends
		data = self.xarray_dataset
		data_new = remove_cyclical_trends(data=data,variable='t2m',cycle='month',new_variable_suffix='denoised')
		assert 't2m' in  data_new.variables, "Trend removed did not keep original data."
		assert 't2m_denoised' in  data_new.variables, "Trend removed did not work."
		return
	def test_removing_yearly_cylcal_trend(self):
		# default constructor
		from clif.preprocessing import remove_cyclical_trends
		data = self.xarray_dataset
		data_new = remove_cyclical_trends(data=data,variable='t2m',cycle='year',new_variable_suffix='denoised')
		assert 't2m' in  data_new.variables, "Trend removed did not keep original data."
		assert 't2m_denoised' in  data_new.variables, "Trend removed did not work."
		return
	def test_removing_hourly_cylcal_trend(self):
		# default constructor
		from clif.preprocessing import remove_cyclical_trends
		data = self.xarray_dataset
		data_new = remove_cyclical_trends(data=data,variable='t2m',cycle='hour',new_variable_suffix='denoised')
		assert 't2m' in  data_new.variables, "Trend removed did not keep original data."
		assert 't2m_denoised' in  data_new.variables, "Trend removed did not work."
		return
	def test_removing_daily_cylcal_trend(self):
		# default constructor
		from clif.preprocessing import remove_cyclical_trends
		data = self.xarray_dataset
		data_new = remove_cyclical_trends(data=data,variable='t2m',cycle='day',new_variable_suffix='denoised')
		assert 't2m' in  data_new.variables, "Trend removed did not keep original data."
		assert 't2m_denoised' in  data_new.variables, "Trend removed did not work."
		return
	def test_check_if_data_shared(self):
		# default constructor
		from clif.preprocessing import remove_cyclical_trends
		data = self.xarray_dataset
		data_new = remove_cyclical_trends(data=data,variable='t2m',cycle='day',new_variable_suffix='denoised')
		data_new['t2m'] *= 0.0
		assert np.sum(data['t2m']).values > 0, "Variables in old xarray dataset are being shared with new ones. Could cause issues."
		return