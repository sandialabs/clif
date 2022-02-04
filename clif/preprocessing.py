from tqdm import tqdm
import numpy as np
import xarray

from .preprocessing_new import *

def remove_cyclical_trends( data, variable=None, cycle='month',new_variable_suffix='denoised', use_groupby=True): 
	'''Removes cyclical trends from xarray time-series data
	
	Computes the average values for ds[varname] by cycle, and removes these trends
	
	Parameters
	----------
	data: xarray.DataArray or xarray.DataSet
		data array object holding the time-series data
	variable: None or str
		variable name for which we want trend removed. Must be in xarray dataset if not None
	cycle: str (hour, day, month=default)
		scale at which we remove cyclical trend
	inplace: bool (default=True)
		return the same xarray data set with modified/ denoised variable
		
	Returns
	--------
	xarray.DataArray
		xarray data set object holding both the varname and filtered version which holds the normalized data

	To do
	-----
	* Need to add option to input xarray.DataArray in addition to Dataset object

	Example Usage
	-------------
	>>> xarray_ds = remove_cyclical_trends(
				xarray_ds,'t2m',cycle='month',new_variable_suffix='trend_removed')
	>>> xarray_ds['t2m_trend_removed']

	'''
	print( 'Normalize by {0} cycles'.format(cycle))
	if variable is not None and isinstance(data,xarray.Dataset):
		newvarname= '{0}_{1}'.format(variable, new_variable_suffix)
		dict = {newvarname : data.variables[variable].copy()}
	try:
		data['time.'+cycle]
	except:
		if isinstance(data,xarray.Dataset):
			print("Converting xarray Dataset data-time to proper format...")
			data = xarray.decode_cf(data)
	# get indices by cycle
	cycle_values = np.sort(np.unique(data["time."+cycle].values))
	print("Removing {0}ly cycles...".format(cycle))
	if use_groupby:
		# Much faster operation using the groupby feature which does array broadcasting and assignment very quickly
		if isinstance(data,xarray.Dataset):
			#compute means according to month
			data_copy = data.copy(deep=True) # create deep copy so you don't overwrite existing data
			mu_by_group = data_copy[variable].groupby("time."+cycle).mean(dim='time')
			data_copy[newvarname] = data_copy[variable].groupby("time."+cycle) - mu_by_group
			return data_copy
		elif isinstance(data,xarray.DataArray):
			mu_by_group = data.groupby("time."+cycle).mean(dim='time')
			data_new = data.groupby("time."+cycle) - mu_by_group
			return data_new
	else: # will be deprecated
		if isinstance(data,xarray.Dataset):
			for i in tqdm(cycle_values):
				cycle_i=np.where(data["time."+cycle]==i)
				climo_for_cycle_i = data[variable].groupby("time."+cycle)[i].mean(dim='time') 
				anoms = data.variables[variable][cycle_i]-climo_for_cycle_i
				# Now put them in the right place in the array. 
				dict[newvarname][cycle_i]=anoms
			data_new = data.assign(dict) # creates a new xarray data set (different mem)
			return data_new
		elif isinstance(data,xarray.DataArray):
			data_new = data.copy(deep=True)
			for i in tqdm(cycle_values):
				cycle_i=np.where(data["time."+cycle]==i)
				climo_for_cycle_i = data.groupby("time."+cycle)[i].mean(dim='time') 
				anoms = data[cycle_i]-climo_for_cycle_i
				# Now put them in the right place in the array. 
				data_new[cycle_i]=anoms
			return data_new

def construct_data_matrix(dataset,variable,row_coord=['time'],col_coord=['lat'],detrend='month',removed_nans=True,lat_lon_weighting = False, return_np_array_only=False,use_groupby=True):
	'''function to preprocess xarray to construct data matrix for fingerprinting
	
	performs the detrending of cyclical information and summing over coordinates not defined in the col_coord list

	Notes:
	------

	Latitude and longitude are assumed to be labeled as 'lat' and 'lon' but will need to change that
	
	'''

	# remove cyclical trend first
	if detrend is not None:
		dataset = remove_cyclical_trends(data=dataset,
										variable=variable,
										cycle=detrend,
										new_variable_suffix='',
										use_groupby=use_groupby)

	variable = variable + '_'
	coords = row_coord + col_coord
	data = dataset[variable] # creates a copy of the variables
	all_coords = list(data.coords.dims)
	marginal_coords = all_coords.copy()
	for c in coords: 
		marginal_coords.remove(c)

	# get lat lon weights if possible
	weights = dataset.area
	weights /= np.sum(weights) # normalize
	weights
	weights_lon = weights.mean(dim='lat')
	weights_lon /= np.sum(weights_lon)
	weights_lat = weights.mean(dim='lon')
	weights_lat /= np.sum(weights_lat)

	if lat_lon_weighting is True:
		data = data * weights # weight data via area of lat lon grid

	# marginalize data array over marginal_coords
	for c in marginal_coords:
		if lat_lon_weighting is True:
			# no need for weighted sums is lat_lon_weighting is True
			data = data.mean(dim=c)
		else:
			# if no lat_lon_weighting (default False), then only weight the latitude before summing over
			if c == 'lat':
				# must use area weighting if marginalizing over latitude
				data = (data*weights_lat).sum(dim='lat')
			else:
				# uniform average weighting for all other coordinates, including longitude
				data = data.mean(dim=c)

	assert len(data.shape) == len(col_coord)+1, "data shape must be of the same dimension (plus 1) as the column coordinates. The plus 1 dimension is for the row coordinate, e.g., time."

	if len(col_coord) > 1:
		# need to stack col_coord variables in order to produce a 2d matrix for use in PCA
		data = data.stack(dim=col_coord)

	# extract numpy object
	X_np = data.values

	if removed_nans:
		# filter out nan variables by leaving out columsn with at least one nan variable. 
		col_index = np.sum(np.isnan(X_np),axis=0)
		X_np_nonan = X_np[:,col_index == 0]

	if return_np_array_only:
		return X_np_nonan
	else:
		# otherwise, return numpy array AND the xarray data object
		# if filtering nans, data might be different than the numpy array
		return X_np_nonan, data

def add_trend_lines_to_eof(times,eof_time_series,change_point):
	'''
	Compute a simple linear regression with least squares fitting of a univariate time-series data, before and after change point event. 
	'''
	# get period before and after event, must be a datetime e.g. np.datetime or cftime object
	before_event = times < change_point
	after_event = times >= change_point
	time_before = times[before_event]
	time_after = times[after_event]
	eof1_ts = eof_time_series
	eof1_ts_before = eof1_ts[before_event]
	eof1_ts_after = eof1_ts[after_event]
	# fig, ax = plt.subplots(1,1,figsize=(6,4))
	# ax.plot(time_before,eof1_ts_before,'b--')
	# ax.plot(time_after,eof1_ts_after,'r*--')

	from sklearn.linear_model import LinearRegression
	times_np = np.arange(len(times)).astype(float)
	times_np /= len(times_np)-1
	X_before = times_np[before_event][:,np.newaxis]
	X_after = times_np[after_event][:,np.newaxis]
	linreg_before = LinearRegression().fit(X_before,eof1_ts_before).predict(X_before)
	linreg_after = LinearRegression().fit(X_after,eof1_ts_after).predict(X_after)
	# ax.plot(time_before,linreg_before,'b')
	# ax.plot(time_after,linreg_after,'r')

	return (time_before,linreg_before), (time_after,linreg_after)