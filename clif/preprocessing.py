from tqdm import tqdm
import numpy as np
import xarray

def remove_cyclical_trends( data, variable, cycle='month',new_variable_suffix='denoised'): 
	'''Removes cyclical trends from xarray time-series data
	
	Computes the average values for ds[varname] by cycle, and removes these trends
	
	Parameters
	----------
	data: xarray.DataArray
		data array object holding the time-series data
	variable: str
		variable name for which we want trend removed. Must be in xarray dataset
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
	newvarname= '{0}_{1}'.format(variable, new_variable_suffix)
	dict = {newvarname : data.variables[variable].copy()}
	try:
		data['time.'+cycle]
	except:
		print("Converting data-time to proper format...")
		data = xarray.decode_cf(data)
	# get indices by cycle
	cycle_values = np.sort(np.unique(data["time."+cycle].values))
	print("Removing {0}ly cycles...".format(cycle))
	for i in tqdm(cycle_values):
		cycle_i=np.where(data["time."+cycle]==i)
		climo_for_cycle_i = data[variable].groupby("time."+cycle)[i].mean(dim='time') 
		anoms = data.variables[variable][cycle_i]-climo_for_cycle_i
		# Now put them in the right place in the array. 
		dict[newvarname][cycle_i]=anoms
	data_new = data.assign(dict) # creates a new xarray data set (different mem)
	return data_new

