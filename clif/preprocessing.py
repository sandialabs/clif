from tqdm import tqdm
import numpy as np
import xarray

def remove_cyclical_trends( xarray_dataset, variable_name, cycle='month',new_variable_suffix='new'): 
	'''Removes cyclical trends from xarray time-series data
	
	Computes the average values for ds[varname] by cycle, and removes these trends
	
	Parameters
	----------
	xarray_dataset: xarray.DataArray
		data array object holding the time-series data
	variable_name: str
		variable name for which we want trend removed. Must be in xarray dataset
	cycle: str (hour, day, month=default)
		scale at which we remove cyclical trend
		
	Returns
	--------
	xarray.DataArray
		xarray data set object holding both the varname and filtered version which holds the normalized data

	Example Usage
	-------------
	>>> xarray_ds = remove_cyclical_trends(
				xarray_ds,'t2m',cycle='month',new_variable_suffix='trend_removed')
	>>> xarray_ds['t2m_trend_removed']

	'''
	print( 'Normalize by {0} cycles'.format(cycle))
	newvarname= '{0}_{1}'.format(variable_name, new_variable_suffix)
	dict = {newvarname : xarray_dataset.variables[variable_name].copy()}
	try:
		xarray_dataset['time.'+cycle]
	except:
		print("Converting data-time to proper format...")
		xarray_dataset = xarray.decode_cf(xarray_dataset)
	# get indices by cycle
	cycle_values = np.sort(np.unique(xarray_dataset["time."+cycle].values))
	print("Removing {0}ly cycles...".format(cycle))
	for i in tqdm(cycle_values):
		cycle_i=np.where(xarray_dataset["time."+cycle]==i)
		climo_for_cycle_i = xarray_dataset[variable_name].groupby("time."+cycle)[i].mean(dim='time') 
		anoms = xarray_dataset.variables[variable_name][cycle_i]-climo_for_cycle_i
		# Now put them in the right place in the array. 
		dict[newvarname][cycle_i]=anoms
	xarray_dataset = xarray_dataset.assign(dict)
	return xarray_dataset

