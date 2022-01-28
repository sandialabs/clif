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

def construct_data_matrix(dataset,variable,row_coord=['time'],col_coord=['lat'],detrend='month',lat_lon_weighting = True):
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
                                        new_variable_suffix='')

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

    # marginalize data array over marginal_coords
    for c in marginal_coords:
        if c == 'lat':
            data = (data*weights_lat).sum(dim='lat')
        else:
            data = data.mean(dim=c)

    assert len(data.shape) == len(col_coord)+1

    if len(col_coord) > 1:
        # need to stack variables
        data = data.stack(dim=col_coord)

    # extract numpy object
    X_original = data.values

    # filter out nan variables
    col_index = np.sum(np.isnan(X_original),axis=0)
    X_new = X_original[:,col_index == 0]
