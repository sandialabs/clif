#!/usr/bin/env python
# coding: utf-8

# # Fourier Analysis of time series data
# 
# In this tutorial, we will go through a simple example of how to plot the fft power spectrum for xarray time series data. We will use the temperature data as a baseline and look at the time series for a particular pressure elevation. Let's go ahead and process the data as in the previous tutorials. 

# In[1]:


# Location of the data for loading
import os
import numpy as np
DATADIR = os.path.join(os.getenv("HOME"),"Research/e3sm_data/fingerprint")
DATAFILE = "Temperature.nc"
WEIGHTFILE="lat_lon_weights.nc"

# Load the data from the data directory, using dask to help manage the memory (optional)
import xarray
data = xarray.open_dataarray(os.path.join(DATADIR, DATAFILE))
weights = xarray.open_dataarray(os.path.join(DATADIR, WEIGHTFILE))


# In[2]:


# Let's transform the data
import clif.preprocessing as cpp
from sklearn.pipeline import Pipeline

# Create the transform pipeline
transform_pipe = Pipeline(
    steps=[
        ("anomaly", cpp.SeasonalAnomalyTransform()),
        ("marginalize", cpp.MarginalizeOutTransform(dims=["lat", "lon"], lat_lon_weights=weights)),
        ("transpose", cpp.Transpose(dims=["plev", "time"])),
    ]
)

# Get the transformed data
data_new = transform_pipe.fit_transform(data)


# Now, let's extract the time_series for a particular pressure elevation to obtain a single time series data array.

# In[3]:


time_series_data = data_new.isel(plev=15)
time_series_data.shape


# In[4]:


# Let use the Fourier time series analysis class to look more closely at the signal
from clif import FourierTimeSeriesAnalysis


# In order to initiate the class, we just need to give a base unit so that we can automatically determine the sampling frequence, etc. Since the data is monthly for 12 years, if we choose a base unit of 'month', the sampling frequency will be 1 (1 sample per month), and if we choose a base unit of 'year' the sampling frequency will be 12 (12 samples per year). The only difference is in the interpretation of the power spectrum at the end. For the purposes of this experiment, we will choose a base unit of a year. 

# In[5]:


# In order to initiate the class, we just need to give a base unit so that we can automatically determine the sampling frequency, etc. 
fourier = FourierTimeSeriesAnalysis(base_unit="year")

# Now we simply fit the time series 
fourier.fit(data=time_series_data)


# In[6]:


# Let's plot what the signal looks like 
import matplotlib.pyplot as plt

fig1, ax1 = plt.subplots(figsize=[8.5,4])
ax1.plot(fourier.t_,time_series_data.values)
ax1.set_ylabel(r"$\Delta$" + u"\u00b0" + "K")


# In[7]:


#Let's plot the power spectrum in frequency vs power
fig2, ax2 = fourier.plot_power_spectrum(xaxis="frequency", logscale=False)
#We can also plot the period vs power
fig2, ax2 = fourier.plot_power_spectrum(xaxis="period", logscale=False)


# In[8]:


# get frequence/period of maximum value
# plt.plot(fourier.period_,fourier.power_)
max_index = np.argmax(fourier.power_)
max_freq = fourier.freq_[max_index]
max_period = fourier.period_[max_index]


# Finally, we can also filter the signal to get a smoother function without the high frequency modes. 

# In[9]:


# Remove all frequencies greater than 2 cycles per year
y_filtered = fourier.transform(freq_cutoff=9.6/4)
fig3, ax3 = plt.subplots(figsize=[8.5,2.0])
# ax3.plot(fourier.t_, y_filtered, label='filtered')
# ax3.plot(fourier.t_, time_series_data.values, label='signal')
# ax3.set_ylabel(r"$\Delta$" + u"\u00b0" + "K")
# ax3.legend()


# In[10]:


# # subtract trend from signal
# # Remove all frequencies greater than 2 cycles per year
# y_filtered = fourier.transform(period_cutoff=6)
# y = time_series_data.values.copy()
# fig4, ax4 = plt.subplots(figsize=[8.5,2.0])
# ax4.plot(fourier.t_, y - y_filtered)
# # ax3.plot(fourier.t_, y, label='signal')
# ax4.set_ylabel(r"$\Delta$" + u"\u00b0" + "K")
# # ax3.legend()


# In[11]:


# data = time_series_data.copy()
# data['time'] = fourier.t_
# from statsmodels.tsa.seasonal import STL
# res = STL(data,period=6).fit()
# res.plot()


# # In[12]:


# from statsmodels.datasets import co2
# data = co2.load().data
# data = data.resample('M').mean().ffill()
# time_index = data.index
# y = data.values.flatten()
# data_ts = xarray.DataArray(y,dims='time',coords={'time':time_index})
# fourier2 = FourierTimeSeriesAnalysis(base_unit='year')
# fourier2.fit(data_ts)
# y_filt = fourier2.transform(freq_cutoff=.5)

# plt.plot(fourier2.t_,y)
# plt.plot(fourier2.t_,y_filt)


# In[25]:
# raise SystemExit(0)

from statsmodels.datasets import co2
data = co2.load().data
data = data.resample('M').mean().ffill()
# time_index = data.index
# y = data.values.flatten()
time_index = fourier.t_
y = time_series_data.values
y = (y - y.min())/(y.max()-y.min())
data_ts = xarray.DataArray(y,dims='time',coords={'time':time_index})
x = 2*np.arange(len(time_index))/(len(time_index)-1) - 1
X = x.copy()[:,np.newaxis]

import tesuract
# pce_grid = {'order': list(range(1,32)),
#     'mindex_type': ['total_order'],
#     'fit_type': ['LassoCV'],
#     'fit_params': [{'alphas':np.logspace(-8,-1,10),'max_iter':100000,'tol':1.0e-2}]}
# pce = tesuract.RegressionWrapperCV(
#     regressor='pce',
#     reg_params=pce_grid,
#     n_jobs=8,
#     scorer='neg_root_mean_squared_error')
# pce.fit(X,y)
# print("Hyper-parameter CV PCE score is {0:.3f}".format(pce.best_score_))

from sklearn.ensemble import RandomForestRegressor
# random forest fit
rf_param_grid = {'n_estimators': [100,500,1000],
               'max_features': ['auto','sqrt','log2'],
               'max_depth': [5,10,15]
               }
from sklearn.model_selection import GridSearchCV
rfreg = GridSearchCV(RandomForestRegressor(), rf_param_grid, scoring='neg_root_mean_squared_error')
rfreg.fit(X,y)
print(sorted(rfreg.cv_results_.keys()))


# In[ ]:




