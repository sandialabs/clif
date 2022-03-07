Fourier Analysis of time series data
====================================

In this tutorial, we will go through a simple example of how to plot the
fft power spectrum for xarray time series data. We will use the
temperature data as a baseline and look at the time series for a
particular pressure elevation. Let’s go ahead and process the data as in
the previous tutorials.

.. code:: ipython3

    # Location of the data for loading
    import os
    DATADIR = os.path.join(os.getenv("HOME"),"Research/e3sm_data/fingerprint")
    DATAFILE = "Temperature.nc"
    WEIGHTFILE="lat_lon_weights.nc"
    
    # Load the data from the data directory, using dask to help manage the memory (optional)
    import xarray
    data = xarray.open_dataarray(os.path.join(DATADIR, DATAFILE))
    weights = xarray.open_dataarray(os.path.join(DATADIR, WEIGHTFILE))

.. code:: ipython3

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

Now, let’s extract the time_series for a particular pressure elevation
to obtain a single time series data array.

.. code:: ipython3

    time_series_data = data_new.isel(plev=15)
    time_series_data.shape




.. parsed-literal::

    (144,)



.. code:: ipython3

    # Let use the Fourier time series analysis class to look more closely at the signal
    from clif import FourierTimeSeriesAnalysis

In order to initiate the class, we just need to give a base unit so that
we can automatically determine the sampling frequence, etc. Since the
data is monthly for 12 years, if we choose a base unit of ‘month’, the
sampling frequency will be 1 (1 sample per month), and if we choose a
base unit of ‘year’ the sampling frequency will be 12 (12 samples per
year). The only difference is in the interpretation of the power
spectrum at the end. For the purposes of this experiment, we will choose
a base unit of a year.

.. code:: ipython3

    # In order to initiate the class, we just need to give a base unit so that we can automatically determine the sampling frequency, etc. 
    fourier = FourierTimeSeriesAnalysis(base_unit="year")
    
    # Now we simply fit the time series 
    fourier.fit(data=time_series_data)

.. code:: ipython3

    # Let's plot what the signal looks like 
    import matplotlib.pyplot as plt
    
    fig1, ax1 = plt.subplots(figsize=[8.5,4])
    ax1.plot(fourier.t_,time_series_data.values)
    ax1.set_ylabel(r"$\Delta$" + u"\u00b0" + "K")




.. parsed-literal::

    Text(0, 0.5, '$\\Delta$°K')




.. image:: time_series_analysis_files/time_series_analysis_8_1.png


.. code:: ipython3

     #Let's plot the power spectrum in frequency vs power
     fig2, ax2 = fourier.plot_power_spectrum(xaxis="frequency", logscale=False)
      #We can also plot the period vs power
     fig2, ax2 = fourier.plot_power_spectrum(xaxis="period", logscale=False)



.. image:: time_series_analysis_files/time_series_analysis_9_0.png



.. image:: time_series_analysis_files/time_series_analysis_9_1.png


Finally, we can also filter the signal to get a smoother function
without the high frequency modes.

.. code:: ipython3

    # Remove all frequencies greater than 2 cycles per year
    y_filtered = fourier.transform(freq_cutoff=2)
    fig3, ax3 = plt.subplots(figsize=[8.5,4.0])
    ax3.plot(fourier.t_, y_filtered, label='filtered')
    ax3.plot(fourier.t_, time_series_data.values, label='signal')
    ax3.set_ylabel(r"$\Delta$" + u"\u00b0" + "K")
    ax3.legend()




.. parsed-literal::

    <matplotlib.legend.Legend at 0x1445fad50>




.. image:: time_series_analysis_files/time_series_analysis_11_1.png

