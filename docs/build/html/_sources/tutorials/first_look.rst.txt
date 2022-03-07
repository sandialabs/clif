First look at the data
======================

In this next section, we will take a first look at plotting and
preprocessing the data.

First, let us load some sample data. In these tutorials, we will use
historical temperature data from a fully coupled E3SM simulation. The
data is monthly Temperature data given as a lat, lon and plev (pressure
in Pa) tensor. There are 5 DECK ensemble members, and we’ve already gone
ahead and averaged those values so we have one data tensor.

.. code:: ipython3

    # Location of the data for loading
    import os
    DATADIR = os.path.join(os.getenv("HOME"),"Research/e3sm_data/fingerprint")
    DATAFILE = "Temperature.nc"

.. code:: ipython3

    # Load the data from the data directory, using dask to help manage the memory (optional)
    import xarray
    data = xarray.open_dataarray(os.path.join(DATADIR, DATAFILE))
    print(data)


.. parsed-literal::

    <xarray.DataArray 'T' (time: 144, plev: 37, lat: 24, lon: 48)>
    [6137856 values with dtype=float32]
    Coordinates:
      * lat      (lat) float64 -84.38 -77.09 -69.76 -62.43 ... 69.76 77.09 84.38
      * lon      (lon) float64 0.0 7.5 15.0 22.5 30.0 ... 330.0 337.5 345.0 352.5
      * plev     (plev) float64 1e+05 9.75e+04 9.5e+04 ... 300.0 200.0 100.0
      * time     (time) object 1985-02-01 00:00:00 ... 1997-01-01 00:00:00


Now, let’s produce some contour view plots of the data. Since this is a
4-dimensional tensor, we need to average out some of the dimensions in
order to use the contour plot functionality.

.. code:: ipython3

    # Let's average out time and plev, and plot the lat lon grid
    data_lat_lon = data.mean(dim=['time','plev'])
    print(data_lat_lon.shape)


.. parsed-literal::

    (24, 48)


Let’s import the contour plot functionality from clif in order to plot
the lat lon field in the style of e3sm diags!

.. code:: ipython3

    # import the lat lon contour plotting class
    import clif.visualization as cviz
    
    # Now we initialize the contout.plot_lat_lon class with some parameters like the color map and titles
    clifplot1 = cviz.contour.plot_lat_lon(
        cmap_name="e3sm_default",
        title="Temperature",
        rhs_title=u"\u00b0" + "K",
        lhs_title="E3SMv1 DECK avg ne30np4",
    )

.. code:: ipython3

    # To view the plot, we simple call the show(data) class method where data is our lat lon data
    clifplot1.show(data_lat_lon)



.. image:: first_look_files/first_look_8_0.png




.. parsed-literal::

    <clif.visualization.contour.plot_lat_lon at 0x14082df90>



\*Note that if you change the figure size, some of the statistics may
not look right.

We can also create other types of contour plots like lat vs plev, or
plev vs time, etc. As an example, let’s look at the pressure level vs
time for this data set.

.. code:: ipython3

    # First, let us average out lat and lon coordinates to get the data as time vs plev
    data_time_plev = data.mean(dim=['lat','lon'])
    print(data_time_plev.shape)


.. parsed-literal::

    (144, 37)


.. code:: ipython3

    # let us load the plev vs time contour plotting class. This will plot time on the x axis and plev on the y axis. 
    clifplot2 = cviz.contour.plot_plev_time(
        cmap_name="e3sm_default",
        title="Temperature",
        rhs_title=u"\u00b0" + "K",
        lhs_title="E3SMv1 DECK avg ne30np4",
    )

.. code:: ipython3

    # By default, the contour plotting will make the row of the input the y coordinate and the column as the x coordinate. Since our data shape is time x plev, we need to transpose this before showing the plot. 
    clifplot2.show(data_time_plev.T)



.. image:: first_look_files/first_look_12_0.png




.. parsed-literal::

    <clif.visualization.contour.plot_plev_time at 0x1433ad310>



Template for new plotting functions (API)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It is easy to customize any new plots as well. There are helper and base
classes in the contour.py module, but in general, any new plot just
needs the following template.

.. code:: python

   from abc import ABC, abstractmethod

   class BasePlot(ABC):
       @abstractmethod
       def draw(self, *args, **kwargs):
           pass

       @abstractmethod
       def set_yaxis_properties(self, *args, **kwargs):
           pass

       @abstractmethod
       def set_xaxis_properties(self, *args, **kwargs):
           pass

       @abstractmethod
       def finish(self, *args, **kwargs):
           pass

       @abstractmethod
       def show(self, *args, **kwargs):
           pass


