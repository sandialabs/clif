..  _summary:

Main features
=============

**clif** contains a few modules that are helpful for working with climate data. We summarize these modules in the table below. 

.. list-table:: **clif** features
   :widths: 25 75
   :header-rows: 1

   * - Features
     - Description
   * - :ref:`preprocessing <preprocessing_transforms_table>`
     - Helpful transforms to preprocess xarray data sets, e.g., weighted averaging, detrending, seasonal anomaly detection, etc. All transforms are written in the style of scikit-learn transforms with the ``fit(...)`` and ``transform(...)`` class methods.   
   * - :ref:`plotting <api_plots>`
     - We provide 2d contour plotting functions in the style of the E3SM diagnostics Python package. It has been re-written with a new API to reduce redundancy and increase modularity. Currently, we have 2d contour plotting scripts for lat x lon, plev x lat, lat x time, and plev x time grids. 
   * - :ref:`fingerprinting/ EOF calculation <api_fp>`
     - We provide a fingerprints class to compute the empirical orthogonal functions of a 2d data array. This method utilizes scikit-learn's PCA transform and has the option of performing a varimax rotation to provide more interpretable (sparse) EOFs. 
   * - :ref:`time series analysis <api_ts>`
     - Fourier and statistical time series analysis tools. This feature is pretty thin, including some simple Fourier analysis, estimation of integrated autocorrelation lag time, and some statistical tests for :ref:`stationarity<api_stats>`. 

In the next few sections, we provide a few tutorials for how to use *clif*. 