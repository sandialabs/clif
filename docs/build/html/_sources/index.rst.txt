..  _main:

clif
====

*clif* contains code to perform climate fingerprinting, i.e. principal component analysis of climate data, a collection of preprocessing scripts for xarray DataArrays, and some time series statistical tools. We use a templated OOP approach so that all classes can be seamlessly integrated with the transform operator in scikit-learn. [#]_ 

For a guide on how to use the EOF class and the preprocessing transforms see the :ref:`EOF API <api_fp>` and :ref:`Preprocessing API <preprocessing_transforms_table>` 


Installation
------------

The code is super easy to install with ``pip``. Make sure you have ``numpy``,
``scikit-learn``, and ``xarray``. Then, after cloning, cd into the ``clif`` directory, i.e. the folder with the ``setup.py``, and run

.. code-block:: python

	pip install .


You can also run a suite of unit tests and regression tests before installation with 

::

   python -m pytest clif/tests

to check that the library works. That's it! Now you are ready to use **clif**. 

Quickstart
-----------

Once you have successfully installed clif, you can compute EOFs of data (as a numpy array for now) as follows.[#]_

.. code-block:: python

   from clif import fingerprints
   from sklearn import datasets

   X = datasets.load_digits().data
   fp = fingerprints(n_eofs=8)
   fp.fit(X)
   EOFs = fp.eofs_

clif also has a bunch of **preprocessing** transforms useful for manipulating xarray DataArrays. To see more information from the documentation, go to the `docs/` folder and open `index.html`. All transforms are templated and use the following pseudo code interface. 

.. code-block:: python

    from clif import preprocessing

    X = load_xarray_data()
    xarrayTransform = preprocessing.TransformName(**init_params)))
    X_transformed = xarrayTransform.fit_transform(X)

.. toctree::
   :maxdepth: 2
   :caption: Introduction
   
   tutorials/getting_started
   overview

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/first_look
   tutorials/intro_to_preproc
   tutorials/time_series_analysis

.. toctree::
   :maxdepth: 2
   :caption: API
   
   api_fingerprints
   api_preprocessing
   api_statistics
   api_plotting
   api_fourier


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. [#] Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011. 