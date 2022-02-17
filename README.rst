**clif**
============

**clif** is a CLImate Fingerprinting library that calculates empirical orthogonal functions for mainly climate data.  

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
----------------

Once you have successfully installed clif, you can compute EOFs of data (as a numpy array for now) as follows.[#]_

.. code-block:: python

    from clif import fingerprints
    from sklearn import datasets

    X = datasets.load_digits().data
    fp = fingerprints(n_eofs=8)
    fp.fit(X)
    EOFs = fp.eofs_


Preprocessing 
-------------

clif also has a bunch of preprocessing transforms useful for manipulating xarray DataArrays. To see more information from the documentation, go to the `docs/` folder and open `index.html`. All transforms are templated and use the following pseudo code interface. 

.. code-block:: python

    from clif import preprocessing

    X = load_xarray_data()
    xarrayTransform = preprocessing.TransformName(**init_params)))
    X_transformed = xarrayTransform.fit_transform(X)