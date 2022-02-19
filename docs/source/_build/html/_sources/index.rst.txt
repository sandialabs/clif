..  _main:

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

.. toctree::
   :maxdepth: 2
   :caption: Usage Guide
   
   introduction

.. toctree::
   :maxdepth: 2
   :caption: API
   
   api_fingerprints
   api_preprocessing



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. [#] Python 3+ is required and clif has only been tested for 
         3.7.6 so far. 