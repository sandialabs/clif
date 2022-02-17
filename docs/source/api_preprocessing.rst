Preprocessing
=============

Here's a summary of the available transforms in clif. You can find the complete transform descriptions :ref:`here <preprocessing_transforms_api>`.  

.. list-table:: Title
   :widths: 100 100
   :header-rows: 1

   * - Name
     - Brief description  
   * - :ref:`SeasonalAnomalyTransform<Removing cyclical trends>`
     - Remove cyclical mean trends from xarray DataArrays
   * - :ref:`ClipTransform<Clipping the data>`
     - Taking slices or subsets of the xarray DataArrays
   * - :ref:`MarginalizeOutTransform<Marginalizing>`
     - Marginalizing out dimensions of the data
   * - :ref:`Transpose<Transposing>`
     - Transpose the data to custom order of dimensions
   * - :ref:`FlattenData<Flattening the data>`
     - Flattening or stacking dimensions of the data
   * - :ref:`LinearDetrendTransform<Linear De-trending>`
     - Removing linear time-series trends for each grid point
   * - :ref:`TransformerMixin<Custom Transform>`
     - Template for creating your own custom preprocessing transform.


.. toctree::
   :maxdepth: 1
   :caption: Transforms
   
   transforms