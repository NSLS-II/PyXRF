==================================
Data analysis at NSLS II beamlines
==================================

Create working directory
========================
For instance at HXN, users usually create a directory under /data/users/2017Q1/
All the experimental data and fitting results will be saved in the created folder.


Load fluorescence data from database
====================================

At your working directory, first go to pyxrf environment, then type ipython.

.. code:: python

    source activate pyxrf
    ipython


Next users need to import important functions from pyxrf by typing

.. code:: python

    from pyxrf.api import *


If only load one file

.. code:: python

    make_hdf(1000, fname=‘scan_1000.h5’)


Note: the first argument of function ”make_hdf” is the run#(i.e., 1000), the second is the hdf filename to be saved. It may take several minutes to load 200by200 dataset. When loading process is done, you will see ‘scan_1000.h5’ is created in your working directory. Use pyxrf to load that data and work on it.
Warning: Data can’t be created if data file already exists. You need to remove the old file first.


If load multiple files

.. code:: python

    make_hdf(1000, 1100)

Note: the first argument is the starting run #, the second argument is the ending run #. This function will automatically transform all the fluorescence data within the run number range. If you want to define a prefix name, you can do make_hdf(1000, 1100, prefix=’abc_’). The default prefix name is ‘scan2D_’.
Warning: Data can’t be created if data file already exists. You need to remove the old file first.

How to load image data from merlin detector

.. code:: python

    export_hdf(1000, ‘myfile_1000.h5’)


Note: the first argument is the run id, while the second is the file name to save data.
Warning: Data can’t be created if data file already exists. You need to remove the old file first.


How to run PyXRF
================
PyXRF is a python-based x-ray fluorescence analysis package. Type the following two lines in a newly opened terminal console to initiate pyxrf.

.. code:: python

    source activate pyxrf
    pyxrf
