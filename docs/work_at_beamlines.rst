==================================
Data analysis at NSLS II beamlines
==================================

For instance at HXN, users usually create a directory under /data/users/2017Q1/
All the experimental data and fitting results will be saved in the created folder.


Load fluorescence data from database
====================================

At your working directory, first go to pyxrf environment, then type ipython.

.. code-block:: python

    source activate pyxrf
    ipython


Next users need to import important functions from pyxrf by typing

.. code-block:: python

    from pyxrf.api import *


If you only load data from one run into a file, then use

.. code-block:: python

    make_hdf(1000, fname='scan_1000.h5')


Note: the first argument of function ”make_hdf” is the runID (i.e., 1000),
the second is the hdf filename to be saved. It may take several minutes to load
200by200 dataset. When loading process is done, you will see ‘scan_1000.h5’
is created in your working directory. Use pyxrf to load that data and work on it.

Warning: Data can’t be created if data file already exists. You need to remove
the old file first. You can also ignore ``fname`` keyword, then the default file
name is used.


If multiple files need to be loaded

.. code-block:: python

    make_hdf(1000, 1100)


Note: the first argument is the starting run ID, the second argument is the
ending run ID. This function will automatically transform all the fluorescence
data within the run number range. If you want to define a prefix name, you can
do ``make_hdf(1000, 1100, prefix='abc_')``. The default prefix name is
``scan2D_``.

Warning: Data can’t be created if data file already exists. You need to remove
the old file first.


Pixel fitting from command line
===============================

The parameter json file needs to be defined first. This can be created from
PyXRF GUI when you do summed spectrum fitting.

.. code-block:: python

    fit_pixel_data_and_save(wd, fname, param_file_name=param_file)

wd is the working directory. fname is the hdf file. param_file is the parameter
json file. You can easily write for loop to fit multiple data.

Please also refer to jupyter notebook example
https://github.com/NSLS-II/PyXRF/blob/master/examples/Batch_mode_fit.ipynb


PyXRF configuration at beamlines
================================

A json was created to define beamline name at /etc/pyxrf/pyxrf.json .

From file https://github.com/NSLS-II/PyXRF/blob/master/pyxrf/model/load_data_from_db.py ,
pyxrf will search which beamline the machine is at, then connects to databroker
accordingly. For instance, if it is hxn machine, pyxrf will read hxn database
information from
https://github.com/NSLS-II/PyXRF/blob/master/pyxrf/db_config/hxn_db_config.py,
where databroker is defined and handlers are registered.

In order to add new beamlines, the similar step should be followed to connect
to database for that beamline.


How to add more beamlines to use PyXRF
======================================

After adding configuration file for new beamline, we need to work on how to
transfer data from databroker into the format PyXRF can take. The only file we
need to work on is ``PyXRF/pyxrf/model/load_data_from_db.py``.

For instance, in order to transfer fluorescence data at HXN beamline to PyXRF
format, you first need to write a function of ``map_data2D_hxn`` in file
``load_data_from_db.py``. Then in function of ``fetch_data_from_db``, you
simply add ``map_data2D_hxn`` for hxn beamline. Please refer to source code to
see how it works.

Those functions on data IO should be moved to
https://github.com/NSLS-II/suitcase-pyxrf later.
