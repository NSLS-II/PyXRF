============
Installation
============

PyXRF works for python 2.7, 3.5 and 3.6 in Linux system. For Mac and Windows, we
only support python 3.5 version. We plan to move PyXRF to conda-forge soon.

First you need to install [conda] (http://continuum.io/downloads). We suggest
anaconda because it is a complete installation of the entire scientific python
stack, but is ~400 MB.  For advanced users, consider downloading [miniconda]
(http://conda.pydata.org/miniconda.html) because it is a smaller download (~20 MB).


Linux or Mac
============

Create a conda environment (say pyxrf_test) with python 3.6.

.. code:: python

    conda create -n pyxrf_test python=3.6


Then go to the environment named pyxrf_test

.. code:: python

    source activate pyxrf_test

At the same environment, install pyxrf by simply typing

.. code:: python

    conda install pyxrf -c lightsource2-tag -c conda-forge

For python 3.5 (Mac), please type

.. code:: python

    conda install pyxrf pyqt=4.11 enaml=0.9.8 -c lightsource2-tag -c conda-forge


Windows
=======

Install windows version of anaconda from (http://continuum.io/downloads).
Then you should see Anaconda Prompt installed in your computer. Double click Anaconda Prompt, and type the following lines to finish the installation.

Create a conda environment(say pyxrf_test) with python3.5.

.. code:: python

    conda create -n pyxrf_test python=3.5

Then go to the environment named pyxrf_test

.. code:: python

    activate pyxrf_test

At the same environment, install pyxrf by simply typing

.. code:: python

    conda install pyxrf xraylib pyqt=4.11 -c lightsource2-tag -c conda-forge


Run PyXRF
=========

At pyxrf_test environment, type

.. code:: python

    pyxrf


Update PyXRF
============

At pyxrf_test environment, type

.. code:: python

    conda update pyxrf -c lightsource2-tag -c conda-forge


Reminder
========

Every time you open a new terminal, make sure to go to pyxrf_test environment
first, then launch the software. For linux and mac system, type

.. code:: python

    source activate pyxrf_test

For windows,

.. code:: python

    activate pyxrf_test

then lanch pyxrf by typing

.. code:: python

    pyxrf


To leave this environment, at Linux and Mac system just type

.. code:: python

    source deactivate


and for Windows, type

.. code:: python

    deactivate
