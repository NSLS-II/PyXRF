============
Installation
============

PyXRF is supported for Python 3.6 and 3.7 in Linux/Mac/Windows systems.
The easiest way to install PyXRF is to load it into a Conda environment from 
:code:`nsls2forge` Anaconda channel. Installation instructions are
identical for all supported OS.

1. Install `Conda <https://www.anaconda.com/distribution>`_. Select Python 3.7 version.
   Anaconda package is recommended to casual users because it installs the entire
   scientific Python stack. Advanced users may consider installing
   `Miniconda <http://conda.pydata.org/miniconda.html>`_ since it is a much smaller
   download (~400 MB vs. ~20 MB).

2. Open a terminal (Linux/Mac) or start Anaconda Prompt from Windows Start Menu (Windows).
   
3. Create a new Conda environment (e.g. :code:`pyxrf-env`) with Python 3.6 or 3.7:

.. code:: python

    conda create -n pyxrf-env python=3.7


4. Activate the created Conda environment (in our example :code:`pyxrf-env`):

.. code:: python

    conda activate pyxrf-env

5. Install PyXRF in the active Conda environment (in our example :code:`pyxrf-env`):

.. code:: python

    conda install pyxrf -c nsls2forge


Starting PyXRF
==============

1. Open a terminal (Linux/Mac) or start Anaconda Prompt from Windows Start Menu (Windows).

2. Activate Conda environment that contains PyXRF installation
   (in our example :code:`pyxrf-env`):

.. code:: python

    conda activate pyxrf-env


3. Start PyXRF by typing:

.. code:: python

    pyxrf


Updating PyXRF
==============

1. Open a terminal (Linux/Mac) or start Anaconda Prompt from Windows Start Menu (Windows)

2. Activate Conda environment that contains PyXRF installation
   (in our example :code:`pyxrf-env`):

.. code:: python

    conda activate pyxrf-env


3. Update PyXRF by typing:

.. code:: python

    conda update pyxrf -c nsls2forge


Leave the activated Conda environment
=====================================
    
.. code:: python

    conda deactivate
    
