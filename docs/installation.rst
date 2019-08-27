============
Installation
============

Current PyXRF version is supported for Python 3.6 and 3.7 in Linux/Mac/Windows systems.
PyXRF is installed from `nsls2forge` channel (anaconda.org). Installation instructions are
identical for all supported OS.

1. Install [conda] (https://www.anaconda.com/distribution). Select Python 3.7 version.
Anaconda package is recommended to casual users because it installs the entire
scientific python stack. For advanced users, consider downloading [miniconda]
(http://conda.pydata.org/miniconda.html) because it is a smaller download (~400 MB vs ~20 MB).

2. Open terminal (Linux/Mac) or start Anaconda Prompt from Windows Start Menu (Windows)

3. Create a conda environment (say `pyxrf-env`) with Python 3.6 or 3.7:

.. code:: python

    conda create -n pyxrf-env python=3.7


4. Then activate the newly created environment (`pyxrf-env`):

.. code:: python

    conda activate pyxrf-env

5. Install pyxrf in the active environment (in our example `pyxrf-env`):

.. code:: python

    conda install pyxrf -c nsls2forge


Starting PyXRF
==============

1. Open terminal (Linux/Mac) or start Anaconda Prompt from Windows Start Menu (Windows)

2. Activate Conda environment that contains PyXRF installation
   (in our example `pyxrf-env`):

.. code:: python

    conda activate pyxrf-env


3. Start PyXRF by typing:

.. code:: python

    pyxrf


Update PyXRF
============

1. Open terminal (Linux/Mac) or start Anaconda Prompt from Windows Start Menu (Windows)

2. Activate Conda environment that contains PyXRF installation
   (in our example `pyxrf-env`):

.. code:: python

    conda activate pyxrf-env


3. Update PyXRF by typing:

.. code:: python

    conda update pyxrf -c nsls2forge


Leave the activated Conda environment
=====================================
    
.. code:: python

    source deactivate
    
