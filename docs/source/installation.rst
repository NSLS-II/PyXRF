============
Installation
============

PyXRF is supported for Python 3.7-3.10 in Linux/Mac/Windows systems.
The easiest way to install PyXRF is to load it into a Conda environment from
``conda-forge`` Anaconda channel. Installation instructions are
identical for all supported OS.

PyXRF is currently not working properly with ``PyQt5`` v5.15, which is the latest available
version. The instructions show how to install earlier version of ``PyQt5`` from PyPI.

.. note::

  **Installation on Windows**: PyXRF can be installed from *conda-forge* only in the environments
  with Python 3.9 and older. Python 3.10 and 3.11 (latest and default) are currently
  not supported. If there is no existing Conda installation, download and
  install *Miniconda3*, choose the option to add *Miniconda3* to system PATH when asked,
  then run the following commands (replace the environment name `pyxrf-env` with
  any convenient name if necessary):

  .. code:: bash

    $ conda create -n pyxrf-env python=3.9 pip -c conda-forge
    $ conda activate pyxrf-env
    $ conda install pyxrf -c conda-forge


.. note::

  If you experience problems using `conda`, in particular if `conda` gets stuck trying to resolve
  the environment, you may try using `mamba`, which is designed as a drop-in replacement for
  `conda`. Once you install *Miniconda* or *Anaconda*, install `mamba` in the base environment:

  .. code:: bash

    $ conda install mamba -n base -c conda-forge

  Then use `mamba` instead of `conda` in all commands. Mamba may not respect pinned versions of
  packages. Since PyXRF currently does not work with `pyqt` v5.15, explicitly specify the version
  `"pyqt<5.15"` during installation, for example

  .. code:: bash

    $ mamba create -n pyxrf-env python=3.9 pip -c conda-forge
    $ mamba activate pyxrf-env
    $ mamba install pyxrf "pyqt<5.15" -c conda-forge

  You may still use `conda` where it works well, e.g. to activate an existing environment.

1. Install `Miniconda3 <http://conda.pydata.org/miniconda.html>`_  or
   `Anaconda <https://www.anaconda.com/distribution>`_. Select the latest version.
   *Miniconda* is sufficient for running PyXRF and contains the minimum number
   of packages. *Anaconda* is much larger download, which also installs GUI
   applications for more convenient management of Conda environments.

   If Conda is already installed on your computer, proceed to the next step.

2. Open a terminal (Linux/Mac) or start Anaconda Prompt from Windows Start Menu (Windows).

3. If you are using existing Conda installation, update Conda:

   .. code:: bash

       $ conda update -n base -c defaults conda

   If you are installing PyXRF on a Windows platform, close and restart Anaconda Prompt.

3. Install PyXRF from ``conda-forge`` Anaconda channels.

   Create new Conda environment with the latest version of Python:

   .. code:: bash

       $ conda create -n pyxrf-env python pip -c conda-forge

   or with the desired version of Python (e.g. 3.9):

   .. code:: bash

       $ conda create -n pyxrf-env python=3.9 pip -c conda-forge

   Activate the new environment:

   .. code:: bash

       $ conda activate pyxrf-env

   Install PyXRF:

   .. code:: bash

       $ conda install pyxrf -c conda-forge

4. Install PyXRF from ``PyPI``:

   Create new Conda environment with the latest version of Python and ``xraylib`` package,
   which is not available from PyPI:

   .. code:: bash

       $ conda create -n pyxrf-env python pip xraylib scikit-beam -c conda-forge

   or with the desired version of Python (e.g. 3.9):

   .. code:: bash

       $ conda create -n pyxrf-env python=3.9 pip xraylib scikit-beam -c conda-forge

   The ``scikit-beam`` package may be installed may be installed from PyPI if necessary.

   Activate the new environment:

   .. code:: bash

       $ conda activate pyxrf-env

   Install PyXRF from ``PyPI``:

   .. code:: bash

       $ pip install pyxrf 'PyQt5<5.15'

   or from source (editable installation):

   .. code:: bash

       $ cd <root-directory-of-the-repository>
       $ pip install 'PyQt5<5.15'
       $ pip install -e .

Starting PyXRF
==============

1. Open a terminal (Linux/Mac) or start Anaconda Prompt from Windows Start Menu (Windows).

2. Activate Conda environment that contains PyXRF installation
   (in our example ``pyxrf-env``):

   .. code:: bash

       $ conda activate pyxrf-env


3. Start PyXRF by typing:

   .. code:: bash

       $ pyxrf


Updating PyXRF
==============

1. Open a terminal (Linux/Mac) or start Anaconda Prompt from Windows Start Menu (Windows)

2. Activate Conda environment that contains PyXRF installation
   (in our example ``pyxrf-env``):

   .. code:: bash

       $ conda activate pyxrf-env


3. Update Conda installation of PyXRF:

   .. code:: bash

       $ conda update pyxrf scikit-beam -c conda-forge

4. Update PyPI installation of PyXRF:

   .. code:: bash

       $ conda update scikit-beam -c conda-forge
       $ pip install --upgrade pyxrf -c conda-forge

   if ``scikit-beam`` was installed using Conda or

   .. code:: bash

       $ pip install --upgrade scikit-beam pyxrf -c conda-forge

   if ``scikit-beam`` was installed from PyPI.


Deactivating Conda environment
==============================

   .. code:: bash

       $ conda deactivate
