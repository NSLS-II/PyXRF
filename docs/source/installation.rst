============
Installation
============

PyXRF is supported for Python 3.9-3.11 in Linux/Mac/Windows systems.
The easiest way to install PyXRF is to load it into a Conda environment from
``conda-forge`` Anaconda channel. Installation instructions are
identical for all supported OS.

.. note::

  **Installation on Windows**: Since *Conda* does not behave very well recently,
  use *Mamba* to install PyXRF from *conda-forge* channel (see the *Mamba* installation
  notes below). For example, the following sequence of commands creates
  a new environment named ``pyxrf-env`` with Python 3.10 and installs PyXRF into
  the environment. Skip the ``mamba create`` command to install PyXRF into
  an existing environment.

  .. code:: bash

    $ mamba create -n pyxrf-env python=3.10 pip -c conda-forge
    $ mamba activate pyxrf-env
    $ mamba install pyxrf -c conda-forge

.. note::

  If you experience problems using *Conda*, in particular if *Conda* gets stuck trying to resolve
  the environment, use *Mamba*, which is designed as a drop-in replacement for *Conda*.
  *Mamba* is included in the `Miniforge distribution <https://github.com/conda-forge/miniforge>`_.
  It can also be installed in the base environment of an existing installation of *Miniconda*
  or *Anaconda*:

  .. code:: bash

    $ conda install -n base --override-channels -c conda-forge mamba 'python_abi=*=*cp*'

  See `Mamba installation instructions <https://mamba.readthedocs.io/en/latest/mamba-installation.html#mamba-install>`_
  for more details.

  **How to use `mamba`:** Simply replace ``conda`` with ``mamba`` when you create an environment,
  install or remove packages, etc. For example

  .. code:: bash

    $ mamba activate pyxrf-env
    $ mamba install pyxrf -c conda-forge


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

       $ pip install pyxrf PyQt5
       $ pip install pyxrf 'PyQt5<5.15'  # Older version of PyQT

   or from source (editable installation):

   .. code:: bash

       $ cd <root-directory-of-the-repository>
       $ pip install PyQt5
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
