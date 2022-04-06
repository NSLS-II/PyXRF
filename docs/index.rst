.. pyxrf documentation master file, created by
   sphinx-quickstart on Fri Mar 17 14:16:24 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

About PyXRF
===========

PyXRF is a python-based sophisticated fluorescence analysis package for fitting and
visualizing X-ray fluorescence data. This package contains a high-level fitting engine,
comprehensive command-line/GUI design, rigorous physics calculation and a powerful visualization interface. The theoretical part of PyXRF is based on MAPS, developed by Stefan Vogt at APS. PyXRF offers some of the unique features as follows.


* Automatic elements finding: Users do not need to spend extra time selecting elements manually.
* Forward calculation: Users can observe the spectrum from forward calculation at real time while adjusting input parameters. This will help users perform sensitivity analysis, and find an appropriate initial guess for fitting.
* Construct your own fitting algorithm: An advanced mode was created for advanced users to construct their own fitting strategies with a full control of each fitting parameter.
* Batch mode: Users can easily perform quick fitting of multiple fluorescence datasets or XANES datasets.
* Interface with NSLS-II database: A specific I/O interface was designed to obtain data directly from BNL/NSLS-II experimental database.



Table of Contents
=================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   work_at_beamlines
   summed_spectrum_fit
   data_output
   data_input
   questions
   credits
   release-notes


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
