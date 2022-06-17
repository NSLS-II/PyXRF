=============
Release Notes
=============

v1.0.17 (2022-06-16)
====================

Added
-----
- Support for Python 3.10

Fixed
-----
- Updated the code for loading raw data at SRX beamline to be compatible with Databroker V1
  (intake version).
- Fixed the code for loading of raw data at HXN beamline.
- Fixed the issue with batch fitting of multiple files (``pyxrf_batch``) observed after upgrade of
  Dask (observed for versions of Dask starting with v2022.2.0, Linux).

v1.0.16 (2022-05-26)
====================

Added
-----
- Support for the detector ``xs4`` (SRX beamline).

Changed
-------
- Databroker v1 is used for loading data at SRX beamline.

v1.0.15 (2022-04-06)
====================

Added
-----
- SRX data, new format: support for step scans using top stages.

Fixed
-----
- Fixed a bug in ``fit_pixel_data_and_save()`` that prevented saving maps as 'TXT' files.
- SRX data, new format: proper handling of scan data using both ``xs`` and ``xs2`` detectors.

v0.0.9.4
========
- Add auto peak finding
- Update readme doc and add beamline config

v0.0.9
========
- Pkg can be used for python 3.6 with pyqt>=5.9
- Use latest databroker, with API like Broker.named. Remove old databroker config.
- Init the work of loading data directly from Database
- Support three beamlines including hxn, srx and xfm
- Bug fix from previous versions

v0.0.7
========
- Image normalization and output normalization are synchronized.
- More examples are added, such as batch mode fitting, and preparation for tomography reconstruction (creating movie).
- Only one strategy is selected for summed spectrum fitting.
- More controls to output data to 2D image, and to visualize on GUI
- Output data is normalized following equation norm_data = data1/data2 * np.mean(data2).

v0.0.4
========
- Add quadratic form to remove background
- Add user peak
- Add mask to only select a given region for fitting
- Save data without running fit again
- Add databroker interface for both SRX and HXN
