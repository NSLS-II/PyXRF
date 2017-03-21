PyXRF Release Notes
===================

v0.0.7 to be
------------
- Image normalization and output normalization are synchronized.
- More examples are added, such as batch mode fitting, and preparation for tomography reconstruction (creating movie).
- Only one strategy is selected for summed spectrum fitting.
- More controls to output data to 2D image, and to visualize on GUI
- Output data is normalized following equation norm_data = data1/data2 * np.mean(data2). 


v0.0.4
--------
- Add quadratic form to remove background
- Add user peak
- Add mask to only select a given region for fitting
- Save data without running fit again
- Add databroker interface for both SRX and HXN
