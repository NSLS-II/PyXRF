# PyXRF

[![Build Status](https://travis-ci.com/NSLS-II/PyXRF.svg)](https://travis-ci.com/NSLS-II/PyXRF)

PyXRF is a python-based sophisticated fluorescence analysis package for
fitting and visualizing X-ray fluorescence data.

[PyXRF documentation](http://nsls-ii.github.io/PyXRF/)

[Report an issue with PyXRF](https://github.com/NSLS-II/pyxrf/issues/new)

-------------------------

We support installation on Linux/Mac/Windows, please refer [How to install PyXRF](http://nsls-ii.github.io/PyXRF/installation.html)

## Documentation

### Input of h5 file for PyXRF
Please download a standard h5 file from the link
https://drive.google.com/file/d/0B45Mm22EF9TNQzFkSW0xa01mbkE/view
This h5 file mainly contains spectrum from 3 detectors, positions of x,y motors and scalers for normalization. Please create h5 file with a similar structure in order to use PyXRF.

For test purposes, a parameter file (https://drive.google.com/file/d/0B45Mm22EF9TNYW11OXozRXVic1E/view) is also provided for users to do fitting for this standard h5 file. However, you should never create parameter file manually. Parameter file can be easily created during the step of automatic peak finding.


### Youtube tutorial
The youtube tutorial of pyxrf is at https://www.youtube.com/watch?v=2nFLccehaHI


## Notes

The core fitting functions are a part of the
[scikit-beam](https://github.com/scikit-beam/scikit-beam) data analysis library for x-ray data analysis.
The design philosophy is to separate data fitting and gui, so it is easy to maintain.
