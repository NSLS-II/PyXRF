# PyXRF

[![Build Status](https://travis-ci.com/NSLS-II/PyXRF.svg)](https://travis-ci.com/NSLS-II/PyXRF)

PyXRF is a python-based sophisticated fluorescence analysis package for
fitting and visualizing X-ray fluorescence data.

[PyXRF documentation](http://nsls-ii.github.io/PyXRF/)

[Report an issue with PyXRF](https://github.com/NSLS-II/pyxrf/issues/new)

-------------------------



## Install PyXRF using Conda [Support Linux/Mac/Windows]
PyXRF works for both python 2.7 and 3.5 in Linux/Mac system. For Windows, we only support python 3.5 version at windows 10(64 bit).

### Linux/Mac
First you need to install [conda] (http://continuum.io/downloads). We suggest
anaconda because it is a complete installation of the entire scientific python
stack, but is ~400 MB.  For advanced users, consider downloading [miniconda]
(http://conda.pydata.org/miniconda.html) because it is a smaller download (~20 MB).

Then create a conda environment(say pyxrf_test) with python3.5.
```
$ conda create -n pyxrf_test python=3.5
```
Then go to the environment named pyxrf_test
```
$ source activate pyxrf_test
```
At the same environment, install pyxrf by simply typing
```
$ conda install pyxrf pyqt=4.11 enaml=0.9.8 -c lightsource2-tag -c conda-forge
```

### Windows
Install windows version of anaconda from (http://continuum.io/downloads).
Then you should see Anaconda Prompt installed in your computer. Double click Anaconda Prompt, and type the following lines to finish the installation.

Create a conda environment(say pyxrf_test) with python3.5.
```
$ conda create -n pyxrf_test python=3.5
```
Then go to the environment named pyxrf_test
```
$ activate pyxrf_test
```
At the same environment, install pyxrf by simply typing
```
$ conda install pyxrf xraylib pyqt=4.11 -c lightsource2-tag -c conda-forge
```

## Run PyXRF
At pyxrf_test environment, type
```
$ pyxrf
```

## Update PyXRF
At pyxrf_test environment, type
```
$ conda update pyxrf -c lightsource2-tag
```

#### Reminder
Every time you open a new terminal, make sure to go to pyxrf_test environment first, then launch the software. For linux and mac system, type
```
$ source activate pyxrf_test
```
for windows,
```
activate pyxrf_test        
```
then lanch pyxrf by typing
```
$ pyxrf
```

To leave this environment, at Linux and Mac system just type
```
$ source deactivate
```
and for Windows, type
```
deactivate          
```

## Documentation

### Input of h5 file for PyXRF
Please download a standard h5 file from the link
https://drive.google.com/file/d/0B45Mm22EF9TNQzFkSW0xa01mbkE/view
This h5 file mainly contains spectrum from 3 detectors, positions of x,y motors and scalers for normalization. Please create h5 file with a similar structure in order to use PyXRF.

For test purposes, a parameter file (https://drive.google.com/file/d/0B45Mm22EF9TNYW11OXozRXVic1E/view) is also provided for users to do fitting for this standard h5 file. However, you should never create parameter file manually. Parameter file can be easily created during the step of automatic peak finding.

### Input of spec file for PyXRF
Users can transfer spec file to hdf file that pyxrf can take. Please see examples at https://github.com/NSLS-II/PyXRF/blob/master/examples/specfile_to_hdf.ipynb

### Youtube tutorial
The youtube tutorial of pyxrf is at https://www.youtube.com/watch?v=2nFLccehaHI


## Notes

The core fitting functions are a part of the [scikit-beam]
(https://github.com/scikit-beam/scikit-beam) data analysis library for x-ray data analysis.
The design philosophy is to separate fitting and gui, so it is easy to maintain.
