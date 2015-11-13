#PyXRF

PyXRF is a python-based sophisticated fluorescence analysis package for fitting and visualizing X-ray fluorescence data. This package contains a high-level fitting engine, comprehensive command-line/GUI design, rigorous physics calculation and a powerful visualization interface. PyXRF offers some of the unique features as follows.
- Automatic elements finding: Users do not need to spend extra time selecting elements manually.
- Forward calculation: Users can observe the spectrum from forward calculation at real time while adjusting input parameters. This will help users perform sensitivity analysis, and find an appropriate initial guess for fitting.
- Construct your own fitting algorithm: An advanced mode was created for professional users to construct their own fitting strategies with a full control of each fitting parameter.
- Batch mode: Users can easily perform quick fitting of multiple fluorescence datasets or XANES datasets.
- Interface with NSLS-II database: A specific I/O interface was designed to obtain data directly from BNL/NSLS-II experimental database.


## Documentation

Tutorial is available at youtube https://www.youtube.com/watch?v=traGVwUP4I0  

You may also refer to http://nbviewer.ipython.org/gist/licode/06654b079fd617aaaeca

Better documentation will be ready soon!


## Installation from Conda

### Linux/Mac (EI Captain is working!)
Currently PyXRF only works for python2.0+, not python3.0.
First you need to install [Conda] (http://continuum.io/downloads).

Then create a conda environment(say pyxrf_test) with python2.7.
```
$conda create -n pyxrf_test python=2.7
```
Then go to the environment named pyxrf_test
```
$source activate pyxrf_test
```
At the same environment, install pyxrf by simply typing
```
$conda install -c licode pyxrf
```

### Run
```
$pyxrf
```

#### Update
```
$conda update -c licode pyxrf
```

#### Reminder
Every time you open a new terminal, make sure to go to pyxrf_test environment first, then launch the software.
```
$source activate pyxrf_test
$pyxrf
```
To leave this environment, just type
```
$source deactivate
```


### Windows
Under development.


## Notes

The core fitting functions are written at scikit-xray https://github.com/scikit-xray/scikit-xray.
The design philosophy is to separate fitting and gui, so it is easy to maintain.
For more questions, please submit issues through github, or contact Li at lili@bnl.gov.
