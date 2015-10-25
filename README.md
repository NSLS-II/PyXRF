#PyXRF

GUI for visualizing and fitting X-ray fluorescence data.


## Documentation

Tutorial is available at youtube https://www.youtube.com/watch?v=traGVwUP4I0  

You may also refer to http://nbviewer.ipython.org/gist/licode/06654b079fd617aaaeca

Better documentation will be ready soon!


## Installation from Conda

### Linux/Mac
Currently PyXRF only works for python2.0+, not python3.0.
First you need to install [Conda] (http://continuum.io/downloads), then create a conda environment(say pyxrf_test) with python2.7.

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
Launch the software.
```
$pyxrf
```


Reminder:
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
