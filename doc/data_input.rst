====================
Load data into pyxrf
====================


* Define working directory by selecting "Working Directory" button, shown in figure 1.

* Load h5 fine by clicking "Load Data File" button as shown in figure 1. This is an example data
  from HXN beam line at NSLS II. (Currently we only support h5 file.)

.. image:: /_static/define_h5file.jpg
Figure 1. Load h5 file into pyxrf

* Once the data is loaded, you can see the name of the data set, scan2D_19568_sum,
  which is just next to blue label "which channel data for fitting" as shown in figure 2.
  This is the summed data from three vortex detectors.

.. image:: /_static/select_data_plot.png
Figure 2. Select which data set to work on

* In order to plot the spectrum, You can click "Plot Selected Exp. Data"
  on the right part of the GUI, as shown in figure 2.

* As default, we only load the summed spectrum. If you also want to load spectrum from each vortex detector,
  click the checkbox "Load data from each channel", and load h5 file again. Then you can select either the summed
  spectrum or data from each detector to work on, shown in figure 3. Most of the time,
  if detector is well aligned, we only need to load the summed spectrum and work on it.

  .. image:: /_static/more_datasets.png
  Figure 3. Select more data sets from different detectors
