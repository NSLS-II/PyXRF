=======================
Summed spectrum fitting
=======================

Once you select which data to fit (either summed spectrum or from given detector), it
is ready to move on to fitting step. Please click the Fit panel on the upper left corner,
which is next to File IO (Figure 1).

Load parameter file
+++++++++++++++++++

In order to perform data fitting, you need to load a json file into pyxrf by
clicking "Import Parameters" (Figure 1). The json file saves all the
parameters for data fitting.

.. image:: /_static/load_parameter_file.png

Figure 1. Load json file

Plot emission lines
+++++++++++++++++++

Let's first focus on the plotting ("Spectrum View" panel on the right). We can easily zoom in spectrum, and turn on elemental emission lines to
see which element is missing (see Figure 2). You can use up and down arrows in your keyboard
to browse the elements one by one.

After loading parameter file, we can immediately see all the elements are listed in the
"element list window" on the left. Each element includes varies of information including
Z number, elemental name, emission energy [kev], peak intensity, relative intensity(normalized to max),
and cross section [barns/atom] (Figure 2).


.. image:: /_static/zoomin_plot.png

Figure 2. Zoom in plot

Add/Remove elements
+++++++++++++++++++

Here we can clearly see that Fe element is missing. Following the instruction on how to add elements in Figure 2,
we need to first select which elemental line to add, and confirm that by click "Add" button.
Then you will see the element is shown in the element list window.

You can also easily delete any element by clicking the Del button shown in Figure 2.

You may also change the value of peak intensity to see plotting change in real time.

Once you are happy with all the changes, please click "Update" button to save all the changes.
