==========================
Frequently asked questions
==========================


Does PyXRF pixel fitting output only the first peak of a given element (i.e., only Ka1 line) or all activated elemental lines (ka1, ka2, kb1...)?
=================================================================================================================================================

The 2D map of fitted elemental result contains all activated elemntal peaks for
a given element, not only the first peak. Please refer to link on comparison
between pyxrf pixel fitting and ROI sum for isolated peak.
(https://github.com/NSLS-II/PyXRF/blob/master/examples/compare_ROI_sum_and_fit.ipynb)
ROI sum is performed for both ka and kb lines, and the result is consistent with fitting
result.


How can I report a bug?
=======================

Please report a bug at github new issues.
https://github.com/NSLS-II/pyxrf/issues/new
