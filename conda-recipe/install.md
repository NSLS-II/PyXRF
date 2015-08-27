Installation Instructions
-------------------------

Basic Installation
------------------

anaconda config --add channels pyxrf
anaconda config --add channels scikit-xray

conda create -n pyxrf pyxrf scikit-xray

anaconda config --add channels pyxrf
anaconda config --add channels scikit-xray

conda create -n pyxrf pyxrf scikit-xray

Installation at NSLS-II
-----------------------

anaconda config --add channels Nikea
anaconda config --add channels $BEAMLINE # where BEAMLINE is CSX, HXN, CHX, etc...
conda create -n pyxrf $BEAMLINE_analysis pyxrf scikit-xray

For example ::

    anaconda config --add channels Nikea
    anaconda config --add channels HXN
    conda create -n pyxrf hxn_analysis pyxrf scikit-xray
