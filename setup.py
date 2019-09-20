#!/usr/bin/env python
from __future__ import (absolute_import, division, print_function)

import versioneer

try:
    from setuptools import setup
except ImportError:
    try:
        from setuptools.core import setup
    except ImportError:
        from distutils.core import setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

requirements = ['setuptools'] + requirements

setup(
    name='pyxrf',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    author='Brookhaven National Laboratory',
    author_email='some_email@bnl.gov',
    url='https://github.com/NSLS-II/PyXRF',
    packages=['pyxrf', 'pyxrf.model', 'pyxrf.view', 'pyxrf.db_config', 'configs'],
    entry_points={'console_scripts': ['pyxrf = pyxrf.gui:run']},
    package_data={'pyxrf.view': ['*.enaml'], 'configs': ['*.json']},
    include_package_data=True,
    install_requires=requirements,
    python_requires='>=3.6',
    license='BSD',
    classifiers=['Development Status :: 3 - Alpha',
                 "License :: OSI Approved :: BSD License",
                 "Programming Language :: Python :: 3.7",
                 "Topic :: Software Development :: Libraries",
                 "Intended Audience :: Science/Research"]
)
