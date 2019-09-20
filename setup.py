from setuptools import find_packages, setup

import versioneer

setup(name='pyxrf',
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      author='Brookhaven National Laboratory',
      packages=find_packages(),
      entry_points={'console_scripts': ['pyxrf = pyxrf.gui:run']},
      package_data={'pyxrf.view': ['*.enaml'], 'configs': ['*.json']},
      include_package_data=True,
      install_requires=['setuptools'],
      license='BSD',
      classifiers=['Development Status :: 3 - Alpha',
                   "License :: OSI Approved :: BSD License",
                   "Programming Language :: Python :: 3.6",
                   "Programming Language :: Python :: 3.7",
                   "Topic :: Software Development :: Libraries",
                   "Intended Audience :: Science/Research"])
