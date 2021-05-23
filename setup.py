from setuptools import setup, find_packages

import versioneer

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

requirements = ["setuptools"] + requirements

setup(
    name="pyxrf",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    author="Brookhaven National Laboratory",
    url="https://github.com/NSLS-II/PyXRF",
    packages=find_packages(),
    entry_points={"console_scripts": ["pyxrf = pyxrf.pyxrf_run:run"]},
    package_data={"configs": ["*.json"], "pyxrf.core": ["*.yaml"]},
    include_package_data=True,
    install_requires=requirements,
    python_requires=">=3.6",
    license="BSD",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Software Development :: Libraries",
        "Intended Audience :: Science/Research",
    ],
)
