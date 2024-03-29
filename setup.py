from __future__ import print_function
import os
import sys

from setuptools import find_packages, setup


ver_file = os.path.join('imlreliability', '_version.py')
with open(ver_file) as f:
    exec(f.read())

DISTNAME = 'IML_reliability'
DESCRIPTION = 'Empirical study on reliability of IML methods.'

AUTHOR = 'L. GAN'
URL = 'https://github.com/DataSlingers/IML_reliability'
LICENSE = 'MIT'
DOWNLOAD_URL = 'https://github.com/DataSlingers/IML_reliability'
VERSION = __version__
CLASSIFIERS = ['Intended Audience :: Science/Research',
               'Intended Audience :: Developers',
               'License :: OSI Approved',
               'Programming Language :: Python',
               'Topic :: Software Development',
               'Topic :: Scientific/Engineering',
               'Operating System :: Microsoft :: Windows',
               'Operating System :: Unix',
               'Operating System :: MacOS',
               'Programming Language :: Python :: 3.7',
               'Programming Language :: Python :: 3.8']


with open('requirements.txt') as f:
    INSTALL_REQUIRES = [l.strip() for l in f.readlines() if l]

setup(name=DISTNAME,
      author=AUTHOR,
      description=DESCRIPTION,
      license=LICENSE,
      url=URL,
      version=VERSION,
      download_url=DOWNLOAD_URL,
      classifiers=CLASSIFIERS,
      packages=find_packages(),
      install_requires=INSTALL_REQUIRES)