import os
import sys
from setuptools import setup
from os import path
from cowboe import __version__

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

try:
    from setuptools import setup
    from setuptools import Command
    from setuptools import Extension
except ImportError:
    sys.exit(
        "Python library setuptools needs to be installed. "
        "running: python -m ensurepip might resolve the issue"
    )

if "bdist_wheel" in sys.argv:
    try:
        import wheel  
    except ImportError:
        sys.exit(
            "setuptools and wheel packages both need to be installed "
            "running: pip install wheel might resolve the issue"
        )

# Checking for suitable python version
if sys.version_info[:2] < (3, 7):
    sys.stderr.write(
        "cowboe requires Python 3.7 or later. "
        "Python %d.%d is being used currently. Try creating a virtual environment.\n" % sys.version_info[:2]
    )
    sys.exit(1)

setup(
   name='cowboe',
   version=__version__,
   description='Construction Of Windows Based On free Energy',
   license="GNU General Public License v3.0",
   
   long_description=long_description,
   long_description_content_type="text/markdown",
   author='Naveen Vasudevan, Li Xi',
   author_email='naveenovan@gmail.com, xili@mcmaster.ca',
   url="https://github.com/kuroonai/cowboe",
   py_modules=['cowboe'],  #same as name
   package_dir={'':'cowboe'},
   data_files = [("", ["LICENSE.txt"])],
   classifiers=[
   "Development Status :: 5 - Production/Stable",
   "Intended Audience :: Science/Research",
   "Intended Audience :: Developers",
   "Programming Language :: Python",
   "Programming Language :: Python :: 3.7",
   "Programming Language :: Python :: 3.8",
   "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
   "Topic :: Scientific/Engineering",
   "Topic :: Scientific/Engineering :: Chemistry",
   "Topic :: Scientific/Engineering :: Bio-Informatics",
   "Operating System :: OS Independent",
   "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)"
   ],
   install_requires=['matplotlib', 'numpy', 'seaborn', 'imageio', 'scipy', 'shapely', 'pandas'],
   python_requires='>=3.7'
)

