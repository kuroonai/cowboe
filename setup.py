from setuptools import setup
from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()
    
setup(
   name='cowboe',
   version='1.0.0',
   description='Construction Of Windows Based On Energy',
   license="GNU General Public License v3.0",
   long_description=long_description,
   long_description_content_type="text/markdown",
   author='Naveen Vasudevan',
   author_email='naveenovan@gmail.com',
   url="https://github.com/kuroonai/cowboe",
   py_modules=['cowboe'],  #same as name
   package_dir={'':'src'},
   data_files = [("", ["LICENSE.txt"])],
   classifiers=[
   "Programming Language :: Python :: 3.5",
   "Programming Language :: Python :: 3.6",
   "Programming Language :: Python :: 3.7",
   "Operating System :: OS Independent",
   "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)"
   ],
   install_requires=['numpy', 'matplotlib','scipy','pandas','shapely']#external packages as dependencies
)
