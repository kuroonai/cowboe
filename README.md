# COWBOE
## _Construction Of Windows Based on Energy_

[![N|Solid](https://xiresearch.org/wp-content/uploads/2019/11/xiresearch-withcolor5-300x96.png)](https://xiresearch.org/)

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://github.com/kuroonai/cowboe)
## What is cowboe
COWBOE is a python 3 tool for parameter selection in Umbrella Sampling which is a free energy calculation method used in Molecular dynamics simulations.

## Features
- It is a Python 3 module hence easy to install.
- Parameter selection which tunes the total number of windows and force constants used with Umbrella sampling are done.
- Parameters are optimized using Nelder-mead / Restricted Nelder-Mead simplex optimization algorithm.
- Progress of the optimization can be tracked.
- Module includes functions to perform comparison and visualization of different PMFs (Potential of Mean Forces / Free energy files) and the NM algorithm results.

## Installation
`cowboe requires Python V3.6 or higher  to run.`
### Creating a virtual environment ###
It is best to install cowboe in a new environment

**Using venv**
For Linux/macOS,
```sh
python3 -m venv /path/to/new/virtual/environment
source <venv>/bin/activate
```
For Windows,
```sh
c:\>python -m venv c:\path\to\myenv
c:\> <venv>\Scripts\activate.bat
```
**Using conda**
```sh
conda create -n cowboe python=3.7
conda activate cowboe
```
### Installing pip ###
pip is already installed if you are using Python 3 >=3.4 downloaded from python.org or if you are working in a Virtual Environment created by virtualenv or venv. Just make sure to upgrade pip using,
```sh
python3 -m pip install --upgrade pip
```
### Installing cowboe ###
cowboe is available on pypi and can be installed using pip as follows,
```sh
pip install cowboe
```
### Building from source ###
Download the source as a zip file from GitHub and extract the files to a new folder. Before building cowboe install the dependencies using pip as shown below and it will resolve all dependencies conflicts.
```sh
cd <location of extracted files with setup.py file>
pip3 install numpy scipy matplotlib seaborn shapely imageio pandas
python setup.py install
```
After the installation cowboe module should be available and it can be checked by using,
```sh
python -c "import cowboe"
```
Any error means cowboe was not installed sucessfully. 

## Usage
cowboe has different functions which perform individual task like running the cowboe algorithm, performing NM optimization, visualization of the pmf curves and summarizing the NM steps. A detailed explanation and examples with the required data files are provided in the examples directory. A simple but comprehensive example is shown below,
```sh
from cowboe import pmftopoints, cowboe, cowboefit, settings_update
from cowboe import cowboeKS, cowboeRNM, cowboeNM, progressfile, NMprogress, cowboe3Dsurface
from cowboe import cowboe_wham, pmfcompare, multi_pmfcompare, cowboe_settings, wham_settings
from cowboe import cowboe_trajcut, cowboe_OVL, cowboe_pmfplot, pmfdiff

os.chdir('location of the examples folder')
cowboe_settings.update({"param B" : 2.0})
pmftopoints(testpmf='test_pmf.txt')
cowboe(A=3.5, V = 0.8 , sc =8, name=3)
wham_settings.update({"tol" : 0.00015})
cowboe_wham(name = 'benchmark.txt', location ='<cowboe/examples/benchmark>', MCtrials = 0)
cowboe_pmfplot(pmf='1.txt', name='1_pmf', splice=0)
cowboefit(test='3.txt',bench='benchmark.txt')
pmfcompare(pmf1='1.txt', pmf2='3.txt', name='1-3-compare')
pmfdiff(pmf1='1.txt', pmf2='2.txt', name='1-2-compare')
multi_pmfcompare(pmfs=['1.txt', '2.txt', '3.txt'], name='multiple-compare', splices=[0,0,0])

A = [2.0, 2.9, 3.5]
V = [0.75, 0.8700, 0.8000]
fit = [1.9834, 1.3844, 4.7587]

cowboeNM(A = A, V = V, fit = fit)
cowboeRNM(A = A, V = V, fit = fit)

p = np.array(   [[[2.    , 0.75  , 1.9943],
                [2.9   , 0.87  , 1.8232],
                [3.5   , 0.8   , 4.7636]],

                [[2.    , 0.75  , 1.9943],
                 [2.9   , 0.87  , 1.8232],
                 [1.6571, 0.82  , 0.9899]],
         
                [[2.4028, 0.94  , 2.0045],
                 [2.9   , 0.87  , 1.8232],
                 [1.6571, 0.82  , 0.9899]],
         
                [[2.0939, 0.7975, 1.4695],
                 [2.9   , 0.87  , 1.8232],
                 [1.6571, 0.82  , 0.9899]],
         
                [[2.0939, 0.7975, 1.4695],
                 [1.1965, 0.7475, 3.5148],
                 [1.6571, 0.82  , 0.9899]],
         
                [[2.0939, 0.7975, 1.4695],
                 [2.3242, 0.8394, 1.8202],
                 [1.6571, 0.82  , 0.9899]]])
                 
progressfile(points=p)
NMprogress(progressfile = 'progress.txt')

'''
The below function creates a 3d surface of the paramater space using the 
different parameter evaluations, inorder the use this functoin ffmpeg must be installed in the 
path

More information on this is available here,

https://www.ffmpeg.org/
https://www.ffmpeg.org/download.html
https://anaconda.org/conda-forge/ffmpeg
'''

cowboe3Dsurface(progressfile = 'progress.txt')
cowboe_trajcut(percentage=50.0, location='</cowboe/examples/benchmark>',\
            name='benchmark',listfile='list.txt',start=0)
cowboeKS(location='</cowboe/examples/benchmark>', \
         listfile='list.txt', percentage = 85)
cowboe_OVL(location='</cowboe/examples/benchmark>'\
           , listfile='list.txt', name = 'benchmark', distplot=False)
settings_update()
```

## Nelder-Mead simplex algorithm
The primary difference between the NM and restricted NM is that the RNM doesnt include the expansion step after reflection. The cowboeNM and cowboeRNM provide possible NM steps for a given simplex and the user would select a step based on the below pseudocode.

   `Pseudocode of the Simplex Nelder-Mead Optimization`
    
    Initialize the simplex with  n-1 random starting parameter value combinations e.g. [A, V], 
    where n is the number of parameters being optimized.
    
    Restricted Nelder-Mead algorithm:
    
    while loop not done
    
        calculate centroid
        calculate reflected
        
        if reflected is better than best solution then
            calculate expanded
            replace worst solution with better of reflected and expanded
            
        else if reflected is worse than all but worst then
            calculate outward contracted 
            
            if outward contracted is better than reflected
                replace worst solution with outward contracted
            end if 
            
            else
                shrink the search area
            
        else if reflected is worse than all 
            calculate inward contracted
            
            if inward contracted is better than worst
                replace worst solution with inward contracted
            end if
            
            else
                shrink the search area
    
        else
            replace worst solution with reflected
            
        end if
        
        if the solution is within tolerance, exit loop
        
    end loop
    
    return best solution found

cowboe and the WHAM wrapper have been given defaults values which can be modified by changing the cowboe.py file in the installation location or to make temporary changes dict update() of can be used as shown above. More information on this can be obtained by calling the settings_update() function in cowboe.

## License

GNU General Public License v3.0
