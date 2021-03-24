## Construction Of Windows Based on Energy - COWBOE

This package consists of the "COWBOE" algorithm to construct windows on the reaction coordinate based on the test PMF values.

## Installation

Run the following to install

'''
pip install cowboe

'''

Module for optimization and selection of parameters for umbrella sampling

##################################################

# COWBOE - Construction Of Windows Based On Energy

##################################################

Current settings for the module are as follow

"PMF unit"                    :        r'PMF - Kcal / (mol $\cdot~\AA^2$)'
"reaction coordinate unit"    :        r'$\AA$'
"polynomial fit order"        :        12 
"param B"                     :        2.0 
"Number of datapoints"        :        100000
"conventional force constant" :        7
"conventional window width"   :        0.5
"conventional no of windows"  :        24
"conv. min of 1st window"     :        2.5
"conv. min of last window"    :        14.5
"fill colour"                 :        'r'
"NM_alpha"                    :        1
"NM_gamma"                    :        2
"NM_beta"                     :        0.5
"NM_delta"                    :        0.5

To update any settings, use dict.update()
        e.g. cowboe_settings.update({"param B" : 2.0001})
        
## usage

from cowboe import pmftopoints, cowboe, cowboefit, cowboeNM, NMprogress, cowboe_settings

cowboe_settings.update({"param B" : 2.0})

A 	= [2.0, 2.9, 3.5]
V 	= [0.75, 0.8700, 0.8000]
fit 	= [21.2642, 17.6376, 24.5880]

pmftopoints(testpmf='test_pmf.txt')

cowboe(A=2.84, V = 0.895 , sc =8)

cowboefit(test='1.txt',bench='bench.txt')

cowboeNM(A = A, V = V, fit = fit)

NMprogress(progressfile = 'progress.txt')

