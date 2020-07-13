# Construction Of Windows Based On Energy - COWBOE

This package consists of the "COWBOE" algorithm to construct windows on the reaction coordinate based on the test PMF values.

## Installation

Run the following to install

'''
pip install cowboe

'''

## usage

from cowboe import pmftopoints, cowboe, cowboeNM

pmftopoints(testpmf = 'test_pmf.dat')
cowboe(A=2.84, B = 2.0, V = 0.895 , sc =8)
cowboefit(test = '1.dat', bench='bench.dat')

A = [2.0, 2.9, 3.5]
V = [0.75, 0.8700, 0.8000]
fit = [21.2642, 17.6376, 24.5880]

cowboeNM(A = A, V = V, fit = fit)


