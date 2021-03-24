#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module for optimization and selection of parameters for umbrella sampling

################################################

COWBOE - Construction Of Windows Based On Energy

################################################

Default settings for the module are as follow

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
"error every"                 :        3
"mark every"                  :        3

To update any settings, use dict.update()
        e.g. cowboe_settings.update({"param B" : 2.0001})

project website: https://github.com/kuroonai/cowboe

Created on Mon Jan 20 15:19:02 2020

@authors:Naveen Kumar Vasudevan, 
        400107764,
        Doctoral Student, 
        The Xi Research Group, 
        Department of Chemical Engineering,
        McMaster University, 
        Hamilton, 
        Canada.
        
        naveenovan@gmail.com
        https://naveenovan.wixsite.com/kuroonai
        
        Dr. Li Xi,
        Assistant Professor
        Department of Chemical Engineering,
        McMaster University, 
        Hamilton, 
        Canada.
        
        xili@mcmaster.ca
        http://xiresearch.org/        

"""
import os
import sys
import math
import glob
import time
import random
import shutil
import pickle
import imageio
import inspect
import json as j
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from datetime import datetime
from scipy.stats import ks_2samp
from matplotlib import animation, cm
from matplotlib import pylab as plt
from math import sqrt, fabs, erf, log
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp1d as inp
from scipy.signal import argrelextrema as extrema


font = {
        'weight' : 'bold',
        'size'   : 10}

matplotlib.rc('font', **font)


def pmftopoints(**kwargs):
    '''
    Takes the test pmf file as input and generates gradient and initial guess
    for windows

    Parameters
    ----------
    testpmf : string, mandatory
            Name of the test pmf file
        
    Returns
    -------
    None.

    '''
    freeenergyfile = kwargs['testpmf']
    
    polyfitorder = cowboe_settings["polynomial fit order"]
    N = cowboe_settings["Number of datapoints"]
    
    location = np.loadtxt(freeenergyfile)[:,0]
    d = np.array([i for i in np.loadtxt(freeenergyfile)[:,1]])
    
    
    # Removing inf values from free energy
    for check in range(len(d)):
        if np.isinf(d[check])==True:
            continue
        else:
            spltice=check
            break
    
    dnoinf = d[spltice::]
    slopetime=location[len(d)-len(dnoinf):]
    
    #polynomial fitting
    p = np.poly1d(np.polyfit(slopetime, dnoinf, polyfitorder))
    d_polyfit_smoothed = p(slopetime)
    d_pol_smoothed = d_polyfit_smoothed
    
    # PMF and smoothened PMF plots
    plt.plot(slopetime,dnoinf,c='r',label='actual')
    plt.plot(slopetime, d_pol_smoothed,c='g',label='polyfit - order = %d'%polyfitorder)
    plt.xlabel(cowboe_settings['reaction coordinate unit'])
    plt.ylabel(cowboe_settings['PMF unit'])
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title('PMF and smoothened curves')
    plt.savefig('PMF-actual+polyfit.{}'.format(cowboe_settings['fig extension']),bbox_inches = 'tight')
    plt.show()
    plt.close()
    
    
    # Calculating and smoothening gradient
    m = np.gradient(d[spltice::], slopetime[1] - slopetime[0])
    m_pol_smooth = np.gradient(d_pol_smoothed, slopetime[1] - slopetime[0])
    np.savetxt('pol_smooth-grad.dat', np.c_[slopetime[:],m_pol_smooth],fmt='%.4f')
    pos = np.loadtxt('pol_smooth-grad.dat')[:,0]
    grad = np.array([abs(i) for i in m])
    Grad_pol_smooth = np.array([abs(i) for i in np.loadtxt('pol_smooth-grad.dat')[:,1]])
    
    # Gradient and smoothened gradient plots
    plt.plot(pos,grad,c='r',label='actual')
    plt.plot(slopetime, Grad_pol_smooth,c='g',label='polyfit - order = %d'%polyfitorder)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel(cowboe_settings['reaction coordinate unit'])
    plt.ylabel(r'$\Delta$ PMF')
    plt.title(r'$\Delta$ PMF and smoothened curves')
    plt.savefig('grdient-actual+polyfit.{}'.format(cowboe_settings['fig extension']),bbox_inches = 'tight')
    plt.show()
    plt.close()
    
    # Flipping the reaction coordinate
    grad_fun_smooth = Grad_pol_smooth
    y = np.flip(grad_fun_smooth)
    x = np.flip(pos)
    
    # Interpolating to get multiple points along the reaction coordinate
    f = inp(x,y, kind='cubic') #used to interpolate function
    x = np.linspace(x[0], x[-1], N)
    y = np.array([f(i) for i in x])
    
        
    # finding the crest and trough of the smoothened curve
    order = 1
    crest, trough = extrema(y, np.greater_equal, order = order )[0],\
            extrema(y, np.less_equal, order = order)[0]
    
    
    # Marking and plotting extreme points and the initial guess windows
    extremes = np.sort(np.concatenate((crest,trough)))
    extreme_values = y[extremes].astype(float)
    
    plt.plot(x,y)
    plt.xlim((x[0]+1, x[-1]-1))
    plt.plot(x[extremes], y[extremes], '*',c ='k')
    plt.ylabel(r'$\Delta$ PMF')
    plt.xlabel(cowboe_settings['reaction coordinate unit'])
    plt.title('Initial window guess')
    
    for exr in x[trough]:
        plt.axvline(exr,ls='--',c='r')    
    for exr in x[crest]:
        plt.axvline(exr,ls='--',c='g')   
    plt.show()
    plt.close()
    
    # Generating and saving the windows bounds for the next step
    bounds = []
    for ext_ind, ext in enumerate(extremes[:-1]):
        newpair = np.arange(extremes[ext_ind],extremes[ext_ind+1]+1)
        bounds.append(tuple(newpair))
    
    bounds = tuple(bounds)
    
    with open('variables.pkl', 'wb') as f:
        pickle.dump([x, y, extremes, extreme_values, crest, trough, bounds], f, protocol=-1)
    
    return None

def cowboe(**kwargs):
    '''
    cowboe algorithm for iteration and window selection

    Parameters
    ----------
    A = float, mandatory
        Optimization parameter for NM algorithm and parameter 1 of cowboe.
        
        
    V = float, mandatory
        Optimization parameter for NM algorithm which controls energy 
        barrier.
    
    sc = int, mandatory
        Sampling considered for each windows in conventional method
        in nano seconds e.g. 8 ns
        
    name = str, mandatory
        Name of the point being evaluated

    Returns
    -------
    None.

    '''
    def Kcalc(windows, A, B, V, kgiven):
        '''
        Calculates the V and K values for the conventional Umbrella sampling.
        V = 0.5 * K * (X - X0)**2
        K = 2*V/(X-X0)**2
        
        '''
        Windows = windows.copy()
        startw = cowboe_settings["conv. min of last window"]
        endw = cowboe_settings["conv. min of 1st window"]
    
        
        Windows[0], Windows[-1]= startw, endw
        
        
        
        V_x = np.linspace(-0.5,0.5,100)
        t_V = [ 0.5*kgiven*(0-i)**2 for i in V_x]
        plt.plot(V_x, t_V)
        plt.axvline(0.0, linestyle='-.', c='k')
        plt.axvline(-0.5, linestyle='--', c='r')
        plt.axvline(0.5, linestyle='--', c='r')
        plt.axhline(t_V[0], linestyle=':', c='g')
        plt.ylabel(r'$\Delta$ V')
        plt.xlabel(cowboe_settings["reaction coordinate unit"])
        plt.title('conventional harmonic potential')
        plt.savefig('nativepotential.{}'.format(cowboe_settings['fig extension']),bbox_inches = 'tight')
        plt.close()
        
        def forceconstant(w):
            wwidth = np.diff(w[::-1])
            k = [2.0*V/(width/2.0)**2 for width in wwidth]
            v = [0.5 * kk * (width/2.0)**2 for kk,width in zip(k,wwidth)]
            return k,v
        
        K, Vs = forceconstant(Windows)
        
        def windowsplot(k, L, R):
            V_x = np.linspace(L,R,100)
            M = (R+L)/2
            dV = [ 0.5*k*(i - M)**2 for i in V_x]
            plt.plot(V_x, dV)
            plt.axvline(M, linestyle='-.', linewidth=0.5,c='k')
            return M
        
        Mss = []
        for k, L, Ri in zip(K, Windows[::-1][:-1], Windows[::-1][1:]):
            Mss.append(windowsplot(k,L,Ri))
        
        
        plt.axhline(V, linestyle='--', c='r')    
        plt.xticks(Mss, rotation=90)
        plt.ylabel(r'$\Delta$ V')
        plt.xlabel(cowboe_settings["reaction coordinate unit"])
        plt.title('Potential from cowboe')
        plt.savefig('native_window_potential_%.2f_%.2f.%s' % (A, B, cowboe_settings['fig extension']), bbox_inches = 'tight')
        plt.show()
        plt.close()
        
        Ms =[]
        k = kgiven
        ww = cowboe_settings["conventional window width"]
        for M in np.arange(cowboe_settings["conv. min of 1st window"],cowboe_settings["conv. min of last window"],ww):
            Ms.append(windowsplot(k,M-ww,M+ww))
        
        
        plt.axhline(0.5*k*ww**2, linestyle='--', c='r')   
        plt.xticks(Ms, rotation=90)
        plt.ylabel(r'$\Delta$ V')
        plt.xlabel(cowboe_settings["reaction coordinate unit"])
        plt.title('conventional potential')
        plt.savefig('native_window_potential_all_%.2f_%.2f.%s' % (A, B, cowboe_settings['fig extension']),bbox_inches = 'tight')
        plt.show()
        plt.close()
        
        
        fig, (ax1,ax2) = plt.subplots(2, sharex=True, figsize=(10,5))
        ax1.bar(np.arange(len(np.diff(Windows[::-1]))), np.diff(Windows[::-1]), color='r')
        ax2.bar(np.arange(len(np.diff(Windows[::-1]))), K, color='g')
        ax1.set(ylabel='Width')
        ax2.set(ylabel='K')
        plt.xticks(np.arange(len(np.diff(Windows[::-1]))))
        plt.xlabel('Window')
        plt.title('Windows/force constant - cowboe')
        plt.savefig('new_window_potential_%.2f_%.2f.%s' % (A, B, cowboe_settings['fig extension']),bbox_inches = 'tight')
        plt.show()
        plt.close()
        
        np.savetxt('K-{}-{}.dat'.format(A,B), np.c_[range(len(K)), K])
        
        return K, Windows, Mss


    
    iniloc = os.getcwd()
    
    A                   = kwargs['A']
    B                   = cowboe_settings['param B']
    V                   = kwargs['V']
    samplingconsidered  = kwargs['sc']
    name                = kwargs['name']
    
    def Kgiven(v):
        return v*2/cowboe_settings['conventional window width']**2
    
    kgiven = Kgiven(V)
    
    def ww(Fmax):

        return round(1/((Fmax/A) + (1/B)), 6)
    

    def close(array_to_check, value):
        return min(enumerate(array_to_check), key=lambda s: abs(s[1] - value))

    def narrow(array_to_check, value):

        if array_to_check[0] < array_to_check[-1]:
            Arr = np.array([entry-value for entry in array_to_check])

            l, r = list(Arr).index(max(Arr[Arr <= 0])), list(
                Arr).index(min(Arr[Arr >= 0]))

            return l, r

        elif array_to_check[0] > array_to_check[-1]:
            Arr = np.array([entry-value for entry in array_to_check])

            l, r = list(Arr).index(max(Arr[Arr <= 0])), list(
                Arr).index(min(Arr[Arr >= 0]))

            return r, l

    def currentmax(begin, end):
        c_max = max(y[begin:end+1])
        return c_max


    def ini_plot(extremes, crest, trough):

        extremes = np.sort(np.concatenate((crest, trough)))

        plt.plot(x, y)
        plt.xlim((x[0]+1, x[-1]-1))
        plt.plot(x[extremes], y[extremes], '*', c='k')
        plt.ylabel(r'$\Delta$ PMF')
        plt.xlabel(cowboe_settings['reaction coordinate unit'])


        for exr in x[trough]:
            plt.axvline(exr, ls='--', c='r')

        for exr in x[crest]:
            plt.axvline(exr, ls='--', c='g')

        plt.savefig('up and down_%.2f_%.2f.%s' % (A, B, cowboe_settings['fig extension']), bbox_inches='tight', dpi=300)
        plt.title('A = %.4f & B = %.4f - initial guess' %(A,B))
        plt.show()
        plt.close()


    
    pointname = name #input('\nEnter name of the point (e.g. 1):\t')
    loc = '{}'.format(pointname)

    if os.path.isdir(loc):
        os.chdir(loc)
    else:
        os.mkdir(loc), os.chdir(loc)

    with open('../variables.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
        x, y, extremes, extreme_values, crest, trough, bounds = pickle.load(f)

    ini_plot(extremes, crest, trough)
    seg = abs(np.diff(x)[0])
    f = inp(x, y, kind='cubic')  # used to interpolate function

    # ALGORITHM

    R = 0
    start = x[R]
    current_max = y[R]
    windows = []
    Rs = []
    windows.append(start)
    Rs.append(R)

    dextremes = dict.fromkeys(crest, 'P')
    dextremes.update(dict.fromkeys(trough, 'V'))

    Rcalc = start - ww(current_max)

    file = True
    if os.path.isfile('LOG_%.2f_%.2f.dat' % (A, B)): os.remove('LOG_%.2f_%.2f.dat' % (A, B))
    if file:

        f = open('LOG_%.2f_%.2f.dat' % (A, B), 'w')
        oldstdout = sys.stdout
        sys.stdout = f

    whilec = -1
    while R <= len(x):
        whilec += 1
        print('\nFor window - {} \n\tstart: {} \twith\t x: {:.6f}'.format(whilec, R, x[R]))
        eranges = np.array([e for e in extremes if x[e] < x[R]])
        print('\tExtreme search range: \n\t\t{}'.format(eranges))
        direction = dict.fromkeys(eranges, 'N')
        print('\tDirection: Made all neutral("N")')
        
    
        for e in eranges:
            print('\n\t\tFor loop with "e" value: ',e)
            Rguess = x[e]
            Rcalc = x[R] - ww(max(y[R:e+1]))
            print('\t\tsearching for max between %d - %d'%(R, e))
            print('\t\tMax value in range is: {:.6f} at {}'.format(max(y[R:e+1]), R+np.argmax(y[R:e+1])))
            print('\t\tThe resulting Width: {}'.format(ww(max(y[R:e+1]))))
            print('\t\tRguess: {:.6f} \t Rcalc: {:.6f}'.format(Rguess, Rcalc))
            
                            
            if Rguess > Rcalc:
                direction[e] = 'R'
                print('\t\tBound is to the right..\n\t\tcontinue..\n')
                continue
            elif Rguess < Rcalc:
                direction[e] = 'L'
                print('\t\tBound is to the left..\n\t\tStopping direction sweep..')
                break

        ir = e 
        il = extremes[ narrow(extremes, e)[0] - 1]
        if il > R : il = il
        elif il <= R : il = R
        print('\n\t\tLeft: {} \t Right: {}\n'.format(il,ir))
                
        print(pd.DataFrame(np.c_[range(R,ir+1),x[R:ir+1], y[R:ir+1], [ww(ic) for ic in y[R:ir+1]]], columns=['Index', 'X', 'Y', 'WW']))
        
        inwhilec = 0
        while True:
                inwhilec += 1
                print('\n\t\tInner While: {}'.format(inwhilec))
                #print(R)
                time.sleep(0)
                im = math.floor((il+ir)/2)    
                if im == len(x)-1: break
                Rg = x[im]
                if R != im+1: 
                    Rc = x[R] - ww(max(y[R:im+1]))
                    print('\t\t\tInitial il: {}\tim: {}\tir: {}'.format(il,im,ir))    
                    print('\t\t\tsearching for max between %d - %d'%(R, im))
                    print('\t\t\tMax value in range is: {:.6f} at {}'.format(max(y[R:im+1]), R+np.argmax(y[R:im+1])))
                    print('\t\t\tThe resulting Width: {:.6f}'.format(ww(max(y[R:im+1]))))
                    print('\t\t\tMid: {}\t\tRc: {:.6f}\t\tRg: {:.6f}'.format(im,Rc,Rg))
                else :
                    Rc = x[R] - ww(y[R])
                    print('\t\t\tInitial il: {}\tim: {}\tir: {}'.format(il,im,ir))    
                    print('\t\t\tsearching for max between %d - %d'%(R, im))
                    print('\t\t\tMax value in range is: {:.6f} at {}'.format(y[R], R))
                    print('\t\t\tThe resulting Width: {:.6f}'.format(ww(y[R])))
                    print('\t\t\tMid: {}\t\tRc: {:.6f}\t\tRg: {:.6f}'.format(im,Rc,Rg))
                
                if Rg > Rc:
                    print('\n\t\t\tRg > Rc: Rg is to the left of calculated\n\t\t\tContinue..\n')
                    time.sleep(0)
                    il = im
                    print('\t\t\tMid point is new Left..')
                    
                    if ir - il == 1 :
                        print('\t\t\tConsecutive points({},{}) have diff direction\n\t\t\tBreaking loop..\n'.format(il,ir))
                        print('\t\t\tY values of consecutive points are il: {:.6f} and ir {:.6f}\n\n'.format(y[il],y[ir]))
                        break
                    else: continue
                    
                elif Rg < Rc:
                    print('\n\t\t\tRg < Rc: Rg is to the right of calculated')
                    time.sleep(0)
                    ir = im
                    print('\t\t\tMid point is new Right..')
                    if ir - il == 1:
                        print('\t\t\tConsecutive points ({},{}) have diff direction\n\t\t\tBreaking loop..\n'.format(il,ir))
                        print('\t\t\tY values of consecutive points are il: {:.6f} and ir {:.6f}\n\n'.format(y[il],y[ir]))
                        time.sleep(0)
                        break
    
                if Rc < x[-1] : break
                if im == len(x)-1: break
            
        if im == len(x)-1: break
                
        if abs(Rc - x[il]) > abs(Rc - x[ir]) :
            R = ir
        elif abs(Rc - x[il]) < abs(Rc - x[ir]) :
            R = il
        
                
        print('\t\t\tFinal il: {}\tim: {}\tir: {}'.format(il,im,ir))    
        windows.append(x[R])
        Rs.append(R)
        print('\t\t\tAppending {:.6f} as window end'.format(x[R]))
        print('\t\t\tR value for next iteration is {}'.format(R))
        print('\nWindow {} is between {:.6f} - {:.6f} at {} - {}'.format(whilec, windows[-2], windows[-1], Rs[-2], Rs[-1]))
        
        
        print('\n\nTotal number of windows = {}\n'.format(len(windows)-1))
    
    
    ###########################

    windows = np.flip(np.unique(np.array(windows)))
    plt.plot(x, y)

    for exr in windows:
        plt.axvline(exr, ls='--', c='r')

    plt.xticks(windows, rotation=90)
    plt.xlim(x[0]+1, x[-1]-1)
    plt.title('A = %.2f & B = %.2f - from cowboe' %(A,B))
    plt.ylabel(r'$\Delta$ PMF')
    plt.xlabel(cowboe_settings['reaction coordinate unit'])
    plt.show()
    plt.close()
    
    Windows = windows.copy()
    startw = cowboe_settings["conv. min of last window"]
    endw = cowboe_settings["conv. min of 1st window"]


    Windows[0], Windows[-1]= startw, endw

    Rpos = np.array(Windows) #np.array(np.flip(windows))
    Windowwidth = np.diff(Rpos)

    total = cowboe_settings['conventional no of windows'] * samplingconsidered  # ns
    # total = 24*4000000*2 #24 windows - 5000000 2fs steps 10 ns
    #Samplingtime = [int((i*total)/(sum(Windowwidth)*400000)) for i in Windowwidth]
    
    if cowboe_settings['equal_sampling']:
        wminus1 = len(windows)-1
        Samplingtime = list(np.full((wminus1,), float(total/wminus1)))
    else:
        fracsamplingtime = [i/sum(Windowwidth) for i in Windowwidth]
        Samplingtime = [(j*total) for j in fracsamplingtime]



    plt.plot(Samplingtime[::-1], 'r^--')
    plt.xlim(-0.25, len(windows)-1.25)
    plt.ylabel('ns', fontsize=14, weight='bold')
    plt.xlabel('Windows', fontsize=14, weight='bold')
    plt.title('A = %.2f & B = %.2f - sampling/window' %(A,B))
    plt.show()
    plt.close()

    # check loop - window width comparison

    actualind = [close(x, i)[0] for i in windows]

    def boundary(indarray):
        bounds = []
        for ext_ind, ext in enumerate(indarray[:-1]):
            newpair = np.arange(indarray[ext_ind], indarray[ext_ind+1]+1)
            bounds.append(np.array(newpair))

        return np.array(bounds, dtype=object)

    bounds = boundary(actualind)

    print('\n')
    for bound in bounds:
        print(bound)
    print('\n')

    maximums = []
    for bound in bounds:
        maximum = max(y[bound])
        maximums.append(maximum)

    actualww = [ww(j) for j in maximums]
    newpositions = [ wws-ac for ac, wws in zip(actualww, windows[:-1])]

    newpositions.insert(0, x[0])

    plt.plot(x, y)
    for exr in windows:
        plt.axvline(exr, ls='--', c='r')

    plt.xticks(windows, rotation=90)
    plt.xlim(x[0]+1, x[-1]-1)

    for exr in newpositions:
        plt.axvline(exr, ls='--', c='g')

    plt.xlim(x[0]+1, x[-1]-1)
    plt.ylabel(r'$\Delta$ PMF')
    plt.xlabel(cowboe_settings['reaction coordinate unit'])
    plt.savefig('alligned_%.2f_%.2f.%s' % (A, B, cowboe_settings['fig extension']), bbox_inches='tight', dpi=300)
    plt.title('A = %.2f & B = %.2f - cowboe vs. brute check' %(A,B))
    plt.show()
    plt.close()


    print('\n', pd.DataFrame(np.c_[actualww, abs(np.diff(windows)), actualww - abs(
        np.diff(windows))], columns=['From Max', 'From calc', 'Abs. diff']))
    print('\n',pd.DataFrame(np.c_[newpositions, windows], columns=['Theoritical', 'Actual']))

    plt.axhline(y=abs(np.diff(x)[0]), linestyle='--', c='r')
    plt.axhline(y=np.diff(x)[0], linestyle='--', c='r')
    Ers = list(actualww - abs(np.diff(windows)))
    if abs(Ers[-1]) > seg:
       plt.plot(Ers[:-1], marker='*', c='k',markersize = 8)
    else:
        plt.plot(list(actualww - abs(np.diff(windows))), marker='*', c='k',markersize = 8)
    plt.xticks(np.arange(0,len(windows)-1))
    plt.title('diff. cowboe vs. brute')
    plt.savefig('difference_%.2f_%.2f.%s' % (A, B, cowboe_settings['fig extension']),bbox_inches = 'tight')
    plt.show()
    plt.close()
    

    K, Windows, Mss = Kcalc(windows, A, B, V, kgiven)
    
    print('\n',pd.DataFrame(np.c_[np.flip(actualww), np.flip(abs(np.diff(windows))), np.flip(actualww - abs(np.diff(windows)))], columns=['Theoritical', 'Actual', 'diff']))

    print('\n',pd.DataFrame(np.c_[np.flip(newpositions), np.flip(windows), np.flip(Windows)], columns=['Theoritical', 'Actual', 'For K']))
    
    print('\n',pd.DataFrame(np.c_[range(len(K)), K], columns=['Window', 'Force constant']))
    
    print('\n\nTotal number of windows = {}\n'.format(len(windows)-1))
    
    forsim = pd.DataFrame(np.c_[np.flip(Windows)[:-1], np.array(Mss) ,np.flip(Windows)[1:], K, np.array(Samplingtime[::-1]) ], columns=['Left', 'Middle', 'Right', 'Force constant','Sampling time'])
    print('\n',forsim)
    np.savetxt('A={}_B={}_V={}.txt'.format(A,B,V), np.c_[np.flip(Windows)[:-1], np.array(Mss) ,np.flip(Windows)[1:], K, np.array(Samplingtime[::-1])], \
               header ='Total number of windows = {}\nSampling time considered = {} ns - for 24 windows\n\
               left\tmiddle\tright\tforce constant\tSampling time'.format(len(windows)-1,samplingconsidered))
    np.save('A = {}_B = {}_V={}'.format(A,B,V), np.c_[np.flip(Windows)[:-1], np.array(Mss) ,np.flip(Windows)[1:], K, np.array(Samplingtime[::-1])])

    
    
    if file:
        f.close()
        sys.stdout = oldstdout
    sys.stdout = oldstdout
    
    
    print('\n',pd.DataFrame(np.c_[np.flip(actualww), np.flip(abs(np.diff(windows))), np.flip(actualww - abs(np.diff(windows)))], columns=['Theoritical', 'Actual', 'diff']))
    
    print('\n',pd.DataFrame(np.c_[np.flip(newpositions), np.flip(windows), np.flip(Windows)], columns=['Theoritical', 'Actual', 'For K']))
    
    print('\n',pd.DataFrame(np.c_[range(len(K)), K], columns=['Window', 'Force constant']))
    
    print('\n\nTotal number of windows = {}\n'.format(len(windows)-1))
    
    print('\n',forsim)
    np.savetxt('A={}_B={}_V={}.txt'.format(A,B,V), np.c_[np.flip(Windows)[:-1], np.array(Mss) ,np.flip(Windows)[1:], K, np.array(Samplingtime[::-1])], \
               header ='Total number of windows = {}\nSampling time considered = {} ns - for 24 windows\n\
               left\tmiddle\tright\tforce constant\tSampling time'.format(len(windows)-1,samplingconsidered))
    np.save('A={}_B={}_V={}'.format(A,B,V), np.c_[np.flip(Windows)[:-1], np.array(Mss) ,np.flip(Windows)[1:], K, np.array(Samplingtime[::-1])])

    os.chdir(iniloc)
    
    print('\nDone!')
    
    return None

def cowboe_wham(**kwargs):
    '''
    WHAM wrapper for PMF generation using trajectory files. 
    Must have wham installed in system path
    
    Grossfield, Alan, “WHAM: the weighted histogram analysis method”, 
    version 2.0.10, http://membrane.urmc.rochester.edu/wordpress/?page_id=126

    Parameters
    ----------
    name : str,
        name for the free energy file.
    location : str
        location of the folder with the trajectory files
    MCtrials : int
        Number of Monte Carlo trails (for bootstrapping). set to 0 if no bootstrapping
        error analysis is required.

    Returns
    -------
    None.

    '''
    currentdir = os.getcwd()
    
    freefile    = kwargs['name']
    loc         = kwargs['location']
    MCnum       = kwargs['MCtrials']
    
    os.chdir(loc)
    
    try: metadatafile = kwargs['metadatafile']
    except:metadatafile = wham_settings["metadatafile"]
    hist_min            = wham_settings["hist_min"]
    hist_max            = wham_settings["hist_max"]
    num_bins            = wham_settings["num_bins"]
    tol                 = wham_settings["tol"]
    temp                = wham_settings["temp"]
    numpad              = wham_settings["numpad"]
    numMCtrials         = MCnum
    randSeed            = random.randint(9999,10000000)
    
    print('\n')
    print("Calling WHAM for PMF generation using trajectory files.\nMust have wham installed in system path\n\ncite:\nGrossfield, Alan, “WHAM: the weighted histogram analysis method”,\nversion 2.0.10, http://membrane.urmc.rochester.edu/wordpress/?page_id=126")
    print('\nUsing: wham %.4f %.4f %d %.6f %.4f %d %s %s %d %d\n'%(hist_min, hist_max, num_bins, tol, temp, numpad, metadatafile, freefile, numMCtrials, randSeed))
    try:
        if metadatafile != 0:
            os.system('wham %.4f %.4f %d %.6f %.4f %d %s %s %d %d > wham_output.txt'%(hist_min, hist_max, num_bins, tol, temp, numpad, metadatafile, freefile, numMCtrials, randSeed))
        else :
            os.system('wham %.4f %.4f %d %.6f %.4f %d %s %s > wham_output.txt'%(hist_min, hist_max, num_bins, tol, temp, numpad, metadatafile, freefile))
        print('\nPMF calculation done!\n\ncopying PMF file to the current directory.')
        shutil.copy(os.path.join(loc,freefile),currentdir)
        os.chdir(currentdir)
    except:
        print('\nwham not found in system path or error copying files')
        os.chdir(currentdir)
        
    
    return None

def pmfcompare(**kwargs):
    '''
    Plots two pmf curves for comparison.

    Parameters
    ----------
    pmf1 : str,
        name of the curve 1 (PMF) file.
    pmf2 : str,
        name of the curve 2 (PMF) file.
    name : str,
        name to save the output with.

    Returns
    -------
    None
    
    '''
    free1 = kwargs['pmf1']
    free2 = kwargs['pmf2']
    pdfname = kwargs['name']
    
    c1 = Path(free1).name.split('.')[0]
    c2 = Path(free2).name.split('.')[0]
    
    
    free1, free2 = np.loadtxt(free1), np.loadtxt(free2)
    f1, f2, e1, e2 = free1[:,0:2], free2[:,0:2], free1[:,2], free2[:,2]
    if cowboe_settings['error bar'] : 
        plt.errorbar(f1[::,0], f1[::,1],yerr=e1,lw=1.5,capsize=2,errorevery=cowboe_settings['error every'],elinewidth=1.5,label=c1)
        plt.errorbar(f2[::,0], f2[::,1],yerr=e2,lw=1.5,capsize=2,errorevery=cowboe_settings['error every'],elinewidth=1.5,label=c2)
    else:
        plt.plot(f1[::,0], f1[::,1],lw=1.5,label=c1)
        plt.plot(f2[::,0], f2[::,1],lw=1.5,label=c2)
    plt.title(r'%s-%s - $\xi$ vs PMF'%(c1,c2))
    plt.xlabel(cowboe_settings["reaction coordinate unit"],fontsize=14,weight='bold')
    plt.ylabel(cowboe_settings["PMF unit"],fontsize=14,weight='bold')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig('{}.{}'.format(pdfname, cowboe_settings['fig extension']), bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()
    
    return None

def pmfdiff(**kwargs):
    '''
    Plots two pmf curves and the abs difference between them.

    Parameters
    ----------
    pmf1 : str,
        name of the curve 1 (PMF) file.
    pmf2 : str,
        name of the curve 2 (PMF) file.
    name : str,
        name to save the output with.

    Returns
    -------
    None
    
    '''
    free1 = kwargs['pmf1']
    free2 = kwargs['pmf2']
    pdfname = kwargs['name']
    
    c1 = Path(free1).name.split('.')[0]
    c2 = Path(free2).name.split('.')[0]
    
    
    free1, free2 = np.loadtxt(free1), np.loadtxt(free2)
    f1, f2, e1, e2 = free1[:,0:2], free2[:,0:2], free1[:,2], free2[:,2]
    if cowboe_settings['error bar'] : 
        plt.errorbar(f1[::,0], f1[::,1],yerr=e1,lw=1.5,capsize=2,errorevery=cowboe_settings['error every'],elinewidth=1.5,label=c1)
        plt.errorbar(f2[::,0], f2[::,1],yerr=e2,lw=1.5,capsize=2,errorevery=cowboe_settings['error every'],elinewidth=1.5,label=c2)
    else:
        plt.plot(f1[::,0], f1[::,1],lw=1.5,label=c1)
        plt.plot(f2[::,0], f2[::,1],lw=1.5,label=c2)
        
    plt.title(r'%s-%s - $\xi$ vs PMF'%(c1,c2))
    plt.xlabel(cowboe_settings["reaction coordinate unit"],fontsize=14,weight='bold')
    plt.ylabel(cowboe_settings["PMF unit"],fontsize=14,weight='bold')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig('{}.{}'.format(pdfname, cowboe_settings['fig extension']), bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()
    
    diff = np.array(f1-f2)[:,1]
    plt.plot(f1[:,0], abs(diff), label='diff', marker = 's', c = 'r')
    plt.title(r'Diff %s-%s - $\xi$ vs PMF'%(c1,c2))
    plt.xlabel(cowboe_settings["reaction coordinate unit"],fontsize=14,weight='bold')
    plt.ylabel(cowboe_settings["PMF unit"],fontsize=14,weight='bold')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig('Diff-{}.{}'.format(pdfname, cowboe_settings['fig extension']), bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()    
    
    return None

def multi_pmfcompare(**kwargs):
    '''
    Plots the pmf curves of multiple files provided through the list.

    Parameters
    ----------
    pmfs : list or tuple,
        names of the PMF files to be plotted.

    name : str,
        name to save the output with.
    
    splices: int,
        index value for each input pmf curve to splice unwanted points.

    Returns
    -------
    None
    
    '''
    frees = kwargs['pmfs']
    pdfname = kwargs['name']
    splices = kwargs['splices']
    
    
    for free1, splice  in zip(frees,splices):
        c1 = Path(free1).name.split('.')[0]
    
        free1 = np.loadtxt(free1)[splice:]
        f1, e1= free1[:,0:2], free1[:,2]
        if cowboe_settings['error bar'] : 
            plt.errorbar(f1[::,0], f1[::,1],yerr=e1,lw=1.5,capsize=2,errorevery=cowboe_settings['error every'],elinewidth=1.5,label=c1)
        else:
            plt.plot(f1[::,0], f1[::,1],lw=1.5,label=c1)
        plt.title(r'%s- $\xi$ vs PMF'%(pdfname))
        plt.xlabel(cowboe_settings["reaction coordinate unit"],fontsize=14,weight='bold')
        plt.ylabel(cowboe_settings["PMF unit"],fontsize=14,weight='bold')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
    plt.savefig('{}.{}'.format(pdfname, cowboe_settings['fig extension']), bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()
    
    return None

def cowboefit(**kwargs):
    '''
    Finds the fitness or area difference between the test and benchmark pmf curves
    Parameters
    ----------
    test : str, mandatory
        name of the pmf curve being tested ( point name used in pmftopoints() ).
    
    bench : str, mandatory
        name of the benchmarck pmf curve
    Returns
    -------
    None.

    '''
    
    def removeinf_and_gradient(freefile):
        '''
        Removes inf entries in the PMF file and calculates the gradient of the same.
    
        Parameters
        ----------
        freefile : str, mandatory
            removes inf entries in the pmf file and computes the gradient of the PMF.
    
        Returns
        -------
        Str, 
        
        Name of the clean PMF file.
    
        '''
    
        f = np.loadtxt(freefile)
        location, d, e = f[:,0], f[:,1], f[:,2]
        
        for check in range(len(d)):
    
               if np.isinf(d[check])==True:
                   continue
               else:
                   splice=check
                   break
               
        # for check in range(len(d)):
    
        #        if np.isinf(d[check])==True or np.isnan(e[check]):
        #            continue
        #        else:
        #            splice=check
        #            break   
               
        #m=np.gradient(d[splice::], location[splice+1] - location[splice])
        cleanedf = np.c_[f[splice:,0], f[splice:,1], f[splice:,2], f[splice:,3]]
        np.savetxt('clean_{}'.format(Path(freefile).name),cleanedf,fmt='%.4f')
        #slopetime=location[100-len(m):]
        #gradient = np.c_[slopetime,m]
        #np.savetxt('gradient-{}'.format(freefile),gradient,fmt='%.4f')
        
        return 'clean_{}'.format(Path(freefile).name)
    
    def pmfplot(free1, free2, pdfname):
        '''
        Plots the error bars of the two curves
    
        Parameters
        ----------
        free1 : str,
            name of the curve 1 (PMF) file.
        free2 : str,
            name of the curve 2 (PMF) file.
        pdfname : str,
            name to save the output with.
    
        Returns
        -------
        None
        
        '''
        c1 = 'benchmark'
        c2 = 'test'
        
        
        free1, free2 = np.loadtxt(free1), np.loadtxt(free2)
        f1, f2, e1, e2 = free1[:,0:2], free2[:,0:2], free1[:,2], free2[:,2]
        if cowboe_settings['error bar'] : 
            plt.errorbar(f1[::,0], f1[::,1],yerr=e1,lw=1.5,capsize=2,errorevery=cowboe_settings['error every'],elinewidth=1.5,label=c1)
            plt.errorbar(f2[::,0], f2[::,1],yerr=e2,lw=1.5,capsize=2,errorevery=cowboe_settings['error every'],elinewidth=1.5,label=c2)
        else:
            plt.plot(f1[::,0], f1[::,1],lw=1.5,label=c1)
            plt.plot(f2[::,0], f2[::,1],lw=1.5,label=c2)
        plt.title(r'%s-%s - $\xi$ vs PMF'%(c1,c2))
        plt.xlabel(cowboe_settings["reaction coordinate unit"],fontsize=14,weight='bold')
        plt.ylabel(cowboe_settings["PMF unit"],fontsize=14,weight='bold')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.savefig('{}.{}'.format(pdfname, cowboe_settings['fig extension']), bbox_inches='tight', dpi=300)
        plt.show()
        plt.close()
    
    def pmf_vdist(gfile, gfile2, savefile) :
        '''
        Finds the maximum vertical distance between curves along the reaction coordinate.
    
        Parameters
        ----------
        gfile : str,
            name of the curve 1 (PMF) file.
        gfile2 : str,
            name of the curve 2 (PMF) file.
        savefile : str,
            name to save the output with.
    
        Returns
        -------
        None
            
    
        '''
        def json_dump(d,filename):
            '''
            Dumps a dict into a json file with the given name    
        
            Parameters
            ----------
            d : dict
                dictionary variable which needs to be dumped as a json file.
            filename : str
                name for the json file.
        
            Returns
            -------
            None.
        
            '''
            json = j.dumps(d)
            f = open("%s.json"%filename,"w")
            f.write(json)
            f.close()
            
            return None
    
        def pickle_dump(d, filename):
            '''
            Dumps a dict into a pickle file with the given name    
        
            Parameters
            ----------
            d : dict
                dictionary variable which needs to be dumped as a pickle file.
            filename : str
                name for the pickle file.
        
            Returns
            -------
            None.
        
            '''
            with open(filename+'.p','wb') as f:
                pickle.dump(d,f)
            
            return None

        
        curve1=np.loadtxt(gfile)
        curve2=np.loadtxt(gfile2)
        
        for check in range(len(curve1)):
        
                if np.isinf(curve1[check,1])==True:
                    continue
                else:
                    splice=check
                    break
        curve1=curve1[splice:]
        
        for check in range(len(curve2)):
        
                if np.isinf(curve2[check,1])==True:
                    continue
                else:
                    splice=check
                    break
        curve2=curve2[splice:]
            
        if len(curve2) > len(curve1): 
            start = np.where(curve1[0,0] == curve2[:,0]) #curve2 longer
            start = int(start[0])
            stop = np.where(curve1[-1,0] == curve2[:,0])
            stop = int(stop[0])+1
            curve2 = curve2[start:stop]       
        
        elif len(curve2) < len(curve1): #
            start = np.where(curve2[0,0] == curve1[:,0]) #curve1 longer
            start = int(start[0])
            stop = np.where(curve2[-1,0] == curve1[:,0])
            stop = int(stop[0])+1
            curve1 = curve1[start:stop]   
        
        cleanc1, cleanc2 = curve1.copy(), curve2.copy()
        
        f1, e1 =  curve1[:,0:2], curve1[:,2] #these are your points for curve 1 
        f2, e2 =  curve2[:,0:2], curve2[:,2] #these are your points for curve 2 
        
        # vdistall = np.array([ abs(i-j) for i,j in zip(f1[:,1],f2[:,1])])
        # vdist1 = []
        
        # for ind,e in enumerate(e1):
            
        #     if np.isnan(e):
        #         #vdist1.append(vdistall[ind]/1.0)
        #         raise Exception("There is a nan entry in the benchmark error\nKindly reevaluate the error\n")
        #     else:
        #         vdist1.append(vdistall[ind]/e)
        #         #vdist1.append(0.0)
                            
        # vdist = max(vdist1)
        # vdistpos = np.argmax(np.array(vdist1))
        
        vdist1 = []  
        
        for tf1, tf2, te1 in zip(f1[:,1],f2[:,1],e1):
            if tf2 > tf1:
                vdist1.append(tf2-(tf1))
            elif tf2 < tf1:
                vdist1.append((tf1)-tf2)
            else:
               vdist1.append(0)
               
        # for tf1, tf2, te1 in zip(f1[:,1],f2[:,1],e1):
        #     if tf2 > tf1+te1:
        #         vdist1.append(tf2-(tf1+te1))
        #     elif tf2 < tf1-te1:
        #         vdist1.append((tf1-te1)-tf2)
        #     else:
        #        vdist1.append(0)   
               
        vdist = max(vdist1)
        vdistpos = np.argmax(np.array(vdist1))
        
        # for ff,ss in zip(f1[:,0], vdist1):
        #     print(ff,ss)
        
        c1 = 'benchmark'
        c2 = 'test'
        
        print("\nFor the %s curve:\n\tThe maximum normalized vertical distance between the curves is:\n\t\t%f at %.4f between (%.4f and %.4f)"%(c2, vdist, f2[vdistpos,0], curve1[0,0], curve1[-1,0]))
        if cowboe_settings['error bar'] : 
            plt.errorbar(f1[::,0], f1[::,1],yerr=e1,lw=1.5,capsize=2,errorevery=cowboe_settings['error every'],elinewidth=1.5,label=c1)
            plt.errorbar(f2[::,0], f2[::,1],yerr=e2,lw=1.5,capsize=2,errorevery=cowboe_settings['error every'],elinewidth=1.5,label=c2)
        else:
            plt.plot(f1[::,0], f1[::,1],lw=1.5,label=c1)
            plt.plot(f2[::,0], f2[::,1],lw=1.5,label=c2)
        # plt.plot(f2[vdistpos, 0], f2[vdistpos, 1], markersize=3, marker='^', c='g')
        # plt.plot(f1[vdistpos, 0], f1[vdistpos, 1], markersize=3, marker='^', c='g')
        try:
            plt.annotate(text='', xy=(f2[vdistpos, 0], f2[vdistpos, 1]), xytext=(f1[vdistpos, 0], f1[vdistpos, 1]), arrowprops=dict(arrowstyle='<->'))
        except:
            plt.annotate(s='', xy=(f2[vdistpos, 0], f2[vdistpos, 1]), xytext=(f1[vdistpos, 0], f1[vdistpos, 1]), arrowprops=dict(arrowstyle='<->'))
        plt.title(r'%s-%s - $\xi$ vs PMF'%(c1,c2))
        plt.xlabel(cowboe_settings["reaction coordinate unit"],fontsize=14,weight='bold')
        plt.ylabel(cowboe_settings["PMF unit"],fontsize=14,weight='bold')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.annotate('max_vdist() = {:.4f}'.format(vdist), # this is the text
                           (max(cleanc1[:,0])-2,min(cleanc2[:,1])+2), # this is the point to label
                           textcoords="offset points", # how to position the text
                           xytext=(0,0), # distance from text to points (x,y)
                           ha='center',bbox=dict(boxstyle='round,pad=0.2', fc='red', alpha=0.3))
        
        plt.savefig(savefile+'.{}'.format(cowboe_settings['fig extension']), bbox_inches='tight', dpi=300)
        plt.show()
        plt.close()
        
        output = {
            'max_vdist':vdist,
            'rc_range':[curve1[0,0], curve1[-1,0]]
            }
        
        #pickle_dump(output,savefile)
        json_dump(output,savefile)
        
        return output
    
    testfile = kwargs['test']
    benchfile = kwargs['bench']
    
    name = Path(testfile).name.split('.')[0]
    
    benchfile = removeinf_and_gradient(benchfile)
    testfile = removeinf_and_gradient(testfile)
    #pmfplot(benchfile, testfile, 'errorbar_pmf_{}'.format(name))
    outp = pmf_vdist(benchfile, testfile, 'vdist_pmf_{}'.format(name))
    #pmf_errordist(benchfile, testfile, 'vdist_pmf_{}'.format(name))
    #pmfarea(benchfile, testfile, 'area_pmf_{}'.format(name))

    
    return outp

def cowboeNM(**kwargs) :
    
    '''
    Nelder-Mead optimization algorithm for the cowboe module.

    Parameters
    ----------
    A = array, mandatory
        A values of the 3 initial points for the 2 parameter optimization.
        
    V = array, mandatory
        V or energy barrier values of the 3 initial points for 
        the 2 parameter optimization.
    
    fit = array, mandatory
        fitness or the area difference value between the benchmark
        and the test case.
        
    Returns
    -------
    conv = dict.
            dictionary with possible moves for the current simplex
    
    #################################################################
    
    Pseudocode of the Simplex Nelder-Mead Optimization
    
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

    '''
    
    
    A = kwargs['A']
    V = kwargs['V']
    fit = kwargs['fit']
    
    
    logA        = {1:np.log(A[0])    ,    2:np.log(A[1])   ,  3:np.log(A[2]) } 
    V           = {1:V[0]           ,    2:V[1]           ,  3:V[2]}
    fit_raw = {'fit_1': fit[0],  'fit_2': fit[1],  'fit_3': fit[2]  }
    
    area        =   list(fit_raw.values())
    F = {1:area[0],      2:area[1],      3:area[2]} #area-pmf
    
    fitfunc = {}
    
    
    def centroid(b, o):  #best, other
        [x1,y1] = b
        [x2,y2] = o
        
        xc = (x1+x2)/2
        yc = (y1+y2)/2
        
        return [xc, yc]
    
    def step_plot(best, worst, other):
    
        Ax = [logA[1],logA[2],logA[3]]
        Vy = [V[1], V[2], V[3]]
        Fu = [F[1], F[2], F[3]]
        
        plt.plot(Ax, Vy,'k.',markersize='5')
        #plt.title('Step - I')
        plt.xlabel('ln A')
        plt.ylabel(r'$\Delta$ V')
        plt.plot([Ax[0],Ax[1]], [Vy[0], Vy[1]], 'k-')
        plt.plot([Ax[1],Ax[2]], [Vy[1], Vy[2]], 'k-')
        plt.plot([Ax[2],Ax[0]], [Vy[2], Vy[0]], 'k-')
        plt.ylim(0, .3)
        plt.xlim((0.8,1.7))
        for x,y,l in zip(Ax,Vy,range(len(Ax))):

            
            if l+1 == best: 
                la = 'best ({:.4f},{:.4f})'.format(np.exp(x),y)
                label = '{:.4f}'.format(Fu[l])
                fontc = 'g'
                plt.plot(Ax[l], Vy[l],'g^',markersize='10', label=la)
                
            elif l+1 ==worst : 
                la = 'worst ({:.4f},{:.4f})'.format(np.exp(x),y)
                label = '{:.4f}'.format(Fu[l])
                fontc = 'r'
                plt.plot(Ax[l], Vy[l],'r^',markersize='10',label=la)
    
            else : 
                la = 'other ({:.4f},{:.4f})'.format(np.exp(x),y)
                label = '{:.4f}'.format(Fu[l])
                fontc = 'y'
                plt.plot(Ax[l], Vy[l],'y^',markersize='10',label=la)

        
            plt.annotate(label, # this is the text
                         (x,y), # this is the point to label
                         textcoords="offset points", # how to position the text
                         xytext=(0,5),
                         color = fontc,# distance from text to points (x,y)
                         ha='center')
            
        
        cen = centroid([logA[best], V[best]], [logA[other], V[other]])
        plt.plot(cen[0],cen[1], 'b^',markersize='10', label='centroid ({:.4f},{:.4f})'.format(np.exp(cen[0]), cen[1]))
        # x,y = cen[0],cen[1]
        # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        if inspect.stack()[1].function != 'Shrink':
            #plt.plot(cen[0],cen[1], 'b^',markersize='10', label='centroid({:.4f},{:.4f})'.format(cen[0], cen[1]))
            x,y = cen[0],cen[1]
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            # plt.annotate('centroid', # this is the text
            #      (x,y), # this is the point to label
            #      textcoords="offset points", # how to position the text
            #      xytext=(0,10), # distance from text to points (x,y)
            #      ha='center',bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3))  
        
        return None
        
    def Reflection(logA, V, F, alpha):
    
        # ranking points based on F value -lower is closer
        
        best     = min(F, key=F.get)
        worst    = max(F, key=F.get)
        other    = 6 - (best+worst)
    
        bes = [logA[best], V[best]]
        wor = [logA[worst], V[worst]] 
        oth = [logA[other], V[other]]
        
        
        fitfunc.update({'best':F[best]})
        fitfunc.update({'worst':F[worst]})
        fitfunc.update({'other':F[other]})
        #step -  I
        
        step_plot(best, worst, other)
        cen = centroid([logA[best], V[best]], [logA[other], V[other]])
    
       
        x,y = cen[0],cen[1]
        # plt.annotate('centroid', # this is the text
        #          (x,y), # this is the point to label
        #          textcoords="offset points", # how to position the text
        #          xytext=(0,10), # distance from text to points (x,y)
        #          ha='center',bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3))
    
        ref = [cen[0] + alpha*(cen[0] - logA[worst]), cen[1] + alpha*(cen[1] - V[worst])]
        plt.plot(ref[0],ref[1], 'g*',markersize='10', label='Reflected ({:.4f},{:.4f})'.format(np.exp(ref[0]), ref[1]))
        
        x,y = ref[0],ref[1]
        # plt.annotate('Reflected', # this is the text
        #          (x,y), # this is the point to label
        #          textcoords="offset points", # how to position the text
        #          xytext=(0,10), # distance from text to points (x,y)
        #          ha='center',bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3))
    
        
    
        plt.plot([ref[0],wor[0]], [ref[1],wor[1]], 'k--')
        
        plt.xlim((min(bes[0],wor[0],oth[0],ref[0])-0.1 , max(bes[0],wor[0],oth[0],ref[0])+0.1))
        plt.ylim((min(bes[1],wor[1],oth[1],ref[1])-0.1 , max(bes[1],wor[1],oth[1],ref[1])+0.1))
        plt.title('reflection')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig('reflected.{}'.format(cowboe_settings['fig extension']),bbox_inches='tight', dpi=300)
        plt.show()
        plt.close()
        
        return cen, ref, best, worst, other, bes, wor, oth
    
    def Expansion(cen, ref, best, worst, other, gamma):
    
        [xr,yr] = ref
        [xc, yc] = cen
        
        [xe,ye] = [xc + gamma*(xr - xc), yc + gamma*(yr - yc)]
        
        step_plot(best, worst, other)
        x,y = xe,ye
        
        exp = [x,y]
        
        plt.plot(xe,ye, 'b*',markersize='10', label = 'expanded ({:.4f},{:.4f})'.format(np.exp(xe),ye))
        
        wor = [logA[worst], V[worst]]
        plt.plot([exp[0],wor[0]], [exp[1],wor[1]], 'k--')
    
        
        #plt.annotate('Expanded', # this is the text
        #         (x,y), # this is the point to label
        #        textcoords="offset points", # how to position the text
        #        xytext=(0,10), # distance from text to points (x,y)
        #        ha='center',bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3))
        
        plt.xlim((min(bes[0],wor[0],oth[0],exp[0])-0.1 , max(bes[0],wor[0],oth[0],exp[0])+0.1))
        plt.ylim((min(bes[1],wor[1],oth[1],exp[1])-0.1 , max(bes[1],wor[1],oth[1],exp[1])+0.1))
        plt.title('expansion')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig('expanded.{}'.format(cowboe_settings['fig extension']),bbox_inches='tight', dpi=300)
        plt.show()
        plt.close()
        
        return exp
    
    def inner_contraction(cen, wor, best, worst, other, beta):
        
        [xc,yc],[xw,yw] = cen,wor
        in_con = [xc + beta*(xw - xc), yc + beta*(yw - yc)]
        
        [xic, yic] = in_con
        
        step_plot(best, worst, other)
    
        plt.plot(xic,yic, 'k*',markersize='10', label = 'inner_c ({:.4f},{:.4f})'.format(np.exp(xic),yic))
        plt.plot([xic,xw], [yic,yw], 'k--')
        plt.plot([xic,xc], [yic,yc], 'k--')
    
    
        
        # plt.annotate('inner_c', # this is the text
        #          (xic, yic), # this is the point to label
        #          textcoords="offset points", # how to position the text
        #          xytext=(0,10), # distance from text to points (x,y)
        #          ha='center',bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3))
        
        plt.xlim((min(bes[0],wor[0],oth[0],in_con[0])-0.1 , max(bes[0],wor[0],oth[0],in_con[0])+0.1))
        plt.ylim((min(bes[1],wor[1],oth[1],in_con[1])-0.1 , max(bes[1],wor[1],oth[1],in_con[1])+0.1))
        plt.title('inner contraction')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig('inner_c.{}'.format(cowboe_settings['fig extension']),bbox_inches='tight', dpi=300)
        plt.show()
        plt.close()
        
        return in_con
        
    def outer_contraction(ref, cen, wor,  best, worst, other, beta ):
        
        [xr,yr], [xc, yc] = ref, cen
        out_con = [xc + beta*(xr - xc), yc + beta*(yr - yc)]
        
        [xoc, yoc] = out_con
        
        step_plot(best, worst, other)
    
        
        plt.plot(xoc,yoc, 'c*',markersize='10', label='outer_c ({:.4f},{:.4f})'.format(np.exp(xoc),yoc))
        
        plt.plot([out_con[0],wor[0]], [out_con[1],wor[1]], 'k--')
    
        
        # plt.annotate('outer_c', # this is the text
        #          (xoc, yoc), # this is the point to label
        #          textcoords="offset points", # how to position the text
        #          xytext=(0,10), # distance from text to points (x,y)
        #          ha='center',bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3))
        
        plt.xlim((min(bes[0],wor[0],oth[0],out_con[0])-0.1 , max(bes[0],wor[0],oth[0],out_con[0])+0.1))
        plt.ylim((min(bes[1],wor[1],oth[1],out_con[1])-0.1 , max(bes[1],wor[1],oth[1],out_con[1])+0.1))
        plt.title('outer contraction')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig('outer_c.{}'.format(cowboe_settings['fig extension']),bbox_inches='tight', dpi=300)
        plt.show()
        plt.close()
        
        return out_con
    
    def Shrink(wor, oth, best, worst, other, delta):
        [xb,yb] = bes
        [xw,yw] = wor
        [xo,yo] = oth
        
        step_plot(best, worst, other)
        s_wor = [xb + delta *(xw-xb), yb + delta*(yw-yb)]
        s_oth = [xb + delta *(xo-xb), yb + delta*(yo-yb)]
        
        x,y = s_wor[0], s_wor[1]
        plt.plot(x,y, 'm*',markersize='10', label='s_worst ({:.4f},{:.4f})'.format(np.exp(x),y))   
        # plt.annotate('s_worst', # this is the text
        #          (x, y), # this is the point to label
        #          textcoords="offset points", # how to position the text
        #          xytext=(0,10), # distance from text to points (x,y)
        #          ha='center',bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3))
        
        x,y = s_oth[0], s_oth[1]
        plt.plot(x,y, 'k*',markersize='10', label='s_other ({:.4f},{:.4f})'.format(np.exp(x),y))   
        # plt.annotate('s_other', # this is the text
        #          (x, y), # this is the point to label
        #          textcoords="offset points", # how to position the text
        #          xytext=(0,10), # distance from text to points (x,y)
        #          ha='center',bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3))    
        
        plt.xlim((min(bes[0],wor[0],oth[0],s_wor[0], s_oth[0])-0.1 , max(bes[0],wor[0],oth[0],s_wor[0], s_oth[0])+0.1))
        plt.ylim((min(bes[1],wor[1],oth[1],s_wor[1], s_oth[1])-0.1 , max(bes[1],wor[1],oth[1],s_wor[1], s_oth[1])+0.1))
        plt.title('shrinking')       
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig('shrink.{}'.format(cowboe_settings['fig extension']),bbox_inches='tight', dpi=300)
        plt.show()
        plt.close()
        
        return s_wor, s_oth

        
    def convertor(p):
        
        return([np.exp(p[0]), p[1]])
    
    def A_convertor(ref, exp, in_con, out_con, s_wor, s_oth):
        
        c_ref       = convertor(ref)
        c_exp       = convertor(exp)
        c_in_con    = convertor(in_con)
        c_out_con   = convertor(out_con)
        c_s_wor     = convertor(s_wor)
        c_s_oth     = convertor(s_oth)
        
        return {
            'reflection':c_ref, 
            'expansion':c_exp, 
            'in_contract':c_in_con, 
            'out_contract':c_out_con, 
            'shrink_worst':c_s_wor, 
            'shrink_other':c_s_oth
            }
    
    # def A_convertor(ref, in_con, out_con, s_wor, s_oth):
        
    #     c_ref       = convertor(ref)
    #     #c_exp       = convertor(exp)
    #     c_in_con    = convertor(in_con)
    #     c_out_con   = convertor(out_con)
    #     c_s_wor     = convertor(s_wor)
    #     c_s_oth     = convertor(s_oth)
        
    #     return {
    #         'reflection':c_ref, 
    #         #'expansion':c_exp, 
    #         'in_contract':c_in_con, 
    #         'out_contract':c_out_con, 
    #         'shrink_worst':c_s_wor, 
    #         'shrink_other':c_s_oth
    #         }
        
    
    alpha   = cowboe_settings['NM_alpha']
    gamma   = cowboe_settings['NM_gamma']
    beta    = cowboe_settings['NM_beta']
    delta   = cowboe_settings['NM_delta']
    
    cen, ref, best, worst, other, bes, wor, oth = Reflection(logA, V, F, alpha)
    exp = Expansion(cen, ref, best, worst, other, gamma)
    in_con = inner_contraction(cen, wor, best, worst, other, beta)
    out_con = outer_contraction(ref, cen, wor, best, worst, other, beta )
    s_wor, s_oth = Shrink(wor, oth, best, worst, other, delta)
    
    conv = A_convertor(ref, exp, in_con, out_con, s_wor, s_oth)
    # conv = A_convertor(ref, in_con, out_con, s_wor, s_oth)
    print('\n')
    print('{}\t\t{}\t{}'.format('Move','A','V'))
    print('==============================')
    for k, v in conv.items():
        conv[k] = [round(i, 4) for i in v]
        print('{}\t{}\t{}'.format(k,conv[k][0],conv[k][1]))
    print('==============================')
    
    return conv

def cowboeRNM(**kwargs) :
    '''
    (Restricted) Nelder-Mead optimization algorithm for the cowboe module  .

    Parameters
    ----------
    A = array, mandatory
        A values of the 3 initial points for the 2 parameter optimization.
        
    V = array, mandatory
        V or energy barrier values of the 3 initial points for 
        the 2 parameter optimization.
    
    fit = array, mandatory
        fitness or the area difference value between the benchmark
        and the test case.
        
    Returns
    -------
    conv = dict.
            dictionary with possible moves for the current simplex
            
    
    Pseudocode of the Simplex Nelder-Mead Optimization
    
    Initialize the simplex with  n-1 random starting parameter value combinations e.g. [A, V], 
    where n is the number of parameters being optimized.
    
    Restricted Nelder-Mead algorithm:
    
    while loop not done
    
        calculate centroid
        calculate reflected
        
        if reflected is better than best solution then
            replace worst solution with reflected
            
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
    '''
    
    
    A = kwargs['A']
    V = kwargs['V']
    fit = kwargs['fit']
    
    
    logA        = {1:np.log(A[0])    ,    2:np.log(A[1])   ,  3:np.log(A[2]) } 
    V           = {1:V[0]           ,    2:V[1]           ,  3:V[2]}
    fit_raw = {'fit_1': fit[0],  'fit_2': fit[1],  'fit_3': fit[2]  }
    
    area        =   list(fit_raw.values())
    F = {1:area[0],      2:area[1],      3:area[2]} #area-pmf
    
    fitfunc = {}
    
    
    def centroid(b, o):  #best, other
        [x1,y1] = b
        [x2,y2] = o
        
        xc = (x1+x2)/2
        yc = (y1+y2)/2
        
        return [xc, yc]
    
    def step_plot(best, worst, other):
    
        Ax = [logA[1],logA[2],logA[3]]
        Vy = [V[1], V[2], V[3]]
        Fu = [F[1], F[2], F[3]]
        
        plt.plot(Ax, Vy,'k.',markersize='5')
        #plt.title('Step - I')
        plt.xlabel('ln A')
        plt.ylabel(r'$\Delta$ V')
        plt.plot([Ax[0],Ax[1]], [Vy[0], Vy[1]], 'k-')
        plt.plot([Ax[1],Ax[2]], [Vy[1], Vy[2]], 'k-')
        plt.plot([Ax[2],Ax[0]], [Vy[2], Vy[0]], 'k-')
        plt.ylim(0, .3)
        plt.xlim((0.8,1.7))
        for x,y,l in zip(Ax,Vy,range(len(Ax))):

            
            if l+1 == best: 
                la = 'best ({:.4f},{:.4f})'.format(np.exp(x),y)
                label = '{:.4f}'.format(Fu[l])
                fontc = 'g'
                plt.plot(Ax[l], Vy[l],'g^',markersize='10', label=la)
                
            elif l+1 ==worst : 
                la = 'worst ({:.4f},{:.4f})'.format(np.exp(x),y)
                label = '{:.4f}'.format(Fu[l])
                fontc = 'r'
                plt.plot(Ax[l], Vy[l],'r^',markersize='10',label=la)
    
            else : 
                la = 'other ({:.4f},{:.4f})'.format(np.exp(x),y)
                label = '{:.4f}'.format(Fu[l])
                fontc = 'y'
                plt.plot(Ax[l], Vy[l],'y^',markersize='10',label=la)

        
            plt.annotate(label, # this is the text
                         (x,y), # this is the point to label
                         textcoords="offset points", # how to position the text
                         xytext=(0,5),
                         color = fontc,# distance from text to points (x,y)
                         ha='center')
            
        
        cen = centroid([logA[best], V[best]], [logA[other], V[other]])
        plt.plot(cen[0],cen[1], 'b^',markersize='10', label='centroid ({:.4f},{:.4f})'.format(np.exp(cen[0]), cen[1]))
        # x,y = cen[0],cen[1]
        # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        if inspect.stack()[1].function != 'Shrink':
            #plt.plot(cen[0],cen[1], 'b^',markersize='10', label='centroid({:.4f},{:.4f})'.format(cen[0], cen[1]))
            x,y = cen[0],cen[1]
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            # plt.annotate('centroid', # this is the text
            #      (x,y), # this is the point to label
            #      textcoords="offset points", # how to position the text
            #      xytext=(0,10), # distance from text to points (x,y)
            #      ha='center',bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3))  
        
        return None
        
    def Reflection(logA, V, F, alpha):
    
        # ranking points based on F value -lower is closer
        
        best     = min(F, key=F.get)
        worst    = max(F, key=F.get)
        other    = 6 - (best+worst)
    
        bes = [logA[best], V[best]]
        wor = [logA[worst], V[worst]] 
        oth = [logA[other], V[other]]
        
        
        fitfunc.update({'best':F[best]})
        fitfunc.update({'worst':F[worst]})
        fitfunc.update({'other':F[other]})
        #step -  I
        
        step_plot(best, worst, other)
        cen = centroid([logA[best], V[best]], [logA[other], V[other]])
    
       
        x,y = cen[0],cen[1]
        # plt.annotate('centroid', # this is the text
        #          (x,y), # this is the point to label
        #          textcoords="offset points", # how to position the text
        #          xytext=(0,10), # distance from text to points (x,y)
        #          ha='center',bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3))
    
        ref = [cen[0] + alpha*(cen[0] - logA[worst]), cen[1] + alpha*(cen[1] - V[worst])]
        plt.plot(ref[0],ref[1], 'g*',markersize='10', label='Reflected ({:.4f},{:.4f})'.format(np.exp(ref[0]), ref[1]))
        
        x,y = ref[0],ref[1]
        # plt.annotate('Reflected', # this is the text
        #          (x,y), # this is the point to label
        #          textcoords="offset points", # how to position the text
        #          xytext=(0,10), # distance from text to points (x,y)
        #          ha='center',bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3))
    
        
    
        plt.plot([ref[0],wor[0]], [ref[1],wor[1]], 'k--')
        
        plt.xlim((min(bes[0],wor[0],oth[0],ref[0])-0.1 , max(bes[0],wor[0],oth[0],ref[0])+0.1))
        plt.ylim((min(bes[1],wor[1],oth[1],ref[1])-0.1 , max(bes[1],wor[1],oth[1],ref[1])+0.1))
        plt.title('reflection')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig('reflected.{}'.format(cowboe_settings['fig extension']),bbox_inches='tight', dpi=300)
        plt.show()
        plt.close()
        
        return cen, ref, best, worst, other, bes, wor, oth
    
    # def Expansion(cen, ref, best, worst, other, gamma):
    
    #     [xr,yr] = ref
    #     [xc, yc] = cen
        
    #     [xe,ye] = [xc + gamma*(xr - xc), yc + gamma*(yr - yc)]
        
    #     step_plot(best, worst, other)
    #     x,y = xe,ye
        
    #     exp = [x,y]
        
    #     plt.plot(xe,ye, 'b*',markersize='10', label = 'expanded ({:.4f},{:.4f})'.format(xe,ye))
        
    #     wor = [logA[worst], V[worst]]
    #     plt.plot([exp[0],wor[0]], [exp[1],wor[1]], 'k--')
    
        
    #     #plt.annotate('Expanded', # this is the text
    #     #         (x,y), # this is the point to label
    #     #        textcoords="offset points", # how to position the text
    #     #        xytext=(0,10), # distance from text to points (x,y)
    #     #        ha='center',bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3))
        
    #     plt.xlim((min(bes[0],wor[0],oth[0],exp[0])-0.1 , max(bes[0],wor[0],oth[0],exp[0])+0.1))
    #     plt.ylim((min(bes[1],wor[1],oth[1],exp[1])-0.1 , max(bes[1],wor[1],oth[1],exp[1])+0.1))
    #     plt.title('expansion')
    #     plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    #     plt.savefig('expanded.{}'.format(cowboe_settings['fig extension']),bbox_inches='tight', dpi=300)
    #     plt.show()
    #     plt.close()
        
    #     return exp
    
    def inner_contraction(cen, wor, best, worst, other, beta):
        
        [xc,yc],[xw,yw] = cen,wor
        in_con = [xc + beta*(xw - xc), yc + beta*(yw - yc)]
        
        [xic, yic] = in_con
        
        step_plot(best, worst, other)
    
        plt.plot(xic,yic, 'k*',markersize='10', label = 'inner_c ({:.4f},{:.4f})'.format(np.exp(xic),yic))
        plt.plot([xic,xw], [yic,yw], 'k--')
        plt.plot([xic,xc], [yic,yc], 'k--')
    
    
        
        # plt.annotate('inner_c', # this is the text
        #          (xic, yic), # this is the point to label
        #          textcoords="offset points", # how to position the text
        #          xytext=(0,10), # distance from text to points (x,y)
        #          ha='center',bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3))
        
        plt.xlim((min(bes[0],wor[0],oth[0],in_con[0])-0.1 , max(bes[0],wor[0],oth[0],in_con[0])+0.1))
        plt.ylim((min(bes[1],wor[1],oth[1],in_con[1])-0.1 , max(bes[1],wor[1],oth[1],in_con[1])+0.1))
        plt.title('inner contraction')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig('inner_c.{}'.format(cowboe_settings['fig extension']),bbox_inches='tight', dpi=300)
        plt.show()
        plt.close()
        
        return in_con
        
    def outer_contraction(ref, cen, wor,  best, worst, other, beta ):
        
        [xr,yr], [xc, yc] = ref, cen
        out_con = [xc + beta*(xr - xc), yc + beta*(yr - yc)]
        
        [xoc, yoc] = out_con
        
        step_plot(best, worst, other)
    
        
        plt.plot(xoc,yoc, 'c*',markersize='10', label='outer_c ({:.4f},{:.4f})'.format(np.exp(xoc),yoc))
        
        plt.plot([out_con[0],wor[0]], [out_con[1],wor[1]], 'k--')
    
        
        # plt.annotate('outer_c', # this is the text
        #          (xoc, yoc), # this is the point to label
        #          textcoords="offset points", # how to position the text
        #          xytext=(0,10), # distance from text to points (x,y)
        #          ha='center',bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3))
        
        plt.xlim((min(bes[0],wor[0],oth[0],out_con[0])-0.1 , max(bes[0],wor[0],oth[0],out_con[0])+0.1))
        plt.ylim((min(bes[1],wor[1],oth[1],out_con[1])-0.1 , max(bes[1],wor[1],oth[1],out_con[1])+0.1))
        plt.title('outer contraction')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig('outer_c.{}'.format(cowboe_settings['fig extension']),bbox_inches='tight', dpi=300)
        plt.show()
        plt.close()
        
        return out_con
    
    def Shrink(wor, oth, best, worst, other, delta):
        [xb,yb] = bes
        [xw,yw] = wor
        [xo,yo] = oth
        
        step_plot(best, worst, other)
        s_wor = [xb + delta *(xw-xb), yb + delta*(yw-yb)]
        s_oth = [xb + delta *(xo-xb), yb + delta*(yo-yb)]
        
        x,y = s_wor[0], s_wor[1]
        plt.plot(x,y, 'm*',markersize='10', label='s_worst ({:.4f},{:.4f})'.format(np.exp(x),y))   
        # plt.annotate('s_worst', # this is the text
        #          (x, y), # this is the point to label
        #          textcoords="offset points", # how to position the text
        #          xytext=(0,10), # distance from text to points (x,y)
        #          ha='center',bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3))
        
        x,y = s_oth[0], s_oth[1]
        plt.plot(x,y, 'k*',markersize='10', label='s_other ({:.4f},{:.4f})'.format(np.exp(x),y))   
        # plt.annotate('s_other', # this is the text
        #          (x, y), # this is the point to label
        #          textcoords="offset points", # how to position the text
        #          xytext=(0,10), # distance from text to points (x,y)
        #          ha='center',bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3))    
        
        plt.xlim((min(bes[0],wor[0],oth[0],s_wor[0], s_oth[0])-0.1 , max(bes[0],wor[0],oth[0],s_wor[0], s_oth[0])+0.1))
        plt.ylim((min(bes[1],wor[1],oth[1],s_wor[1], s_oth[1])-0.1 , max(bes[1],wor[1],oth[1],s_wor[1], s_oth[1])+0.1))
        plt.title('shrinking')       
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig('shrink.{}'.format(cowboe_settings['fig extension']),bbox_inches='tight', dpi=300)
        plt.show()
        plt.close()
        
        return s_wor, s_oth
        
    def convertor(p):
        
        return([np.exp(p[0]), p[1]])
    
    # def A_convertor(ref, exp, in_con, out_con, s_wor, s_oth):
        
    #     c_ref       = convertor(ref)
    #     c_exp       = convertor(exp)
    #     c_in_con    = convertor(in_con)
    #     c_out_con   = convertor(out_con)
    #     c_s_wor     = convertor(s_wor)
    #     c_s_oth     = convertor(s_oth)
        
    #     return {
    #         'reflection':c_ref, 
    #         'expansion':c_exp, 
    #         'in_contract':c_in_con, 
    #         'out_contract':c_out_con, 
    #         'shrink_worst':c_s_wor, 
    #         'shrink_other':c_s_oth
    #         }
    
    def A_convertor(ref, in_con, out_con, s_wor, s_oth):
        
        c_ref       = convertor(ref)
        #c_exp       = convertor(exp)
        c_in_con    = convertor(in_con)
        c_out_con   = convertor(out_con)
        c_s_wor     = convertor(s_wor)
        c_s_oth     = convertor(s_oth)
        
        return {
            'reflection':c_ref, 
            #'expansion':c_exp, 
            'in_contract':c_in_con, 
            'out_contract':c_out_con, 
            'shrink_worst':c_s_wor, 
            'shrink_other':c_s_oth
            }
        
    
    alpha   = cowboe_settings['NM_alpha']
    #gamma   = cowboe_settings['NM_gamma']
    beta    = cowboe_settings['NM_beta']
    delta   = cowboe_settings['NM_delta']
    
    cen, ref, best, worst, other, bes, wor, oth = Reflection(logA, V, F, alpha)
    #exp = Expansion(cen, ref, best, worst, other, gamma)
    in_con = inner_contraction(cen, wor, best, worst, other, beta)
    out_con = outer_contraction(ref, cen, wor, best, worst, other, beta )
    s_wor, s_oth = Shrink(wor, oth, best, worst, other, delta)
    
    #conv = A_convertor(ref, exp, in_con, out_con, s_wor, s_oth)
    conv = A_convertor(ref, in_con, out_con, s_wor, s_oth)
    print('\n')
    print('{}\t\t{}\t{}'.format('Move','A','V'))
    print('==============================')
    for k, v in conv.items():
        conv[k] = [round(i, 4) for i in v]
        print('{}\t{}\t{}'.format(k,conv[k][0],conv[k][1]))
    print('==============================')
    
    return conv

def NMprogress(**kwargs):
    '''
    provies an update on the progress of the algorithm by generating gif for steps
    and summarizies the function values in term of figures and convergence.

    Parameters
    ----------
    progessfile : str, mandatory
                Name of the pogress file which hosts the parameter values in
                3 columns A, V, fit() in groups of 3 (three vertices of the simplex
                at any given step)
        

    Returns
    -------
    None.

    '''
    def area(x,y):
    
        a = math.sqrt((x[0]-x[1])**2 + (y[0]-y[1])**2)
        b = math.sqrt((x[1]-x[2])**2 + (y[1]-y[2])**2)
        c = math.sqrt((x[0]-x[2])**2 + (y[0]-y[2])**2)
        
        # calculate the semi-perimeter
        s = (a + b + c) / 2
        
        # calculate the area
        area = (s*(s-a)*(s-b)*(s-c)) ** 0.5
        
        return round(area,8)
    
    def triangle(A,ind):
        Ax = np.array([np.log(i) for i in A[:,0]])
        Vy = np.array(A[:,1])
        F  = np.array(A[:,2])
        
        
        #plt.plot(Ax, Vy,'k^',markersize='10')
        plt.xlabel('ln A')
        plt.ylabel(r'$\Delta$ V')
        plt.plot([Ax[0],Ax[1]], [Vy[0], Vy[1]], 'k-')
        plt.plot([Ax[1],Ax[2]], [Vy[1], Vy[2]], 'k-')
        plt.plot([Ax[2],Ax[0]], [Vy[2], Vy[0]], 'k-')
    
        a = area(Ax,Vy)
        #stopcheck1, stopcheck2 = cowboestop(fit = F)
        stopcheck1 = cowboestop(fit = F)
    
        
        plt.xlim(min([np.log(i) for i in points[:,:,0].flatten()])-0.15 , max([np.log(i) for i in points[:,:,0].flatten()])+0.15)
        
        plt.ylim(min(points[:,:,1].flatten())-0.15 , max(points[:,:,1].flatten())+0.15)
    
        best = np.argmin(F)
        worst = np.argmax(F)
            
        for x,y,l in zip(Ax,Vy,range(len(Ax))):
            
            if    l == best    : 
                label = '{:.4f}'.format(F[l])
                fontc = 'g'
                plt.plot(Ax[l], Vy[l],'g^',markersize='10')
            elif  l == worst   : 
                label = '{:.4f}'.format(F[l])
                fontc = 'r'
                plt.plot(Ax[l], Vy[l],'r^',markersize='10')
            else               : 
                label = '{:.4f}'.format(F[l])
                fontc = 'y'
                plt.plot(Ax[l], Vy[l],'y^',markersize='10')
        
                
            plt.annotate(label, # this is the text
                         (x,y), # this is the point to label
                         textcoords="offset points", # how to position the text
                         xytext=(0,10), # distance from text to points (x,y)
                         color = fontc,
                         ha='center')
        
        plt.annotate('area = {}'.format(a), # this is the text
                          (max([np.log(i) for i in points[:,:,0].flatten()])-0.35,max(points[:,:,1].flatten())+0.1), # this is the point to label
                          textcoords="offset points", # how to position the text
                          xytext=(0,0), # distance from text to points (x,y)
                          ha='center',bbox=dict(boxstyle='round,pad=0.2', fc='red', alpha=0.3))
        
        plt.annotate('{}'.format(ind+1), # this is the text
                          xy=(1.35,0.95),
                          xytext=(1.35,0.95), 
                          color = 'k',
                          fontsize=18, 
                          ha='center',bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3))
        
        
        
        
        #return a, stopcheck1, stopcheck2
        return a, stopcheck1
    
    def cowboegif(**kwargs):
        '''
        Makes gif file with snapshots of the different steps the NM algorithm takes.
    
        Parameters
        ----------
        FPS : int, mandatory
            frames per second for the gif file to be generated.
    
        Returns
        -------
        None.
    
        '''
       
        FPS = kwargs['FPS']
        
        filenames=[]
        images = []
        
        source = [f for f in os.listdir('.') if os.path.isfile(f) if f.endswith('jpg')]
    
        source = [str(i)+'.jpg' for i in sorted([int(i.split('.')[0]) for i in source])]
        
        for file in source:
            if file.endswith('.jpg'):
                filenames.append(file)
                    
        for filename in filenames:
            images.append(imageio.imread(filename))
        imageio.mimsave('cowboe_NM_steps_FPS_{}.gif'.format(FPS), images, fps=FPS)
        
        return None

    def cowboestop(**kwargs):
        '''
        Provides the stoping criteria check based on this formula,
        
        stopvalue > sqrt( (1/3.0) * ((f1 - favg)**2 + (f1 - favg)**2 + (f1 - favg)**2 ))
        
    
        Parameters
        ----------
        F : array
            An array of all the fitness values in the current simplex i.e. [f1,f2,f3]
    
        Returns
        -------
        stopcheck : float
            Value evaluated from the stop criteria formula  (RMSD) provided.
    
        '''  
        F = kwargs['fit']
        
        # return np.sqrt( (1/3.0) * ((F[0] - np.mean(F))**2 \
        #                            + (F[1] - np.mean(F))**2 \
        #                                + (F[2] - np.mean(F))**2 )), \
        #     2.0*((np.max(F) - np.min(F)) / (np.max(F) + np.min(F) + 0.1))
        
        return np.sqrt( (1/3.0) * ((F[0] - np.mean(F))**2 \
                                   + (F[1] - np.mean(F))**2 \
                                       + (F[2] - np.mean(F))**2 ))
    
    def triangleprop(A):
    
        x = np.array([np.log(i) for i in A[:,0]])
        y = np.array(A[:,1])
        F  = np.array(A[:,2])
        
        best = np.argmin(F)

        def area(x,y):
        
            a = ((x[0]-x[1])**2 + (y[0]-y[1])**2) ** 0.5
            b = ((x[1]-x[2])**2 + (y[1]-y[2])**2) ** 0.5
            c = ((x[0]-x[2])**2 + (y[0]-y[2])**2) ** 0.5
            
            # calculate the semi-perimeter
            s = (a + b + c) / 2.0
            
            # calculate the area
            are = (s*(s-a)*(s-b)*(s-c)) ** 0.5
            
            return round(are,8)
        
        def centroid(x,y):
            
            cx = (x[0]+x[1]+x[2])/3.0
            cy = (y[0]+y[1]+y[2])/3.0
            
            return (round(cx,4), round(cy,4))
        
        def circumcircle(x,y):
            
            a = ((x[0]-x[1])**2 + (y[0]-y[1])**2) ** 0.5
            b = ((x[1]-x[2])**2 + (y[1]-y[2])**2) ** 0.5
            c = ((x[0]-x[2])**2 + (y[0]-y[2])**2) ** 0.5
            
            r = (a*b*c) / ((a+b+c) * (b+c-a) * (c+a-b) * (a+b-c)) ** 0.5
            
            return round(r, 6)
        
        def funcdistance(x,y):
            
            cx = (x[0]+x[1]+x[2])/3.0
            cy = (y[0]+y[1]+y[2])/3.0
            fd = ((x[best] - cx) ** 2 + (y[best] - cy) ** 2) ** 0.5
            
            return round(fd, 6)
        
        def flatness(x,y):
            
            a = ((x[0]-x[1])**2 + (y[0]-y[1])**2) ** 0.5
            b = ((x[1]-x[2])**2 + (y[1]-y[2])**2) ** 0.5
            c = ((x[0]-x[2])**2 + (y[0]-y[2])**2) ** 0.5
            
            w = max(a,b,c)
            ar = area(x,y)
            
            fl = ar/(w**3)
            
            return round(fl,6)
            
                 
        
        are = area(x,y)
        centroid = centroid(x,y)
        r = circumcircle(x,y)
        fd = funcdistance(x,y)
        fl = flatness(x,y)
        
        return are, centroid, r, fd, fl
    
    pre_points = np.loadtxt(kwargs['progressfile'])
    points = pre_points.reshape((int(len(pre_points)/3),3,3))
    
    folname = 'cowboe_NM_steps_{}'.format(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.mkdir(folname)
    os.chdir(folname)
    
    
    areas = []
    stopchecks1 = []
    #stopchecks2 = []                       
    for ind, poi in enumerate(points):
        plt.cla()
        #ar, stopcheck1,  stopcheck2 = triangle(poi, ind)
        ar, stopcheck1 = triangle(poi, ind)
        areas.append(ar)
        stopchecks1.append(stopcheck1)
        #stopchecks2.append(stopcheck2)
        
        plt.savefig('{}.{}'.format(ind+1,cowboe_settings['fig extension']), dpi=300)
        plt.show()
        plt.close()
        
        a,c,r,f,fl = triangleprop(poi)
        
        print(f'\n\nThe area of the simplex is:\t\t{a}')
        print(f'The flatness of the simplex is:\t\t{fl}')
        print(f'The centroid of the simplex is:\t\t{c}')
        print(f'The circum circle radius of the simplex is:\t\t{r}')
        print(f'The distance between centroid and the best point of the simplex is:\t\t{f}\n\n')
    
    plt.plot(areas, 'r^--')
    plt.ylim((0,np.max(areas)*1.10))
    plt.xlabel('simplexes',weight='bold')
    plt.ylabel('area',weight='bold')
    plt.title('Area of simplexes')
    plt.savefig('Area of simplexes.{}'.format('pdf'),bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()
    
    plt.plot(stopchecks1, 'bs--')
    plt.ylim((0,np.max(stopchecks1)*1.10))
    plt.axhline(y=np.min(stopchecks1)+0.1, c='g', ls='-.')
    plt.xlabel('simplexes',weight='bold')
    plt.ylabel('RMSD - stopping criteria',weight='bold')
    plt.title('RMSD - fit()')
    plt.savefig('RMSF-fit().{}'.format('pdf'),bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()
    
    # plt.plot(stopchecks2, 'k*--')
    # plt.ylim((0,np.max(stopchecks2)*1.10))
    # plt.axhline(y=np.min(stopchecks2)+0.1, c='g', ls='-.')
    # plt.show()
    # plt.close()
    print('\nConstructing GIF image of the NM steps ...')
    cowboegif(FPS = 1)
    print('\nDone ...!')
    os.chdir('..')
    
    return None

def cowboe3Dsurface(**kwargs):
    '''
    Constructs a 3D surface plot based on the evaluvated unique parameter combinations
    and saves it in mp4 format. Need to have ffmpeg installed in path to use this function.

    Parameters
    ----------
    progessfile : str, mandatory
                Name of the pogress file which hosts the parameter values in
                3 columns A, V, fit() in groups of 3 (three vertices of the simplex
                at any given step)
    
    fps         : int, optional
                Frames per second for the animation.
                Default value is 15
                
    dpi         : int, optional
                Dots per inch value for the animation.
                Default value is 300
    
    name        : str, optional
                file name for the animation.
                Default value is 'cowboe3Dsurface.mp4'
        

    Returns
    -------
    None.

    '''


    pre_points = np.loadtxt(kwargs.get('progressfile'))
    fps = kwargs.get('fps', 15)
    dpi = kwargs.get('dpi', 300)
    name = kwargs.get('name', 'cowboe3Dsurface.mp4')
    
    # Set up formatting for the movie files
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=fps, metadata=dict(artist='Me'), bitrate=1800)
    points = pre_points.reshape((int(len(pre_points)/3),3,3))

    x = np.array([np.log(i) for i in points[:,:,0].flatten()])
    y = points[:,:,1].flatten()
    z = points[:,:,2].flatten()
    
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    
    tri = ax.plot_trisurf(x, y, z, linewidth=0.2, antialiased=False, cmap=cm.coolwarm, edgecolor='none')
    fig.colorbar(tri)
    ax.set_zlim(min(z)-0.25, max(z)+0.25);
    
    def rotate(angle):
        ax.view_init(azim=angle)
    
    print('\n constructing 3D surface plot for the points and fit() values')
    rot_animation = animation.FuncAnimation(fig, rotate, frames=np.arange(0,360,1), interval=50)
    rot_animation.save(name,  writer=writer, dpi=dpi)
        
    return None

def progressfile(**kwargs):
    '''
    To create the progress file from an n*3*3 array. Gives the progress.txt file.

    Parameters
    ----------
    points : numpy array in the shape of n*3*3
        An array of list of all the simplexes obtained.

    Returns
    -------
    

    '''
    points=kwargs['points']
    with open('progress.txt','w') as progfile:
        
        for p in points:
            progfile.write('\n# New row/simplex/step\n')
            for i in p:
                progfile.write(str(i).strip('[]'))
                progfile.write('\n')
    return None

def cowboeKS(**kwargs):
    '''
    Computes the Kolmogorov-Smirnov statistic on 2 samples.
    The null hypothesis is that
    the two individual samples were extracted from the same distribution
    and if the p-value is large or the KS statistics is small, then we 
    cannot reject the hypothesis that the distributions of the two samples are the same.

    Parameters
    ----------
    location : str,
    
        location of all the trajectory files for a given point.
    
    listfile : str,
    
        name of the file containing the list of all the trajectory files
        in the ascending order of windows. Can be same as the metadata file
        used for wham calculation or just a list of each window's trajectory
        file's names.
        
    
    percentage : float or int,
    
        percentage of the total sampling to perform the test on
        eg. if 80 is givenm window n's total data will be compared to 80%
        of its data to test the null hypothesis. 
        

    Returns
    -------
    None.

    '''
    def distribution(w, data1, data2, loc):
    
        sns.distplot(sorted(data1),hist=False)
        sns.distplot(sorted(data2),hist=False)
        plt.xlabel(cowboe_settings["reaction coordinate unit"])
        plt.ylabel('pdf')      
        plt.title('window - {}'.format(w))
        plt.savefig(os.path.join(os.sep,loc,'dist_{}.{}'.format(w,cowboe_settings['fig extension'])),bbox_inches='tight', dpi=300)
        plt.show()
    
    loc = kwargs['location']
    per = kwargs['percentage']
    listfile = kwargs['listfile']
    
    wins = len(glob.glob1(loc,"*.traj"))
    print('\nFound {} individual window\'s trajectories in the folder'.format(wins))
    
    with open(os.path.join(os.sep,loc,listfile)) as lfile:
        lines = lfile.readlines()
        l=[]
        for line in lines:
            words = line.split()
            l.append(words[0])
        
        pvalues = []
        KSstats = []
        for w,trajfile in enumerate(l):
            data = np.loadtxt(os.path.join(os.sep,loc,trajfile))[:,1]
            datacut = data[:int(round(len(data)*per/100.0))]
            l1 = len(data)
            l2 = len(datacut)
            D = cowboe_settings["KS coefficent D"]*np.sqrt((l1+l2)/(l1*l2))
            KSstat, pvalue = ks_2samp(data, datacut)
            pvalues.append(pvalue)
            KSstats.append(KSstat)
            print('\n Window {} :\n\tp:\t{}\n\tKS:\t{}\n'.format(w,pvalue, KSstat))
            distribution(w, data,datacut, loc)
            
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.bar(range(len(l)),pvalues)
    plt.axhline(y = 0.05, ls='--',c='g')
    plt.title('p-values')
    plt.xlabel('windows')
    plt.xticks(range(len(l)))
    plt.savefig(os.path.join(os.sep,loc,'pvalues.{}'.format(cowboe_settings['fig extension'])),bbox_inches='tight', dpi=300)
    plt.show()
    
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.bar(range(len(l)),KSstats)
    ax.axhline(y=D,ls='--',c='r')
    plt.title('KS-statistics')
    plt.xlabel('windows')
    plt.xticks(range(len(l)))
    plt.savefig(os.path.join(os.sep,loc,'KS-statistics.{}'.format(cowboe_settings['fig extension'])),bbox_inches='tight', dpi=300)
    plt.show()
    
    return None

def cowboe_OVL (**kwargs):
    '''
    Calculates the coefficient of overlap (OVL) for different window's
    distribution

    Parameters
    ----------
    location : str,
    
        location of all the trajectory files for a given point.
    
    listfile : str,
    
        Name of the file containing the list of all the trajectory files
        in the ascending order of windows. Can be same as the metadata file
        used for wham calculation or just a list of each window's trajectory
        file's names.
    
    name    : str,
        
        Name for the Overlap calculation (generally name of the point).
    
    distplot : bool,
        
        switches the distribution plot with overlap on or off
        
        

    Returns
    -------
    None
        

    '''
    
    def overlap(hist1,hist2,i,j):

        def cdf(s, x):
            "Cumulative distribution function.  P(X <= x)"
    
            return 0.5 * (1.0 + erf((x - np.mean(s)) / (np.std(s) * sqrt(2.0))))
        
        x, y = hist1,hist2
        m1, m2 = np.mean(x), np.mean(y)
        s1, s2 = np.std(x), np.std(y)
        v1, v2 = np.var(x), np.var(y)  
        if (s2, m2) < (s1, m1):
            x, y = y, x
        dv = v2 - v1
        dm = fabs(m2- m1)
        a = m1 * v2 - m2 * v1
        b = s1 * s2 * sqrt(dm**2.0 + dv * log(v2 / v1))
        x1 = (a + b) / dv
        x2 = (a - b) / dv

        if distplot:
            fig, ax = plt.subplots()
            sns.distplot(hist1,label='win %d'%i,color='r')
            sns.distplot(hist2,label='win %d'%j,color='b')
            plt.legend()
            plt.savefig(os.path.join(os.sep,loc,'windows-{}&{}.{}'.format(i,j,cowboe_settings['fig extension'])), bbox_inches='tight', dpi=300)
            plt.show()
        output = 1.0 - (fabs(cdf(y,x1) - cdf(x,x1)) + fabs(cdf(y,x2) - cdf(x,x2)))
        
        return round(output,4)
    
    loc = kwargs['location']
    name = kwargs['name']
    listfile = kwargs['listfile']
    distplot = kwargs['distplot']
    
    wins = len(glob.glob1(loc,"*.traj"))
    print('\nFound {} individual window\'s trajectories in the folder'.format(wins))
    
    with open(os.path.join(os.sep,loc,listfile)) as lfile:
        lines = lfile.readlines()
        l=[]
        
    for line in lines:
        words = line.split()
        l.append(words[0])
    
    OVL = np.zeros((wins,wins))
    
    for w,trajfile in enumerate(l):
        d1 = np.loadtxt(os.path.join(os.sep,loc,trajfile))[:,1]
        print('\n')
        
        for jo in range(w,wins):
            if w == jo:
                OVL[w][jo]=1.0
                continue

            d2 = np.loadtxt(os.path.join(os.sep,loc,l[jo]))[:,1]


            hist1=np.array(sorted(d1))    
            hist2=np.array(sorted(d2))

            output = overlap(hist1,hist2,w,jo)      
            
            OVL[jo][w] = output
            OVL[w][jo] = output
            if output == 0.00000: 
                break
                
            
            print('\n\tOverlap coefficient between Window %d and %d is: %f'%(w,jo,output))
            #print('Overlap coefficient between Window %d and %d is: %f'%(jo,i,output2))
    
    
    np.savetxt('OVL-%s.txt'%name,OVL,fmt='%.6f')
    
    OVL2=np.loadtxt('OVL-%s.txt'%name)
    plt.matshow(OVL2, cmap='plasma', interpolation='nearest')
    #plt.imshow(OVL2, cmap='plasma', interpolation='nearest')
    plt.title('OVL - %s\n'%name)
    #plt.xlim(0,wins-0.5)
    #plt.ylim(0,wins-0.5)
    plt.colorbar()
    plt.clim(0,1)
    #plt.xticks(range(wins), rotation='vertical')
    #plt.yticks(range(wins), rotation='horizontal')
    plt.savefig(os.path.join(os.sep,loc,'OVL-{}.{}'.format(name,cowboe_settings['fig extension'])),bbox_inches='tight', dpi=300)
    plt.show()      

    return None     

def cowboe_trajcut(**kwargs):
    '''
    Slices the trajectories and creates new trajectory files with given percentage
    of the total sampling.

    Parameters
    ----------
    
    start   : int,
        index value of the starting point from where 
        the percentage of the data will be extracted. to start from the
        beginnning just provide "0" (without the quotes).
    
    percentage : float,
        Percentage of the sampling (points) to use from total sampling.
    
    location : str,
        location of all the trajectory files for a given point.
    
    listfile : str,

        Name of the file containing the list of all the trajectory files
        in the ascending order of windows. Can be same as the metadata file
        used for wham calculation or just a list of each window's trajectory
        file's names (with extension).
    
    name :  str,
        name of the point.

    Returns
    -------
    None.

    '''
    
    per = kwargs['percentage']
    loc = kwargs['location']
    listfile = kwargs['listfile']
    name = kwargs['name']
    start = kwargs['start']
    

    slfol = os.path.join(os.sep,loc,'{}_{}_percentage'.format(name,per))
   
    if os.path.isdir(slfol):pass
    else:os.mkdir(slfol)
    
    trajfiles = glob.glob1(loc,"*.traj")

    for i, trajfile in enumerate(trajfiles):
        data=np.genfromtxt(os.path.join(os.sep,loc,trajfile))
        stop = start+int(round(len(data)*per/100.0))
        datanew=data[start:stop]
        newfile=os.path.join(os.sep,slfol,trajfile)
        np.savetxt(newfile,datanew)
    shutil.copy(os.path.join(os.sep,loc,listfile),slfol)
    
    return None   

def cowboe_pmfplot(**kwargs):
    '''
    Plots the pmf curve of a given free energy file.

    Parameters
    ----------
    pmf : str,
        name of the curve 1 (PMF) file.
    name : str,
        name to save the output with.

    Returns
    -------
    None
    
    '''
    
    free1  = kwargs['pmf']
    pdfname = kwargs['name']
    splice = kwargs['splice']
    
    c1 = 'PMF-I'
        
    free1  = np.loadtxt(free1)[splice:]
    f1, e1 = free1[:,0:2],free1[:,2]
    if cowboe_settings['error bar'] : 
        plt.errorbar(f1[::,0], f1[::,1],yerr=e1,lw=1.5,capsize=2,errorevery=cowboe_settings['error every'],elinewidth=1.5,label=c1)
    else:
        plt.plot(f1[::,0], f1[::,1],lw=1.5,label=c1)
    plt.title('PMF')
    plt.xlabel(cowboe_settings["reaction coordinate unit"],fontsize=14,weight='bold')
    plt.ylabel(cowboe_settings["PMF unit"],fontsize=14,weight='bold')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig('{}.{}'.format(pdfname, cowboe_settings['fig extension']), bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()    

def settings_update():
    '''
    Prints instruction for updating default settings for the cowboe module and
    the wham calculation.

    Returns
    -------
    None.

    '''

    print(f"\nTo make temporary runtime updates to variables cowboe_settings and wham_settings use,\n\
          Dict update e.g. cowboe_settings.update({{'param B' : 2.0001}})\n\n\
              The defaults values are,\n\n\tcowboe_settings = \n\n{cowboe_settings}\n\n\twham_settings=\n\n{wham_settings}")
              
    print(f"\nTo make permanent changes to the defaults values,\nedit the respective dict in the module initiation below \" if __name__ == '__main__' \":, in the cowboe.py file at the following location:\n\t'{os.path.dirname(inspect.getfile(cowboe))}'")

    return None

if __name__ == '__main__':
    
    cowboe_settings = {
    "PMF unit"                      : r'PMF - Kcal / (mol $\cdot~\AA^2$)',
    "reaction coordinate unit"      : r"$\AA$",     
    "polynomial fit order"          : 12,  
    "param B"                       : 2.0,         
    "Number of datapoints"          : 10**5,        
    "conventional force constant"   : 7,            
    "conventional window width"     : 0.5,
    "conventional no of windows"    : 24,
    "equal_sampling"                : True,
    "conv. min of 1st window"       : 2.5,
    "conv. min of last window"      : 14.5,
    "fill colour"                   : 'r',
    "NM_alpha"                      : 1,
    "NM_gamma"                      : 2,
    "NM_beta"                       : 0.5,
    "NM_delta"                      : 0.5,
    "error every"                   : 3,
    "error bar"                     : False,
    "mark every"                    : 3,
    "fig extension"                 : 'jpg',
    "KS coefficent D"               : 1.36
    }
    
    wham_settings = {
    "metadatafile"        : 'list.txt',
    "hist_min"            : 2.0,
    "hist_max"            : 14.5,
    "num_bins"            : 100,
    "tol"                 : 0.0001,
    "temp"                : 300.0,
    "numpad"              : 0,
    "randSeed"            : random.randint(9999,10000000)
    }
       
else : 

    cowboe_settings = {
    "PMF unit"                      : r'PMF - Kcal / (mol $\cdot~\AA^2$)',
    "reaction coordinate unit"      : r"$\AA$",     
    "polynomial fit order"          : 12,  
    "param B"                       : 2.0,         
    "Number of datapoints"          : 10**5,        
    "conventional force constant"   : 7,            
    "conventional window width"     : 0.5,
    "conventional no of windows"    : 24,
    "equal_sampling"                : True,
    "conv. min of 1st window"       : 2.5,
    "conv. min of last window"      : 14.5,
    "fill colour"                   : 'r',
    "NM_alpha"                      : 1,
    "NM_gamma"                      : 2,
    "NM_beta"                       : 0.5,
    "NM_delta"                      : 0.5,
    "error every"                   : 3,
    "error bar"                     : False,
    "mark every"                    : 3,
    "fig extension"                 : 'jpg',
    "KS coefficent D"               : 1.36
    }
    
    wham_settings = {
    "metadatafile"        : 'list.txt',
    "hist_min"            : 2.0, 
    "hist_max"            : 14.5,
    "num_bins"            : 100,
    "tol"                 : 0.0001,
    "temp"                : 300.0,
    "numpad"              : 0,
    "randSeed"            : random.randint(9999,10000000)
    }

    
