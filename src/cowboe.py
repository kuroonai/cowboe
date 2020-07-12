#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Module for optimization and selection of parameters for umbrella sampling

COWBOE - "Construction Of Windows Based On Energy"

Created on Mon Jan 20 15:19:02 2020

@author:Naveen Kumar Vasudevan, 
        400107764,
        Doctoral Student, 
        The Xi Research Group, 
        Department of Chemical Engineering,
        McMaster University, 
        Hamilton, 
        Canada.
        
        naveenovan@gmail.com
        https://naveenovan.wixsite.com/kuroonai
"""

import numpy as np
from matplotlib import pylab as plt
import os
import pickle
from scipy.interpolate import interp1d as inp
from scipy.signal import argrelextrema as extrema
import pandas as pd
import sys
import math
import time
import inspect
from shapely.geometry import Polygon
import pickle as p
import json as j

A = [2.0, 2.9, 3.5]
V = [0.75, 0.8700, 0.8000]
fit = [21.2642, 17.6376, 24.5880]

global cowboe_settings

cowboe_settings = {
    "PMF unit"                      : r'PMF - Kcal / (mol $\cdot~\AA^2$)',
    "reaction coordinate unit"      : r"$\AA$",
    "polynomial fit order"          : 12, #adjust order to get smooth fit for the curves
    "Number of datapoints"          : 10**5,
    "conventional force constant"   : 7,
    "conventional window width"     : 0.5,
    "conventional no of windows"    : 24,
    "conv. min of 1st window"       : 2.5,
    "conv. min of last window"      : 14.5,
    "fill colour"                   : 'r',
    "NM_alpha"                      : 1,
    "NM_gamma"                      : 2,
    "NM_beta"                       : 0.5,
    "NM_delta"                      : 0.5
    }

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
    plt.savefig('PMF-actual+polyfit.pdf',bbox_inches = 'tight')
    plt.show()
    
    
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
    plt.savefig('grdient-actual+polyfit.pdf',bbox_inches = 'tight')
    plt.show()
    
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
        
    for exr in x[trough]:
        plt.axvline(exr,ls='--',c='r')    
    for exr in x[crest]:
        plt.axvline(exr,ls='--',c='g')   
    plt.show()
    
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
        
    B = float, mandatory
        parameter 2 for cowboe.
        
    V = float, mandatory
        Optimization parameter for NM algorithm which controls energy 
        barrier.
    
    sc = int, mandatory
        Sampling considered in nano seconds e.g. 8 ns

    Returns
    -------
    None.

    '''

    iniloc = os.getcwd()
    
    A                   = kwargs['A']
    B                   = kwargs['B']
    V                   = kwargs['V']
    samplingconsidered  = kwargs['sc']
    
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

        plt.savefig('up and down_%.2f_%.2f.pdf' % (A, B), bbox_inches='tight')
        plt.title('A = %.4f & B = %.4f' %(A,B))
        plt.show()


    
    pointname = input('\nEnter name of the point (e.g. 1):\t')
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
    plt.title('A = %.2f & B = %.2f' %(A,B))
    plt.ylabel(r'$\Delta$ PMF')
    plt.xlabel(cowboe_settings['reaction coordinate unit'])
    plt.show()
    
    Windows = windows.copy()
    startw = cowboe_settings["conv. min of last window"]
    endw = cowboe_settings["conv. min of 1st window"]


    Windows[0], Windows[-1]= startw, endw

    Rpos = np.array(Windows) #np.array(np.flip(windows))
    Windowwidth = np.diff(Rpos)

    total = cowboe_settings['conventional no of windows'] * samplingconsidered  # ns
    # total = 24*4000000*2 #24 windows - 5000000 2fs steps 10 ns
    fracsamplingtime = [i/sum(Windowwidth) for i in Windowwidth]
    #Samplingtime = [int((i*total)/(sum(Windowwidth)*400000)) for i in Windowwidth]
    Samplingtime = [(j*total) for j in fracsamplingtime]

    plt.plot(Samplingtime[::-1], 'r^--')
    plt.xlim(-0.25, len(windows)-1.25)
    plt.ylabel('ns', fontsize=14, weight='bold')
    plt.xlabel('Windows', fontsize=14, weight='bold')
    plt.title('A = %.2f & B = %.2f' %(A,B))
    plt.show()

    # check loop - window width comparison

    actualind = [close(x, i)[0] for i in windows]

    def boundary(indarray):
        bounds = []
        for ext_ind, ext in enumerate(indarray[:-1]):
            newpair = np.arange(indarray[ext_ind], indarray[ext_ind+1]+1)
            bounds.append(np.array(newpair))

        return np.array(bounds)

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
    plt.savefig('alligned_%.2f_%.2f.pdf' % (A, B), bbox_inches='tight')
    plt.title('A = %.2f & B = %.2f' %(A,B))
    plt.show()


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
    plt.savefig('difference_%.2f_%.2f.pdf' % (A, B),bbox_inches = 'tight')
    plt.show()
    

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


def Kcalc(windows, A, B, V, kgiven):
    '''
    
    V = 0.5 * K * (X - X0)**2
    K = 2*V/(X-X0)**2
    
    '''
    Windows = windows.copy()
    startw = cowboe_settings["conv. min of last window"]
    endw = cowboe_settings["conv. min of 1st window"]

    
    Windows[0], Windows[-1]= startw, endw
    
    
    
    V_x = np.linspace(-0.5,0.5,100)
    test_V = [ 0.5*kgiven*(0-i)**2 for i in V_x]
    plt.plot(V_x, test_V)
    plt.axvline(0.0, linestyle='-.', c='k')
    plt.axvline(-0.5, linestyle='--', c='r')
    plt.axvline(0.5, linestyle='--', c='r')
    plt.axhline(test_V[0], linestyle=':', c='g')
    plt.ylabel(r'$\Delta$ V')
    plt.xlabel(cowboe_settings["reaction coordinate unit"])

    plt.savefig('nativepotential.pdf',bbox_inches = 'tight')
    plt.show()
    
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

    plt.savefig('native_window_potential_%.2f_%.2f.pdf' % (A, B),bbox_inches = 'tight')
    plt.show()
    
    Ms =[]
    k = kgiven
    ww = cowboe_settings["conventional window width"]
    for M in np.arange(cowboe_settings["conv. min of 1st window"],cowboe_settings["conv. min of last window"],ww):
        Ms.append(windowsplot(k,M-ww,M+ww))
    
    
    plt.axhline(0.5*k*ww**2, linestyle='--', c='r')   
    plt.xticks(Ms, rotation=90)
    plt.ylabel(r'$\Delta$ V')
    plt.xlabel(cowboe_settings["reaction coordinate unit"])

    plt.savefig('native_window_potential_all_%.2f_%.2f.pdf' % (A, B),bbox_inches = 'tight')
    plt.show()
    
    
    fig, (ax1,ax2) = plt.subplots(2, sharex=True, figsize=(10,5))
    ax1.bar(np.arange(len(np.diff(Windows[::-1]))), np.diff(Windows[::-1]), color='r')
    ax2.bar(np.arange(len(np.diff(Windows[::-1]))), K, color='g')
    ax1.set(ylabel='Width')
    ax2.set(ylabel='K')
    plt.xticks(np.arange(len(np.diff(Windows[::-1]))))
    plt.xlabel('Window')
    plt.savefig('new_window_potential_%.2f_%.2f.pdf' % (A, B),bbox_inches = 'tight')
    plt.show()
    
    np.savetxt('K-{}-{}.dat'.format(A,B), np.c_[range(len(K)), K])
    
    return K, Windows, Mss


def removeinf_and_gradient(freefile):
    '''
    

    Parameters
    ----------
    freefile : str, mandatory
        removes inf entries in the pmf file and computes the gradient of the PMF.

    Returns
    -------
    None.

    '''

    f = np.loadtxt(freefile)
    location, d = f[:,0], f[:,1]
    
    for check in range(len(d)):

           if np.isinf(d[check])==True:
               continue
           else:
               splice=check
               break
        
    m=np.gradient(d[splice::], location[splice+1] - location[splice]) 
    slopetime=location[100-len(m):]
    gradient = np.c_[slopetime,m]
    np.savetxt('gradient-{}'.format(freefile),gradient,fmt='%.4f')
   
def pmfarea(gfile, gfile2, savefile) :
    '''
    Finds the absolute area difference or fitness between two curves

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
    
    
    x_y_curve1 =  curve1[:,0:2] #these are your points for curve 1 
    x_y_curve2 =  curve2[:,0:2] #these are your points for curve 2 
    
    cleancurve1 = x_y_curve1.copy()
    cleancurve2 = x_y_curve2.copy()
    
    def cleanarea(cleanc1,cleanc2,area_loc):
        polygon_points = [] #creates a empty list where we will append the points to create the polygon
        
        cc1 = cleanc1.copy()
        cc2 = cleanc2.copy()
        
        #to get the actual area diff comment the below for loop block
        if area_loc == 'above' or area_loc == 'below':
            entry=0
            for p1, p2 in zip(cc1, cc2):
                
                 if area_loc == 'below':
                     #to get area above perfect curve
                     if p1[1]<=p2[1]:
                         cc1[entry]=cc2[entry]
                
                 elif area_loc == 'above':
                      #to get area below perfect curve     
                     if p1[1]>=p2[1]:
                         cc1[entry]=cc2[entry]
                
                 entry+=1
        else:pass
        
        for xyvalue in cc1:
            polygon_points.append([xyvalue[0],xyvalue[1]])
        
        for xyvalue in cc2[::-1]:
            polygon_points.append([xyvalue[0],xyvalue[1]])
            
        for xyvalue in cc1[0:1]:
            polygon_points.append([xyvalue[0],xyvalue[1]])
        polygon = Polygon(polygon_points)
        
        return polygon.area
    
    areaa, areab = cleanarea(cleancurve1, cleancurve2, 'above'), cleanarea(cleancurve1, cleancurve2, 'below')
    area = areaa + areab
    rc= curve1[-1,0]-curve1[0,0] #14.437500  - 3.062500 #
    deltapmf=area*(curve1[1,0]-curve1[0,0])/rc#(area[count]*rc[count]*(curve1[1,0]-curve1[0,0]))/rc[count]#(area[count])*(curve1[1,0]-curve1[0,0])#(area[count]/rc[count])*(curve1[1,0]-curve1[0,0])
    newarea= area/rc
    
    c1 = 'benchmark'
    c2 = 'test'
    
    print("\nFor the %s curve:\n\tThe area between the curves is:\t%f between (%.4f and %.4f)  \n\t\tAbove : %f\n\t\tBelow : %f\n\tThe delta pmf value is:\t%f\n\tThe dist-norm area is:\t%f"%(c2, area, curve1[0,0], curve1[-1,0], areaa, areab, deltapmf, newarea))
    plt.plot(x_y_curve1[:,0],x_y_curve1[:,1],c='b',lw=1,marker='+',markevery=5,markersize=5,label="%s"%c1)
    plt.plot(x_y_curve2[:,0],x_y_curve2[:,1],c='k',lw=1,marker='^',markevery=5,markersize=5,label="%s"%c2)      
    plt.fill_between(x_y_curve1[:,0], x_y_curve1[:,1],x_y_curve2[:,1],color=cowboe_settings['fill colour'],label='area-diff = {:.4f}'.format(area))
    
    plt.title(r'%s-%s - $\xi$ vs PMF'%(c1,c2))
    plt.xlabel(cowboe_settings["reaction coordinate unit"],fontsize=14,weight='bold')
    plt.ylabel(cowboe_settings["PMF unit"],fontsize=14,weight='bold')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.annotate('fit() = {:.4f}'.format(area), # this is the text
                       (max(x_y_curve2[:,0])-2,min(x_y_curve2[:,1])+2), # this is the point to label
                       textcoords="offset points", # how to position the text
                       xytext=(0,0), # distance from text to points (x,y)
                       ha='center',bbox=dict(boxstyle='round,pad=0.2', fc='red', alpha=0.3))
    
    plt.savefig(savefile+'.pdf', bbox_inches='tight')
    plt.show()
    plt.close()
    
    output = {
        'area':area,
        'above':areaa,
        'below':areab,
        'rc':rc,
        'deltapmf':deltapmf,
        'norm_area':newarea,
        'deltapmf_above':areaa*(curve1[1,0]-curve1[0,0])/rc,
        'deltapmf_below':areab*(curve1[1,0]-curve1[0,0])/rc,
        'rc_range':[curve1[0,0], curve1[-1,0]]
        }
    pickle_dump(output,savefile)
    json_dump(output,savefile)
    return output


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
        p.dump(d,f)
        
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
    testfile = kwargs['test']
    benchfile = kwargs['bench']
    name = testfile.split('.')[0]
    
    removeinf_and_gradient(testfile)
    pmfarea(benchfile, testfile, 'area_pmf_{}'.format(name))

    
    return None


'''
Simplex Nelder-Mead Optimization (Amoeba Search)
Pseudocode


initialize search area (simplex) using random starting parameter values

while not done loop
    compute centroid
    compute reflected
    
    if reflected is better than best solution then
        replace worst solution with reflected
        
    else if reflected is worse than all but worst then
        compute outward contracted 
        
        if outward contracted is better than reflected
            replace worst solution with outward contracted
        end if 
        
        else
            shrink the search area
        
    else if reflected is worse than all 
        compute inward contracted
        
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


def cowboeNM(**kwargs) :
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
                label = 'best({:.4f},{:.4f})'.format(np.exp(x),y)
                fontc = 'g'
            elif l+1 ==worst : 
                label = 'worst({:.4f},{:.4f})'.format(np.exp(x),y)
                fontc = 'r'
    
            else : 
                label = 'other({:.4f},{:.4f})'.format(np.exp(x),y)
                fontc = 'grey'
        
            plt.annotate(label, # this is the text
                         (x,y), # this is the point to label
                         textcoords="offset points", # how to position the text
                         xytext=(0,5),
                         color = fontc,# distance from text to points (x,y)
                         ha='center')
            
        
        cen = centroid([logA[best], V[best]], [logA[other], V[other]])
        plt.plot(cen[0],cen[1], 'b^',markersize='10')
        x,y = cen[0],cen[1]
        if inspect.stack()[1].function != 'Shrink':
            plt.annotate('centroid', # this is the text
                 (x,y), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center',bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3))  
        
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
        plt.annotate('centroid', # this is the text
                 (x,y), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center',bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3))
    
        ref = [cen[0] + alpha*(cen[0] - logA[worst]), cen[1] + alpha*(cen[1] - V[worst])]
        plt.plot(ref[0],ref[1], 'g*',markersize='10')
        
        x,y = ref[0],ref[1]
        plt.annotate('Reflected', # this is the text
                 (x,y), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center',bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3))
    
        
    
        plt.plot([ref[0],wor[0]], [ref[1],wor[1]], 'k--')
        
        plt.xlim((min(bes[0],wor[0],oth[0],ref[0])-0.1 , max(bes[0],wor[0],oth[0],ref[0])+0.1))
        plt.ylim((min(bes[1],wor[1],oth[1],ref[1])-0.1 , max(bes[1],wor[1],oth[1],ref[1])+0.1))
    
        plt.savefig('reflected.pdf',bbox_inches='tight')
        plt.show()
        
        return cen, ref, best, worst, other, bes, wor, oth
    
    # def Expansion(cen, ref, best, worst, other, gamma):
    
    #     [xr,yr] = ref
    #     [xc, yc] = cen
        
    #     [xe,ye] = [xc + gamma*(xr - xc), yc + gamma*(yr - yc)]
        
    #     step_plot(best, worst, other)
    #     x,y = xe,ye
        
    #     exp = [x,y]
        
    #     plt.plot(xe,ye, 'g*',markersize='10')
        
    #     wor = [logA[worst], V[worst]]
    #     plt.plot([exp[0],wor[0]], [exp[1],wor[1]], 'k--')
    
        
    #     plt.annotate('Expanded', # this is the text
    #              (x,y), # this is the point to label
    #              textcoords="offset points", # how to position the text
    #              xytext=(0,10), # distance from text to points (x,y)
    #              ha='center',bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3))
        
    #     plt.xlim((min(bes[0],wor[0],oth[0],exp[0])-0.1 , max(bes[0],wor[0],oth[0],exp[0])+0.1))
    #     plt.ylim((min(bes[1],wor[1],oth[1],exp[1])-0.1 , max(bes[1],wor[1],oth[1],exp[1])+0.1))
    
    #     plt.savefig('expanded.pdf',bbox_inches='tight')
    #     plt.show()
        
    #     return exp
    
    def inner_contraction(cen, wor, best, worst, other, beta):
        
        [xc,yc],[xw,yw] = cen,wor
        in_con = [xc + beta*(xw - xc), yc + beta*(yw - yc)]
        
        [xic, yic] = in_con
        
        step_plot(best, worst, other)
    
        plt.plot(xic,yic, 'g*',markersize='10')
        
        plt.plot([xic,xw], [yic,yw], 'k--')
        plt.plot([xic,xc], [yic,yc], 'k--')
    
    
        
        plt.annotate('inner_c', # this is the text
                 (xic, yic), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center',bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3))
        
        plt.xlim((min(bes[0],wor[0],oth[0],in_con[0])-0.1 , max(bes[0],wor[0],oth[0],in_con[0])+0.1))
        plt.ylim((min(bes[1],wor[1],oth[1],in_con[1])-0.1 , max(bes[1],wor[1],oth[1],in_con[1])+0.1))
    
        plt.savefig('inner_c.pdf',bbox_inches='tight')
        plt.show()
        
        return in_con
        
    def outer_contraction(ref, cen, wor,  best, worst, other, beta ):
        
        [xr,yr], [xc, yc] = ref, cen
        out_con = [xc + beta*(xr - xc), yc + beta*(yr - yc)]
        
        [xoc, yoc] = out_con
        
        step_plot(best, worst, other)
    
        
        plt.plot(xoc,yoc, 'g*',markersize='10')
        
        plt.plot([out_con[0],wor[0]], [out_con[1],wor[1]], 'k--')
    
        
        plt.annotate('outer_c', # this is the text
                 (xoc, yoc), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center',bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3))
        
        plt.xlim((min(bes[0],wor[0],oth[0],out_con[0])-0.1 , max(bes[0],wor[0],oth[0],out_con[0])+0.1))
        plt.ylim((min(bes[1],wor[1],oth[1],out_con[1])-0.1 , max(bes[1],wor[1],oth[1],out_con[1])+0.1))
                 
        plt.savefig('outer_c.pdf',bbox_inches='tight')
        plt.show()
        
        return out_con
    
    def Shrink(wor, oth, best, worst, other, delta):
        [xb,yb] = bes
        [xw,yw] = wor
        [xo,yo] = oth
        
        step_plot(best, worst, other)
        s_wor = [xb + delta *(xw-xb), yb + delta*(yw-yb)]
        s_oth = [xb + delta *(xo-xb), yb + delta*(yo-yb)]
        
        x,y = s_wor[0], s_wor[1]
        plt.plot(x,y, 'g*',markersize='10')   
        plt.annotate('s_worst', # this is the text
                 (x, y), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center',bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3))
        
        x,y = s_oth[0], s_oth[1]
        plt.plot(x,y, 'g*',markersize='10')   
        plt.annotate('s_other', # this is the text
                 (x, y), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center',bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3))    
        
        plt.xlim((min(bes[0],wor[0],oth[0],s_wor[0], s_oth[0])-0.1 , max(bes[0],wor[0],oth[0],s_wor[0], s_oth[0])+0.1))
        plt.ylim((min(bes[1],wor[1],oth[1],s_wor[1], s_oth[1])-0.1 , max(bes[1],wor[1],oth[1],s_wor[1], s_oth[1])+0.1))
    
        
        plt.savefig('shrink.pdf',bbox_inches='tight')
        plt.show()
        
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
    



    