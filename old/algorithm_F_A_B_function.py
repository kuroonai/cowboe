#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
	Created on Wed Feb  5 10:05:31 2020
	
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

# python ~/Desktop/P3/pmftest/algorithm-m.py > /media/sf_dive/Research_work/afinalpaperrun/analysis/OPT/test/algorithm/alout.dat

import numpy as np
import pandas as pd
import os
import sys
import matplotlib as mpl
#mpl.use('Agg')
from matplotlib import pylab as plt
import pickle
from scipy.interpolate import interp1d as inp
#import time
import collections
import math
import time

def Kgiven(v):
    return v*2/0.5**2

def Vgiven(k):
    return 0.5*k*0.5**2

def bvn(A, B, V, kgiven, sc)	:
    samplingconsidered = sc
    def ww(Fmax):
        fc = 7.
        winw = 0.5
        #B = 4.5
        # A = 1.8 #fc*winw**2
        return round(1/((Fmax/A) + (1/B)), 6)
    
    def Aforw (Fmax, B, win):
        return(Fmax/(1/win - 1/B))

    def close(array_to_check, value):
        return min(enumerate(array_to_check), key=lambda s: abs(s[1] - value))

    def narrow(array_to_check, value):

        if array_to_check[0] < array_to_check[-1]:
            Arr = np.array([entry-value for entry in array_to_check])
            # print(Arr)
            l, r = list(Arr).index(max(Arr[Arr <= 0])), list(
                Arr).index(min(Arr[Arr >= 0]))
            #if l == r: r+=1
            # return array_to_check[l],array_to_check[r]
            return l, r

        elif array_to_check[0] > array_to_check[-1]:
            Arr = np.array([entry-value for entry in array_to_check])
            # print(Arr)
            l, r = list(Arr).index(max(Arr[Arr <= 0])), list(
                Arr).index(min(Arr[Arr >= 0]))
            #if l == r: r+=1
            # return array_to_check[l],array_to_check[r]
            return r, l

    def currentmax(begin, end):
        c_max = max(y[begin:end+1])
        return c_max

    def binary_search(start, stop, step, point, loc):
        l = []
        l.append(start)
        temp = step
        while l[-1] >= stop and l[-1] > point:
            # print(l[-1]+temp)
            l.append(l[-1] + temp)

        aa = close(l, point)[0]
        return loc+aa

    def ini_plot(extremes, crest, trough):

        plt.plot(y)
        plt.plot(crest, y[crest], "x", c='g')
        plt.plot(trough, y[trough], 'x', c='r')
        plt.savefig('extremes_%.2f_%.2f.pdf' % (A, B), bbox_inches='tight')
        plt.title('A = %.2f & B = %.2f' %(A,B))
        plt.show()

        extremes = np.sort(np.concatenate((crest, trough)))

        plt.plot(x, y)
        plt.xlim((x[0]+1, x[-1]-1))
        plt.plot(x[extremes], y[extremes], '*', c='k')

        for exr in x[trough]:
            plt.axvline(exr, ls='--', c='r')

        for exr in x[crest]:
            plt.axvline(exr, ls='--', c='g')

        plt.savefig('up and down_%.2f_%.2f.pdf' % (A, B), bbox_inches='tight')
        plt.title('A = %.2f & B = %.2f' %(A,B))
        plt.show()



    # font = {'family': 'font.sans-serif',
    #         'weight': 'bold',
    #         'size': 10}
    # mpl.rc('font', **font)
    
    # mpl.rcParams['font.sans-serif'] = 'Helvetica'
    # plt.rcParams['axes.labelsize'] = 16
    # plt.rcParams['axes.labelweight'] = 'bold'

    # wws = [ww(i) for i in np.linspace(0,20,100)]
    # plt.plot(np.linspace(0,20,100),wws)
    # plt.xlabel('magnitude of gradient dF/dr')
    # plt.ylabel('Window width - (A)')
    # plt.title('B = %f'%B);plt.show()
    
    pointname = input('\npoint name:\t')
    loc = '/media/sf_dive/Research_work/afinalpaperrun/analysis/OPT/test/algorithm/bvn/{}'.format(pointname)

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
    odextremes = collections.OrderedDict(sorted(dextremes.items()))
    step = 0

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
        # if len(direction) == 1:
        #     Rc = x[R] - ww(max(y[R:e+1]))
        #     windows.append(Rc)
        #     break
        ir = e 
        il = extremes[ narrow(extremes, e)[0] - 1]
        if il > R : il = il
        elif il <= R : il = R
        print('\n\t\tLeft: {} \t Right: {}\n'.format(il,ir))
        
        # if len(eranges) == 1 and abs(x[R] - x[eranges[0]]) <= abs(np.diff(x)[0]) : break
        
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
        
        # if y[il] < y[ir] : R = ir
        # elif   y[il] > y[ir]: R = il
        
    
        #else: R = im
        
        # if abs(Rc - x[il]) > abs(Rc - x[ir]) :
        #     if y[ir] < y [im]:
        #         R = ir
        #     else:
        #         R = im
        # elif abs(Rc - x[il]) < abs(Rc - x[ir]) : 
        #     if y[il] < y[im]:
        #         R = il
        #     else: 
        #         R = im
                
        print('\t\t\tFinal il: {}\tim: {}\tir: {}'.format(il,im,ir))    
        windows.append(x[R])
        Rs.append(R)
        print('\t\t\tAppending {:.6f} as window end'.format(x[R]))
        print('\t\t\tR value for next iteration is {}'.format(R))
        print('\nWindow {} is between {:.6f} - {:.6f} at {} - {}'.format(whilec, windows[-2], windows[-1], Rs[-2], Rs[-1]))
        
        #if whilec == 2: sys.exit()
        #if im == len(x)-1 and il == len(x)-1 and ir == len(x)-1 : break
        #if ir == len(x)-1 and il == len(x)-2:break
        
        print('\n\nTotal number of windows = {}\n'.format(len(windows)-1))
    
    
        #if whilec == 2: sys.exit()
        #if im == len(x)-1 and il == len(x)-1 and ir == len(x)-1 : break
        #if ir == len(x)-1 and il == len(x)-2:break
    ###########################

    windows = np.flip(np.unique(np.array(windows)))
    plt.plot(x, y)
    #plt.plot(x[extremes], y[extremes], '*',c ='k')

    for exr in windows:
        plt.axvline(exr, ls='--', c='r')

    plt.xticks(windows, rotation=90)
    plt.xlim(x[0]+1, x[-1]-1)
    plt.title('A = %.2f & B = %.2f' %(A,B))
    plt.show()
    
    Windows = windows.copy()
    startw = 14.5
    endw = 2.5

    Windows[0], Windows[-1]= startw, endw

    Rpos = np.array(Windows) #np.array(np.flip(windows))
    Windowwidth = np.diff(Rpos)

    total = 24 * samplingconsidered  # ns
    # total = 24*4000000*2 #24 windows - 5000000 2fs steps 10 ns
    fracsamplingtime = [i/sum(Windowwidth) for i in Windowwidth]
    #Samplingtime = [int((i*total)/(sum(Windowwidth)*400000)) for i in Windowwidth]
    Samplingtime = [(j*total) for j in fracsamplingtime]
    #plt.savefig(plotloc+'nWidowmarker_%.2f_%.2f.pdf'%(A,B),bbox_inches = 'tight')
    #plt.title('B = %f'%B);plt.show()

    plt.plot(Samplingtime[::-1], 'r^--')
    plt.xlim(-0.25, len(windows)-1.25)
    # plt.ylim(2,40)
    plt.ylabel('ns', fontsize=14, weight='bold')
    plt.xlabel('Windows', fontsize=14, weight='bold')
    #plt.savefig(plotloc+'nsamplingtime%.2f_%.2f.pdf'%(A,B),bbox_inches = 'tight')
    plt.title('A = %.2f & B = %.2f' %(A,B))
    plt.show()

    # check loop - window width comparison

    actualind = [close(x, i)[0] for i in windows]

    #actualind = Rs
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
    #newpositions = [close(x, wws-ac)[1] for ac, wws in zip(actualww, windows[:-1])]
    newpositions = [ wws-ac for ac, wws in zip(actualww, windows[:-1])]

    # newpositions.append(x[-1])
    newpositions.insert(0, x[0])

    plt.plot(x, y)
    for exr in windows:
        plt.axvline(exr, ls='--', c='r')

    plt.xticks(windows, rotation=90)
    plt.xlim(x[0]+1, x[-1]-1)
    #plt.title('B = %f'%B);plt.show()

    # plt.plot(x,y)
    for exr in newpositions:
        plt.axvline(exr, ls='--', c='g')

    # plt.xticks(newpositions,rotation=90)
    plt.xlim(x[0]+1, x[-1]-1)
    plt.savefig('alligned_%.2f_%.2f.pdf' % (A, B), bbox_inches='tight')
    plt.title('A = %.2f & B = %.2f' %(A,B))
    plt.show()

    #######################

    plt.plot(x, y)
    for exr in windows:
        plt.axvline(exr, ls='--', c='r')

    plt.xticks(windows, rotation=90)
    plt.xlim(x[0]+1, x[-1]-1)
    plt.savefig('n_actual_%.2f_%.2f.pdf' % (A, B), bbox_inches='tight')
    plt.title('A = %.2f & B = %.2f' %(A,B))
    plt.show()

    plt.plot(x, y)
    for exr in newpositions:
        plt.axvline(exr, ls='--', c='g')

    plt.xticks(newpositions, rotation=90)
    plt.xlim(x[0]+1, x[-1]-1)
    plt.savefig('n_theory_%.2f_%.2f.pdf' % (A, B), bbox_inches='tight')
    plt.title('A = %.2f & B = %.2f' %(A,B))
    plt.show()

    ##############################

    plt.plot(windows), plt.plot(newpositions)
    plt.title('A = %.2f & B = %.2f' %(A,B))
    plt.show()

    #pd.options.display.float_format = '{:,.4f}'.format

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
    errors = [actualww - abs(np.diff(windows))]
    

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


    print('\nDone!')

    with open('/media/sf_dive/Research_work/afinalpaperrun/analysis/OPT/test/algorithm/{:.2f}-{:.2f}.pkl'.format(A,B), 'wb') as f:  # Python 3: open(..., 'wb')
        data_to_save = {'x':x, 'y':y, 'extremes':extremes, 'A':A, 'B':B, '\
                        wins':len(windows)-1, 'newpos':newpositions, 'windows':windows,\
                            'errors':errors, 'sample':Samplingtime}
        #pickle.dump([x, y, extremes, A, B, len(windows)-1, newpositions, windows, errors, Samplingtime], f, protocol=-1)
        pickle.dump(data_to_save, f, protocol=-1)
    
    
    print('\n')
    writeinputdic(pointname,'graham',A,B,V,windows)
    print('\n')
    writeinputdic(pointname,'cedar',A,B,V,windows)
    print('\n')
    writeinputdic(pointname,'beluga',A,B,V,windows)
    print('\n')
    writeinputdic(pointname,'niagara',A,B,V,windows)
    print('\n')
    
    return None #'Done!'#A, B, len(windows)-1, tuple(newpositions), tuple(windows), errors

def writeinputdic(pointname, server, A, B, V, windows):
    
    
    if server == 'cedar' :
        tc = 192
        sf = '/scratch/vasudevn/OPT/NEW'
    elif server == 'graham' : 
        tc = 160
        sf = '/project/6003277/vasudevn/OPT/NEW'
    elif server == 'beluga' : 
        tc = 160
        sf ='/lustre04/scratch/vasudevn/OPT'
    elif server == 'niagara': 
        tc = 160
        sf = '/gpfs/fs0/scratch/x/xili/vasudevn/OPT'
    
    A = str(A)
    B = str(B)
    V = str(V)
    
    print("p%s_%s = {\n\
    'A'             :%s,\n\
    'B'             :%s,\n\
    'V'             :%s,\n\
    'wins'          :%d,\n\
    'sc'            :8,\n\
    'lmr'           :'/A=%s_B=%s_V=%s.txt',\n\
    'subloc'        :'%s/%s',\n\
    'loc'           :'/media/sf_dive/Research_work/afinalpaperrun/analysis/OPT/test/algorithm/bvn/%s',\n\
    'datafileloc'   :'/media/sf_dive/Research_work/afinalpaperrun/analysis/OPT/test/algorithm/datafiles/100mc',\n\
    'outputloc'     :'/media/sf_dive/Research_work/afinalpaperrun/analysis/OPT/test/algorithm/folder',\n\
    'server'        :'%s',\n\
    'total_cpu'     :%d,\n\
    'pair'          :16.0,\n\
    'skin'          :2.0 ,\n\
    'ts'            :2,\n\
    'kspace'        :'pppm 1e-6',\n\
    'mpc'           :512,\n\
    'mail'          :'TRUE',\n\
    'ringcoms'      :'TRUE',\n\
    'traj'          :'TRUE',\n\
    'f'             :['100mc'],\n\
    'justsh'        :'nd'\n\
            }"%(pointname,server,A,B,V,len(windows)-1,A,B,V,sf,pointname,pointname,server,tc ))
          
def pickleunload(f,key):
    with open('%s'%f, 'rb') as f:  # Python 3: open(..., 'rb')
        #px, py, pextremes, pA, pB, ptot_windows, pnewpositions, pwindows, perrors, pSamplingtime = pickle.load(f)
        data_loaded = pickle.load(f)
    #return px, py, pextremes, pA, pB, ptot_windows, pnewpositions, pwindows, perrors, pSamplingtime
    return data_loaded[key]

def Kcalc(windows, A, B, V, kgiven):
    '''
    
    V = 0.5 * K * (X - X0)**2
    K = 2*V/(X-X0)**2
    
    '''
    Windows = windows.copy()
    startw = 14.5
    endw = 2.5
    
    Windows[0], Windows[-1]= startw, endw
    
    
    
    V_x = np.linspace(-0.5,0.5,100)
    test_V = [ 0.5*kgiven*(0-i)**2 for i in V_x]
    plt.plot(V_x, test_V)
    plt.axvline(0.0, linestyle='-.', c='k')
    plt.axvline(-0.5, linestyle='--', c='r')
    plt.axvline(0.5, linestyle='--', c='r')
    plt.axhline(test_V[0], linestyle=':', c='g')
    plt.ylabel(r'$\Delta$ V')
    plt.xlabel(r'$\xi$')
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
    plt.xlabel(r'$\xi$')
    plt.savefig('native_window_potential_%.2f_%.2f.pdf' % (A, B),bbox_inches = 'tight')
    plt.show()
    
    Ms =[]
    k = kgiven
    ww = 0.5
    for M in np.arange(2.5,14.5,.5):
        Ms.append(windowsplot(k,M-ww,M+ww))
    
    
    plt.axhline(0.5*k*ww**2, linestyle='--', c='r')   
    plt.xticks(Ms, rotation=90)
    plt.ylabel(r'$\Delta$ V')
    plt.xlabel(r'$\xi$')
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

