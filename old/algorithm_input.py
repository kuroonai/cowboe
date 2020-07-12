#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 16:54:00 2020

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

import sys
import os
import shutil
import time
import random
import numpy as np
import datetime
from matplotlib import pylab as plt
from shapely.geometry import Polygon
import pickle as p
import seaborn as sns
from scipy import stats
from math import sqrt, fabs, erf, log
from pdf2image import convert_from_path
import json as j
import glob
import pandas as pd
import glob
from scipy.stats import norm
import scipy.constants as constants

p16_graham = {
    'A'             :2.8949,
    'B'             :2.0,
    'V'             :0.9675,
    'wins'          :16,
    'sc'            :8,
    'lmr'           :'/A=2.8949_B=2.0_V=0.9675.txt',
    'subloc'        :'/project/6003277/vasudevn/OPT/NEW/16',
    'loc'           :'/media/sf_dive/Research_work/afinalpaperrun/analysis/OPT/test/algorithm/bvn/16',
    'datafileloc'   :'/media/sf_dive/Research_work/afinalpaperrun/analysis/OPT/test/algorithm/datafiles/100mc',
    'outputloc'     :'/media/sf_dive/Research_work/afinalpaperrun/analysis/OPT/test/algorithm/folder',
    'server'        :'graham',
    'total_cpu'     :160,
    'pair'          :16.0,
    'skin'          :2.0 ,
    'ts'            :2,
    'kspace'        :'pppm 1e-6',
    'mpc'           :512,
    'mail'          :'TRUE',
    'ringcoms'      :'TRUE',
    'traj'          :'TRUE',
    'f'             :['100mc'],
    'justsh'        :'nd'
            }


p16_cedar = {
    'A'             :2.8949,
    'B'             :2.0,
    'V'             :0.9675,
    'wins'          :16,
    'sc'            :8,
    'lmr'           :'/A=2.8949_B=2.0_V=0.9675.txt',
    'subloc'        :'/scratch/vasudevn/OPT/NEW/16',
    'loc'           :'/media/sf_dive/Research_work/afinalpaperrun/analysis/OPT/test/algorithm/bvn/16',
    'datafileloc'   :'/media/sf_dive/Research_work/afinalpaperrun/analysis/OPT/test/algorithm/datafiles/100mc',
    'outputloc'     :'/media/sf_dive/Research_work/afinalpaperrun/analysis/OPT/test/algorithm/folder',
    'server'        :'cedar',
    'total_cpu'     :192,
    'pair'          :16.0,
    'skin'          :2.0 ,
    'ts'            :2,
    'kspace'        :'pppm 1e-6',
    'mpc'           :512,
    'mail'          :'TRUE',
    'ringcoms'      :'TRUE',
    'traj'          :'TRUE',
    'f'             :['100mc'],
    'justsh'        :'nd'
            }


p16_beluga = {
    'A'             :2.8949,
    'B'             :2.0,
    'V'             :0.9675,
    'wins'          :16,
    'sc'            :8,
    'lmr'           :'/A=2.8949_B=2.0_V=0.9675.txt',
    'subloc'        :'/lustre04/scratch/vasudevn/OPT/16',
    'loc'           :'/media/sf_dive/Research_work/afinalpaperrun/analysis/OPT/test/algorithm/bvn/16',
    'datafileloc'   :'/media/sf_dive/Research_work/afinalpaperrun/analysis/OPT/test/algorithm/datafiles/100mc',
    'outputloc'     :'/media/sf_dive/Research_work/afinalpaperrun/analysis/OPT/test/algorithm/folder',
    'server'        :'beluga',
    'total_cpu'     :160,
    'pair'          :16.0,
    'skin'          :2.0 ,
    'ts'            :2,
    'kspace'        :'pppm 1e-6',
    'mpc'           :512,
    'mail'          :'TRUE',
    'ringcoms'      :'TRUE',
    'traj'          :'TRUE',
    'f'             :['100mc'],
    'justsh'        :'nd'
            }


p16_niagara = {
    'A'             :2.8949,
    'B'             :2.0,
    'V'             :0.9675,
    'wins'          :16,
    'sc'            :8,
    'lmr'           :'/A=2.8949_B=2.0_V=0.9675.txt',
    'subloc'        :'/gpfs/fs0/scratch/x/xili/vasudevn/OPT/16',
    'loc'           :'/media/sf_dive/Research_work/afinalpaperrun/analysis/OPT/test/algorithm/bvn/16',
    'datafileloc'   :'/media/sf_dive/Research_work/afinalpaperrun/analysis/OPT/test/algorithm/datafiles/100mc',
    'outputloc'     :'/media/sf_dive/Research_work/afinalpaperrun/analysis/OPT/test/algorithm/folder',
    'server'        :'niagara',
    'total_cpu'     :160,
    'pair'          :16.0,
    'skin'          :2.0 ,
    'ts'            :2,
    'kspace'        :'pppm 1e-6',
    'mpc'           :512,
    'mail'          :'TRUE',
    'ringcoms'      :'TRUE',
    'traj'          :'TRUE',
    'f'             :['100mc'],
    'justsh'        :'nd'
            }

p17_graham = {
    'A'             :1.6552,
    'B'             :2.0,
    'V'             :0.8849,
    'wins'          :22,
    'sc'            :8,
    'lmr'           :'/A=1.6552_B=2.0_V=0.8849.txt',
    'subloc'        :'/project/6003277/vasudevn/OPT/NEW/17',
    'loc'           :'/media/sf_dive/Research_work/afinalpaperrun/analysis/OPT/test/algorithm/bvn/17',
    'datafileloc'   :'/media/sf_dive/Research_work/afinalpaperrun/analysis/OPT/test/algorithm/datafiles/100mc',
    'outputloc'     :'/media/sf_dive/Research_work/afinalpaperrun/analysis/OPT/test/algorithm/folder',
    'server'        :'graham',
    'total_cpu'     :160,
    'pair'          :16.0,
    'skin'          :2.0 ,
    'ts'            :2,
    'kspace'        :'pppm 1e-6',
    'mpc'           :512,
    'mail'          :'TRUE',
    'ringcoms'      :'TRUE',
    'traj'          :'TRUE',
    'f'             :['100mc'],
    'justsh'        :'nd'
            }


p17_cedar = {
    'A'             :1.6552,
    'B'             :2.0,
    'V'             :0.8849,
    'wins'          :22,
    'sc'            :8,
    'lmr'           :'/A=1.6552_B=2.0_V=0.8849.txt',
    'subloc'        :'/scratch/vasudevn/OPT/NEW/17',
    'loc'           :'/media/sf_dive/Research_work/afinalpaperrun/analysis/OPT/test/algorithm/bvn/17',
    'datafileloc'   :'/media/sf_dive/Research_work/afinalpaperrun/analysis/OPT/test/algorithm/datafiles/100mc',
    'outputloc'     :'/media/sf_dive/Research_work/afinalpaperrun/analysis/OPT/test/algorithm/folder',
    'server'        :'cedar',
    'total_cpu'     :192,
    'pair'          :16.0,
    'skin'          :2.0 ,
    'ts'            :2,
    'kspace'        :'pppm 1e-6',
    'mpc'           :512,
    'mail'          :'TRUE',
    'ringcoms'      :'TRUE',
    'traj'          :'TRUE',
    'f'             :['100mc'],
    'justsh'        :'nd'
            }

def step_allocator(st, step_size, samplingconsidered, w) :  
    evensteps = False
    if evensteps:
        st_step = np.full((w,), int((24*samplingconsidered*10**6/step_size)/w))
        #st_step[np.argmin(st_step)] += int(24*samplingconsidered*10**6/step_size - np.sum(st_step))
    else:                           
        st_step = np.array([int(s*10**6/step_size) for s in st])
        st_step[np.argmin(st_step)] += int(24*samplingconsidered*10**6/step_size - np.sum(st_step))
    #st_step[np.argmin(st_step)] += 23*samplingconsidered*10**6/step_size - np.sum(st_step)
    return st_step

def reruntime(need, done):
    return int((need-done)*30*60/6000000)+15
    
def lammps_input(inputs):
    '''
    generated lammps input file and other related files

    '''
       
    
    A                   =inputs['A']
    B                   =inputs['B']
    V                   =inputs['V']
    wins                = inputs['wins']
    loc                 = inputs['loc']
    lmr                 = inputs['lmr']
    datafileloc         = inputs['datafileloc']
    server              = inputs['server']
    total_cpu           = inputs['total_cpu']
    pair                = inputs['pair']
    skin                = inputs['skin']
    ts                  = inputs['ts']
    kspace              = inputs['kspace']
    mpc                 = inputs['mpc']
    mail                = inputs['mail']
    ringcoms            = inputs['ringcoms']
    traj                = inputs['traj']
    outputloc           = inputs['outputloc']
    subloc              = inputs['subloc']
    f                   = inputs['f']
    justsh              = inputs['justsh'] 
    sc                  = inputs['sc']
    
    try:
        wstart = inputs['wstart']
        wstop  = inputs['wstop']+1
    
    except:
        wstart = 0
        wstop  = wins
        
    lmr_data = np.loadtxt(loc+lmr,dtype='float64')
    left, middle, right, K, samplingtime = lmr_data[:,0], lmr_data[:,1], lmr_data[:,2], \
                                 lmr_data[:,3], lmr_data[:,4]
    
    st_step = step_allocator(samplingtime, 2, sc, wins) #section in ns, stepsize in fs, sampling for each windows 

    #TIME = np.array([int(round(step*30/6000000))+1 for step in st_step])
    #TIME_min = np.array([int(60*(step*30/6000000))+60 for step in st_step])
    if server == 'graham': TIME_min = np.array([int(step/3600)+60 for step in st_step])
    elif server =='cedar' or server == 'niagara': TIME_min = np.array([int(step/3900) for step in st_step])
    else: TIME_min = np.array([int(step/3600)+60 for step in st_step])
    #print(st_step, TIME_min)
    
    if server=="beluga" or server=='niagara':
        corepnode=40
    elif server=="cedar":
        corepnode=48
        total_cpu=192
    else: 
        corepnode=32
        
    node = int(total_cpu/corepnode)
    
    st=time.time()
    oldstdout=sys.stdout
    
    os.chdir(outputloc)
    print('\nFor\tA : {}\tB : {}\t\tV : {}\n'.format(A,B,V))
    print('\nStart: {:.4f} {}, Stop: {:.4f} {}, Windows: {:d}\n'.format(left[0], chr(8491), right[-1], chr(8491), wins))
    print(('Pair distance \t= %.4f %s\nTime step \t= %.4f\nKspace \t\t= %s\nNo. of nodes \t= %d')%(pair, str(chr(8491)), ts, kspace, node))
    print(('Server \t\t= %s\nMem-per-cpu \t= %s MB\nCore-per-node \t= %d\ntotal-cpu \t= %d')%(server,mpc,corepnode,total_cpu))
    print(('Mailing \t= %s\nTrajectory \t= %s\n')%(mail,traj))
    
    ftime = str(datetime.datetime.now().strftime('%H-%M-%S_%d-%m-%Y'))
    
    for fold in f:
        
        '''
        change to output folder 100mc, 110mc etc
        check and creates folder with timestamp
        changes location to the new folder
        '''
        
        folder=fold
        path=outputloc+'/%s'%folder
        os.chdir(path)
        
        os.mkdir('%s_%s-%s'%(server, ftime, fold))

        path = path+'/'+'%s_%s-%s'%(server, ftime, fold)
        os.chdir(path)
        
        #################### to get proper data file close to middle posisiton ##############
        data_dict = {}
        actual_data_files = np.arange(2.5,14.5,0.5)
        
        for index, d in enumerate(actual_data_files):
            data_dict.update({'%d.data'%index:d})
        
        vals = np.fromiter(data_dict.values(), dtype=float)
        #print(data_dict,vals)        
        for index, m in enumerate(middle):
            val_min = np.argmin(np.array([abs(d-m) for d in vals]))
            scr = '{}/{}.data'.format(datafileloc,val_min)
            dst = '{}/{}.data'.format(path,index)
            shutil.copy(scr, dst)
        
        ####################to make colvar %d.inp file#########################
        no=0
        for i in range(wstart, wstop):
            
            l = left[i]
            m = middle[i]
            r = right[i]
        
            filename='%d.inp'%i
            no=no+1
            sys.stdout=open(filename,'w')
            print(('indexFile ../group.ndx\ncolvarsTrajAppend no\ncolvarsTrajFrequency 500\ncolvarsRestartFrequency 500\n\ncolvar {\nname dist\ndistanceZ {    \nmain { indexGroup pol }    \nref { indexGroup lay1 }\naxis (1,0,0) \nforceNoPBC yes   \n}\nlowerBoundary %f\nupperBoundary %f\nhardUpperBoundary yes\nhardLowerBoundary  yes\n}\ncolvar {\nname d\ndistance {\ngroup1 {indexGroup e1}\ngroup2 {indexGroup e2}\n}\n}\n\n\ncolvar {\nname dv\ndistanceVec {\ngroup1 {indexGroup e1}\ngroup2 {indexGroup e2}\n}\n}\n\n\ncolvar {\nname dd\ndistanceDir {\ngroup1 {indexGroup e1}\ngroup2 {indexGroup e2}\n}\n}\n\n\ncolvar {\nname di\ndistanceInv {\ngroup1 {indexGroup e1}\ngroup2 {indexGroup e2}\n}\n}\n\nharmonic {  \ncolvars dist  \noutputCenters on  \nforceConstant %.4f \ncenters %f         # initial distance  \ntargetCenters %f  # final distance  \ntargetNumSteps %d  \ntargetNumstages 0\noutputAccumulatedWork on\n}')%(l,r,K[i],m,m,st_step[i]))
            sys.stdout.close();
        sys.stdout=oldstdout;
        
        ####################to make colvar ini%d.inp file#########################
    
        no=0
        for i in range(wstart, wstop):
        
            l = left[i]
            m = middle[i]
            r = right[i]
        
            filename='ini%d.inp'%i
            no=no+1
            sys.stdout=open(filename,'w')
            print(('indexFile ../group.ndx\ncolvarsTrajAppend no\ncolvarsTrajFrequency 500\ncolvarsRestartFrequency 500\n\ncolvar {\nname dist\ndistanceZ {    \nmain { indexGroup pol }    \nref { indexGroup lay1 }\naxis (1,0,0)    \n}\n}\n\nharmonic {  \ncolvars dist  \noutputCenters on  \nforceConstant %.4f  \ncenters %f         # initial distance  \ntargetCenters %f  # final distance  \ntargetNumSteps 200000  \ntargetNumstages 0\noutputAccumulatedWork on\n}')%(K[i],m,m))
    
            sys.stdout.close();
        sys.stdout=oldstdout;
        
        ########################to make shell script#############################
    
        for i in range (wins):
            filename='%d.sh'%i
            sys.stdout=open(filename,'w')
            print('#!/bin/bash -l')
            if server =="cedar" or server =="beluga" or server == 'niagara':    
                print('#SBATCH --account=def-xili')
            else:
                print('#SBATCH --account=rrg-xili')
            print(('#SBATCH -N %d')%node)
            if server != 'niagara':
                print(('#SBATCH -n %d')%(node*corepnode))
                print(('#SBATCH --mem-per-cpu=%dmb')%mpc)
                if K[i] > 100.0 :
                    print(('#SBATCH --time=00:%d:00')%int(TIME_min[i]+60))
                else:
                    print(('#SBATCH --time=00:%d:00')%int(TIME_min[i]))
                
            else:
                print(('#SBATCH --ntasks=%d')%(node*corepnode))
                if TIME_min[i] <= 24*60 and K[i] > 100.0 : print('#SBATCH --time=00:{}:00'.format(TIME_min[i]+60))
                elif TIME_min[i] <= 24*60 and K[i] < 100.0 : print('#SBATCH --time=00:{}:00'.format(TIME_min[i]))
                else : print(('#SBATCH --time=23:59:59'))
                
            if mail=='TRUE':
                print('#SBATCH --mail-type=ALL')
                print('#SBATCH --mail-user=vasudevn@mcmaster.ca')
            
            
            print(('#SBATCH -J {}-win-{}').format(subloc.split('/')[-1],i))
            
            if server=="beluga":
                print('\nmodule load nixpkgs/16.09  intel/2018.3  openmpi/3.1.2\nmodule load lammps-omp/20190807\n')
                print(('srun -n %d lmp_icc_openmpi  < %d.in')%(node*corepnode,i))
            elif server == "cedar" or server == "graham":
                print('\nmodule load nixpkgs/16.09  intel/2018.3  openmpi/3.1.2\nmodule load lammps-omp/20190807\n')
                print(('srun -n %d lmp_icc_openmpi  < %d.in')%(node*corepnode,i))
            elif server == 'niagara':
                print('\nmodule load intel/2018.2 intelmpi/2018.2 fftw-mpi/3.3.7\n')
                print(('srun lmp  < %d.in')%(i))
            sys.stdout.close();
        sys.stdout=oldstdout;
        
        ########################to make lammps in file#############################
    

        if folder =="100mc": #8086
            groupids=["group cel id 1:10080","group pol id 10081:10209","group water id 10210:28209","group celandpol id 1:10080 10081:10209","group lay1 id 757:1008 1765:2016 2773:3024 3781:4032 4789:5040 5797:6048 6805:7056 7813:8064	 8821:9072 9829:10080","group celandwater id 1:10080 10210:28209","group2ndx ../group.ndx cel pol water lay1 celandpol celandwater polandwater r1 r2 r3 r4 r5 e1 e2","group polandwater id 10081:10209	10210:28209","group r1 id 10081:10102 10206:10209","group r2 id 10103:10122 10202:10205","group r3 id 10123:10141 10194:10197 10198:10201","group r4 id 10141:10160 10186:10189 10190:10193","group r5 id 10161:10185","group e1 id 10083","group e2 id 10166"]
        if folder =="100pe": #8087
            groupids=["group cel id 1:10080","group pol id 10081:10188","group water id 10189:28188","group celandpol id 1:10080 10081:10188","group lay1 id 757:1008 1765:2016 2773:3024 3781:4032 4789:5040 5797:6048 6805:7056 7813:8064	 8821:9072 9829:10080","group celandwater id 1:10080 10189:28188","group2ndx ../group.ndx cel pol water lay1 celandpol celandwater polandwater r1 r2 r3 r4 r5 e1 e2","group polandwater id 10081:10188	10189:28188","group r1 id 10081:10103","group r2 id 10104:10124","group r3 id 10125:10145","group r4 id 10146:10166","group r5 id 10167:10188","group e1 id 10083","group e2 id 10177"]
        if folder =="110mc": #13462
            groupids=["group cel id 1:10080","group pol id 10081:10209","group water id 10210:28209","group celandpol id 1:10080 10081:10209","group lay1 id 1:504 757:1260 2521:3024 4537:5040 6553:7056","group celandwater id 1:10080 10210:28209","group2ndx ../group.ndx cel pol water lay1 celandpol celandwater polandwater r1 r2 r3 r4 r5 e1 e2","group polandwater id 10081:10209	10210:28209","group r1 id 10081:10102 10206:10209","group r2 id 10103:10122 10202:10205","group r3 id 10123:10141 10194:10197 10198:10201","group r4 id 10141:10160 10186:10189 10190:10193","group r5 id 10161:10185","group e1 id 10083","group e2 id 10166"]
        if folder =="110pe": #13463
            groupids=["group cel id 1:10080","group pol id 10081:10188","group water id 10189:28188","group celandpol id 1:10080 10081:10188","group lay1 id 1:504 757:1260 2521:3024 4537:5040 6553:7056","group celandwater id 1:10080 10189:28188","group2ndx ../group.ndx cel pol water lay1 celandpol celandwater polandwater r1 r2 r3 r4 r5 e1 e2","group polandwater id 10081:10188	10189:28188","group r1 id 10081:10103","group r2 id 10104:10124","group r3 id 10125:10145","group r4 id 10146:10166","group r5 id 10167:10188","group e1 id 10083","group e2 id 10177"]
        if folder =="110smc": #13540
            groupids=["group cel id 1:10200","group pol id 10201:10329","group water id 10330:28449","group celandpol id 1:10200 10201:10329","group lay1 id 1:2640","group celandwater id 1:10200 10330:28449","group2ndx ../group.ndx cel pol water lay1 celandpol celandwater polandwater r1 r2 r3 r4 r5 e1 e2","group polandwater id 10201:10329 10330:28449","group r1 id 10201:10222 10326:10329","group r2 id 10223:10242 10322:10325","group r3 id 10243:10261 10314:10321","group r4 id 10262:10280 10306:10313","group r5 id 10281:10305","group e1 id 10203","group e2 id 10286"]
        if folder =="110spe":#13541
            groupids=["group cel id 1:10200","group pol id 10201:10308","group water id 10309:28428","group celandpol id 1:10200 10201:10308","group lay1 id 1:2640","group celandwater id 1:10200 10309:28428","group2ndx ../group.ndx cel pol water lay1 celandpol celandwater polandwater r1 r2 r3 r4 r5 e1 e2","group polandwater id 10201:10308 10309:28428","group r1 id 10201:10223","group r2 id 10224:10244","group r3 id 10245:10265","group r4 id 10266:10286","group r5 id 10287:10308","group e1 id 10203","group e2 id 10297"]
    
            
        no=0;
        for k in range(wstart, wstop):
        
            filename='%d.in'%k
            no=no+1
            sys.stdout=open(filename,'w')
            print(('#colvar - %s - window - %d\n\nshell mkdir dz\nshell cd dz\nshell mkdir results\nshell mkdir restarts\n')%(folder,k))
            print(('#initial configuration\n\nunits real\natom_style full\ndimension 3\nnewton on\nboundary  p p p\nneighbor %.2f bin\nneigh_modify    delay 0 every 1 check yes page 1000000 one 50000')%skin)
            print(('\nbond_style      harmonic\nangle_style     harmonic\ndihedral_style  harmonic\npair_style lj/cut/coul/long %f\npair_modify mix geometric shift yes\nkspace_style    %s\nspecial_bonds lj/coul 0 0 1 angle yes dihedral yes')%(pair,kspace))
            print(('\n#reading data and thermo details\n\nread_data       ../%d.data\n')%k)
            print(('#group assignment\n\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n')%(groupids[0],groupids[1],groupids[2],groupids[4],groupids[3],groupids[5],groupids[7],groupids[8],groupids[9],groupids[10],groupids[11],groupids[12],groupids[13],groupids[14],groupids[6]))
    
            
            if folder =="100mc" or folder =="100pe":
                shake="fix     	1 all shake 1e-4 20 0 b 3 4 6 7 #a 12"
                recen="fix recen1	lay1 recenter 5.800 1.389 1.278 units box"
            elif folder=="110mc" or folder=="110pe":
                shake="fix     	1 all shake 1e-4 20 0 b 3 4 6 7 #a 12"
                recen="fix recen1	lay1 recenter 8.21961 0.445948 -0.483543 units box"
            else:
                shake="fix     	1 all shake 1e-4 20 0 b 3 4 6 9 #a 12"
                recen="fix recen1	lay1 recenter 8.62593 0.401141 -0.163193 units box"
                
            print(('\nreset_timestep  0\ntimestep	%d\n')%ts)
            print(('\nvelocity all create 300.0 %d rot yes dist gaussian')%random.randint(9999,10000000))
            print(('\n#NVT fix and shake\n%s\nfix 		2 all nvt temp 300 300 100\nfix colvar1 all colvars ../ini%d.inp tstat 2 output %s%d unwrap no\n%s\nrun 200000\nunfix colvar1\nunfix 1\nunfix 2\nunfix recen1\n')%(shake,k,folder,k,recen))
            print('\nreset_timestep  0\ntimestep	2\n')
            print(('\n#NVT fix and shake\n%s\nfix 		2 all nvt temp 300 300 100\nfix colvar1 all colvars ../%d.inp tstat 2 output %s%d unwrap no\n\n')%(shake,k,folder,k))
            print(('\n#compute section\ncompute  com1 cel com\nfix com11 cel ave/time 1000 1 1000 c_com1[*] file ./results/%d-cel.dat\ncompute  com2 lay1 com\nfix com12 lay1 ave/time 1000 1 1000 c_com2[*] file ./results/%d-lay1.dat\ncompute  com3 pol com\nfix com13 pol ave/time 1000 1 1000 c_com3[*] file ./results/%d-pol.dat\n')%(k,k,k))
            if ringcoms=='TRUE': 
                print('\ncompute rgpol pol gyration\ncompute msd1 pol msd\n\nvariable rgp1 equal c_rgpol\n')
                print(('variable st equal step\n\nvariable etoe equal "sqrt((xcm(e1,x)-xcm(e2,x))^2 + (xcm(e1,y)-xcm(e2,y))^2 + (xcm(e1,z)-xcm(e2,z))^2 )"\nvariable oetoe equal 1/v_etoe\n\nvariable as equal "(xcm(e2,x) - xcm(e1,x))*v_oetoe"\nvariable asy equal "(xcm(e2,y) - xcm(e1,y))*v_oetoe"\n\nvariable asz equal "(xcm(e2,z) - xcm(e1,z))*v_oetoe"\n\nvariable k1 equal "sqrt((xcm(r1,x)-xcm(r2,x))^2 + (xcm(r1,y)-xcm(r2,y))^2 + (xcm(r1,z)-xcm(r2,z))^2 )"\nvariable k2 equal "sqrt((xcm(r2,x)-xcm(r3,x))^2 + (xcm(r2,y)-xcm(r3,y))^2 + (xcm(r2,z)-xcm(r3,z))^2 )"\nvariable k3 equal "sqrt((xcm(r3,x)-xcm(r4,x))^2 + (xcm(r3,y)-xcm(r4,y))^2 + (xcm(r3,z)-xcm(r4,z))^2 )"\nvariable k4 equal "sqrt((xcm(r4,x)-xcm(r5,x))^2 + (xcm(r4,y)-xcm(r5,y))^2 + (xcm(r4,z)-xcm(r5,z))^2 )"\n\nvariable angx equal angmom(pol,x)\nvariable angy equal angmom(pol,y)\nvariable angz equal angmom(pol,z)\n\nfix etoes all print 500 "${st} ${as} ${asy} ${asz} ${etoe}" file ./results/%d-etoes.dat title "#etoe"\nfix rogpol all print 500 "${st} ${rgp1}" file ./results/%d-rogpol.dat title "#Rg-pol"\nfix angmompol all print 500 "${st} ${angx} ${angy} ${angz}" file ./results/%d-angmompol.dat title "#angmom_x angmom_y angmom_z"\nfix msdpol pol ave/time 500 1 500 c_msd1[*] file ./results/%d-msdpol.dat\nfix kuhns all print 500 "${st} ${k1} ${k2} ${k3} ${k4}" file ./results/%d-kuhns.dat title "#k1 k2 k3 k4"')%(k,k,k,k,k))
            print(('\n%s')%recen)
            print('\nvariable d equal c_com3[1]-c_com2[1]')
            print('\n#output section\nrestart		1000  ./restarts/crystal.restart1 ./restarts/crystal.restart2\nrestart 50000 ./restarts/ps.restart')
            if traj=='TRUE':print('\ndump    dump3 all custom 500 ps1.lammpstrj id type xu yu zu vx vy vz ix iy iz\ndump dcd1 all dcd 500 ps1.dcd')
            else:print('\n#no traj for this run#\n')
            print(('\n\nwrite_data	npt.data\n\nthermo          500\nthermo_style  	custom step press temp vol density  pe ke etotal evdwl ecoul elong epair ebond eangle edihed  emol   c_com3[1] c_com2[1] v_d \nthermo_modify flush yes\n\nrun %d upto')%st_step[k])
            
            
            sys.stdout.close(); 
        sys.stdout=oldstdout;
        
    
        no=0;
        for k in range(wstart, wstop):
        
            filename='%drestart.in'%k
            no=no+1
            sys.stdout=open(filename,'w')
            print(('#colvar - %s - window - %d - restart\n\nshell mkdir dz-restart\nshell cd dz-restart\nshell mkdir results\n')%(folder,k))
            print(('#initial configuration\n\nunits real\natom_style full\ndimension 3\nnewton on\nboundary  p p p\nneighbor %.2f bin\nneigh_modify    delay 0 every 1 check yes page 1000000 one 50000')%skin)
            print(('\nbond_style      harmonic\nangle_style     harmonic\ndihedral_style  harmonic\npair_style lj/cut/coul/long %f\npair_modify mix geometric shift yes\nkspace_style    %s\nspecial_bonds lj/coul 0 0 1 angle yes dihedral yes')%(pair,kspace))
            print(('\n#reading data and thermo details\n\nread_data       ../restart.data\n'))
            print(('#group assignment\n\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n')%(groupids[0],groupids[1],groupids[2],groupids[4],groupids[3],groupids[5],groupids[7],groupids[8],groupids[9],groupids[10],groupids[11],groupids[12],groupids[13],groupids[14],groupids[6]))
            print(('\n\ntimestep 2'))
            
            if folder =="100mc" or folder =="100pe":
                shake="fix     	1 all shake 1e-4 20 0 b 3 4 6 7 #a 12"
                recen="fix recen1	lay1 recenter 5.800 1.389 1.278 units box"
            elif folder=="110mc" or folder=="110pe":
                shake="fix     	1 all shake 1e-4 20 0 b 3 4 6 7 #a 12"
                recen="fix recen1	lay1 recenter 8.21961 0.445948 -0.483543 units box"
            else:
                shake="fix     	1 all shake 1e-4 20 0 b 3 4 6 9 #a 12"
                recen="fix recen1	lay1 recenter 8.62593 0.401141 -0.163193 units box"
                
            print(('\n#NVT fix and shake\n%s\nfix 		2 all nvt temp 300 300 100\nfix colvar1 all colvars ../%d.inp tstat 2 output %s%d unwrap no\n\n')%(shake,k,folder,k))
            print(('\n#compute section\ncompute  com1 cel com\nfix com11 cel ave/time 1000 1 1000 c_com1[*] file ./results/%d-cel.dat\ncompute  com2 lay1 com\nfix com12 lay1 ave/time 1000 1 1000 c_com2[*] file ./results/%d-lay1.dat\ncompute  com3 pol com\nfix com13 pol ave/time 1000 1 1000 c_com3[*] file ./results/%d-pol.dat\n')%(k,k,k))
            if ringcoms=='TRUE': 
                print('\ncompute rgpol pol gyration\ncompute msd1 pol msd\n\nvariable rgp1 equal c_rgpol\n')
                print(('variable st equal step\n\nvariable etoe equal "sqrt((xcm(e1,x)-xcm(e2,x))^2 + (xcm(e1,y)-xcm(e2,y))^2 + (xcm(e1,z)-xcm(e2,z))^2 )"\nvariable oetoe equal 1/v_etoe\n\nvariable as equal "(xcm(e2,x) - xcm(e1,x))*v_oetoe"\nvariable asy equal "(xcm(e2,y) - xcm(e1,y))*v_oetoe"\n\nvariable asz equal "(xcm(e2,z) - xcm(e1,z))*v_oetoe"\n\nvariable k1 equal "sqrt((xcm(r1,x)-xcm(r2,x))^2 + (xcm(r1,y)-xcm(r2,y))^2 + (xcm(r1,z)-xcm(r2,z))^2 )"\nvariable k2 equal "sqrt((xcm(r2,x)-xcm(r3,x))^2 + (xcm(r2,y)-xcm(r3,y))^2 + (xcm(r2,z)-xcm(r3,z))^2 )"\nvariable k3 equal "sqrt((xcm(r3,x)-xcm(r4,x))^2 + (xcm(r3,y)-xcm(r4,y))^2 + (xcm(r3,z)-xcm(r4,z))^2 )"\nvariable k4 equal "sqrt((xcm(r4,x)-xcm(r5,x))^2 + (xcm(r4,y)-xcm(r5,y))^2 + (xcm(r4,z)-xcm(r5,z))^2 )"\n\nvariable angx equal angmom(pol,x)\nvariable angy equal angmom(pol,y)\nvariable angz equal angmom(pol,z)\n\nfix etoes all print 500 "${st} ${as} ${asy} ${asz} ${etoe}" file ./results/%d-etoes.dat title "#etoe"\nfix rogpol all print 500 "${st} ${rgp1}" file ./results/%d-rogpol.dat title "#Rg-pol"\nfix angmompol all print 500 "${st} ${angx} ${angy} ${angz}" file ./results/%d-angmompol.dat title "#angmom_x angmom_y angmom_z"\nfix msdpol pol ave/time 500 1 500 c_msd1[*] file ./results/%d-msdpol.dat\nfix kuhns all print 500 "${st} ${k1} ${k2} ${k3} ${k4}" file ./results/%d-kuhns.dat title "#k1 k2 k3 k4"')%(k,k,k,k,k))
            print(('\n%s')%recen)
            print('\nvariable d equal c_com3[1]-c_com2[1]')
            print('\n#output section\nrestart		1000  ../dz/restarts/crystal.restart1 ../dz/restarts/crystal.restart2\nrestart 50000 ../dz/restarts/ps.restart')
            if traj=='TRUE':print('\ndump    dump3 all custom 500 ps1.lammpstrj id type xu yu zu vx vy vz ix iy iz\ndump dcd1 all dcd 500 ps1.dcd')
            else:print('\n#no traj for this run#\n')
            print(('\n\nwrite_data	npt.data\n\nthermo          500\nthermo_style  	custom step press temp vol density  pe ke etotal evdwl ecoul elong epair ebond eangle edihed  emol   c_com3[1] c_com2[1] v_d \nthermo_modify flush yes\n\nrun %d upto')%st_step[k])
            
            
            sys.stdout.close();
        sys.stdout=oldstdout;
        

        
        ########################## copying files #############################

        for jj in range(wstart, wstop):
            win=jj
            os.mkdir(str(win))
            scr1="%s/%d.inp"%(path,jj)
            scr2="%s/ini%d.inp"%(path,jj)
            scr3="%s/%d.in"%(path,jj)
            scr4="%s/%d.sh"%(path,jj) 
            scr5="%s/%d.data"%(path,jj)
            scr6="%s/%drestart.in"%(path,jj)
            #scr7="%s/%drerun.in"%(path,jj)
            dst="%s/%d/"%(path,jj)

            if justsh == "inp":
                shutil.move(scr1,dst) #shell script
            if justsh == "iinp":
                shutil.move(scr2,dst) #shell script
            if justsh == "in":
                shutil.move(scr3,dst) #shell script
            if justsh == "sh":
                shutil.move(scr4,dst) #shell script
            if justsh == "data":
                shutil.move(scr5,dst) #shell script
            if justsh == "restart":
                shutil.move(scr6,dst) #shell script
            
            if justsh == "n":
                shutil.move(scr1,dst) #colvar input
                shutil.move(scr2,dst) #initial colvar input
                shutil.move(scr3,dst) #lammps input
                shutil.move(scr4,dst) #shell script
                shutil.move(scr6,dst) #lammps restart input
                #shutil.move(scr7,dst) #lammps rerun input
                
            elif justsh == "nd":
                shutil.move(scr1,dst) #colvar input
                shutil.move(scr2,dst) #initial colvar input
                shutil.move(scr3,dst) #lammps input
                shutil.move(scr4,dst) #shell script
                shutil.move(scr5,dst) #data file
                shutil.move(scr6,dst) #lammps restart input
                #shutil.move(scr7,dst) #lammps rerun input


        ########################### prep file for job submission ###############
                
        sys.stdout= open('sub.sh','a')
        for i in range(wstart, wstop):
            print(("cd %s/%d/;\ndos2unix %d.sh;\nsbatch %d.sh;")%(subloc,i,i,i)) 
        sys.stdout.close();
        sys.stdout=oldstdout
        
        ########################### spliting file #############################
        
        sys.stdout= open('split.sh','a')
        print('source ~/ENV/p37/bin/activate')
        for i in range(wstart, wstop):
            print('python /home/vasudevn/datafiles/tsplitsingle.py {} {}/{}/dz;'.format(i, subloc, i))
        sys.stdout.close();
        sys.stdout=oldstdout
        
        ##########################  wham calculation file #####################
        
        
        sys.stdout = open('list.dat','w')
        for i in range(wstart, wstop):
            print('{}{}.colvars.split.traj {} {} {}'.format(folder, i, middle[i], K[i], 10))
        sys.stdout.close();
        sys.stdout=oldstdout
        
        ########################### cp all traj once done #####################
        
        sys.stdout = open('cp.sh','w')
        for i in range(wstart, wstop):
            print('cp {}/{}/dz/{}{}.colvars.split.traj -t {};'.format(subloc, i, folder, i,subloc))
        sys.stdout.close();
        sys.stdout=oldstdout
        
        #######################################################################
        
        json_dump(inputs,'lammps_input_json')
        et=time.time()
        print(('\n\ntotal execution time:'), end=' ')
        print(round((et-st),6), end=' ')
        print('seconds\n')
        
        for index, t in enumerate(TIME_min):
            if K[index] > 100.0:
                print('window - {} = {} mins'.format(index, t+60))
            else:
                print('window - {} = {} mins'.format(index, t))
        print('\n')
        
    os.chdir('/home/naveen/Desktop/P3/pmftest')
    return True

loc = '/media/sf_dive/Research_work/afinalpaperrun/analysis/OPT/test/algorithm/pmf-calculation'
os.chdir(loc)

def auto(l,name):
    
    distplot = False
    loc2 = l+'/{}'.format(name)
    
    wins = len(glob.glob1(loc2,"*.traj"))
    
    os.chdir(loc2)
    
    mywhamer(wins, '{}.dat'.format(name), loc2, 50)
    removeinf_and_gradient(loc2,'{}.dat'.format(name))
    
    loc = l+'/'
    shutil.copy('{}.dat'.format(name),loc)
    shutil.copy('gradient-{}.dat'.format(name),loc)
    os.chdir(loc)
    
    pmfplot(loc, 'b.dat', '{}.dat'.format(name), name)
    pmfer(loc, '{}.dat'.format(name))
    
    pmfarea(loc, 'b.dat', '{}.dat'.format(name), 'area_pmf_{}'.format(name))
    pmfarea(loc, 'gradient-b.dat', 'gradient-{}.dat'.format(name), 'area_grad_{}'.format(name))
    
    os.chdir(loc2)
    
    if distplot:
        distribution(loc2, wins)
        OVLap (loc2, wins, name)
    
    
def mywhamer(wins,freefile,loc,MCnum):

    os.chdir(loc)
    
    metadatafile        = 'list.dat'
    hist_min            = 2.0#input('hist_min:')
    hist_max            = 14.5#input('hist_max:')
    num_bins            = 100#input('bins:')
    tol                 = 0.0001#input('tol:')
    temp                = 300.0#float(input('temperature:'))
    numpad              = 0#input('numpad:')
    numMCtrials         = MCnum #50#input('num_MC_trials:')#
    randSeed            = random.randint(9999,10000000)
    
    # metadatafile        = 'list.dat'
    # hist_min            = 2.0#2.0#input('hist_min:')#
    # hist_max            = 3.5#3.5#input('hist_max:')#
    # num_bins            = 10#12#input('bins:')#
    # tol                 = 0.0001#input('tol:')#
    # temp                = 300.0#float(input('coord:'))#
    # numpad              = 0#input('numpad:')#
    # numMCtrials         = MCnum #50#input('num_MC_trials:')#
    # randSeed            = random.randint(9999,10000000)

    print('Using: wham %.4f %.4f %d %.6f %.4f %d %s %s %d %d\n'%(hist_min, hist_max, num_bins, tol, temp, numpad, metadatafile, freefile, numMCtrials, randSeed))
    os.system('wham %.4f %.4f %d %.6f %.4f %d %s %s %d %d'%(hist_min, hist_max, num_bins, tol, temp, numpad, metadatafile, freefile, numMCtrials, randSeed))



def Ax(prob):
    pr = -constants.Boltzmann * 300.0 * np.log(prob)
    #print(pr)
    return pr

def Ux(x0, x, k):
    u = -0.5 * k * (x0 - x) **2
    #print(u)
    return u

def mywhamer2(loc,):
    
    wins = input('window number:\t')
    loc = '/media/sf_dive/Research_work/afinalpaperrun/analysis/OPT/test/algorithm/pmf-calculation/5/impact'
    MCnum = 50
    os.chdir(loc)
    
    
    if wins == '1':
        #2.8753 2.92235 2.9694 748.8648 
        hist_min            = 2.82090855
        hist_max            = 3.04956445
        k                   = 748.8648
        x0                  = 2.92235
        metadatafile        = 'list1.dat'
        freefile            = 'impact-{}.dat'.format(wins)
    
    elif wins == '25':      
        #12.0687 12.5970 13.1252 5.9489 
        hist_min            = 10.97862675
        hist_max            = 13.95527625
        k                   = 5.9489 
        x0                  = 12.5970
        metadatafile        = 'list25.dat'
        freefile            = 'impact-{}.dat'.format(wins)
    
    
        
    num_bins            = 100#int(input('bins:'))
    tol                 = 0.0001#input('tol:')
    temp                = 300.0#float(input('temperature:'))
    numpad              = 0#input('numpad:')
    numMCtrials         = MCnum #50#input('num_MC_trials:')#
    randSeed            = random.randint(9999,10000000)
    
    
    
    print('Using: wham %.4f %.4f %d %.6f %.4f %d %s %s %d %d\n'%(hist_min, hist_max, num_bins, tol, temp, numpad, metadatafile, freefile, numMCtrials, randSeed))
    os.system('wham %.4f %.4f %d %.6f %.4f %d %s %s %d %d'%(hist_min, hist_max, num_bins, tol, temp, numpad, metadatafile, freefile, numMCtrials, randSeed))
    
    data = sorted(np.loadtxt('100mc{}.colvars.split.traj'.format(wins))[:,1])
    # Fit a normal distribution to the data:
    mu, std = norm.fit(data)
    #data = [ (x - mu) / std for x in data ]
    # Plot the histogram.
    plt.hist(data, bins=100, density=True, alpha=0.6, color='g')
    
    # Plot the PDF.
    xmin, xmax = plt.xlim()
    print(xmin,xmax)
    imp = np.loadtxt('impact-{}.dat'.format(wins))
    x = imp[:,0] #np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2)
    title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
    plt.title(title)
    
    plt.show()
    
    pdash = []
    for prob in p:
        pdash.append(Ax(prob))
    
    udash = []
    
    for xi in x:
        udash.append(Ux(x0, xi, k))
    
    np.savetxt('{}-eqn.dat'.format(wins),np.c_[x,imp[:,1],pdash,udash])
    
    plt.plot(x,imp[:,1], label = 'A(x)', c = 'g')
    plt.xlabel('x')
    plt.ylabel('A(x)')
    plt.legend()
    plt.show()
    
    plt.plot(x,pdash, label = "-kT ln P'(x)", c = 'r')
    plt.xlabel('x')
    plt.ylabel("-kT ln P'(x)")
    plt.legend()
    plt.show()
    
    plt.plot(x,udash, label = "-U'(x)", c = 'b')
    plt.xlabel('x')
    plt.ylabel("-U'(x)")
    plt.legend()
    plt.show()
    
    plus = np.array(udash) + np.array(pdash)
    plt.plot(x,plus, label = "-kT ln P'(x)-U'(x)", c = 'b')
    plt.xlabel('x')
    plt.ylabel("-kT ln P'(x)-U'(x)")
    plt.legend()
    plt.show()
    
    #return x,pdash,udash,plus
    



def removeinf_and_gradient(loc,freefile):
    os.chdir(loc)
    f = np.loadtxt(loc+'/'+freefile)
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
    plt.plot(gradient[:,0],gradient[:,1])
    np.savetxt('gradient-{}'.format(freefile),gradient,fmt='%.4f')
    plt.show()
    plt.close()


def pmfplot(loc, free1, free2, pdfname):
    n1=free1
    n2=free2
    os.chdir(loc)
    free1, free2 = np.loadtxt(free1), np.loadtxt(free2)
    f1, f2, e1, e2 = free1[:,0:2], free2[:,0:2], free1[:,2], free2[:,2]
    plt.errorbar(f1[::,0], f1[::,1],yerr=e1,lw=1.0,capsize=1,errorevery=5,markeredgewidth=0.5,markersize='3',elinewidth=0.5,label=n1.split('.')[0])
    plt.errorbar(f2[::,0], f2[::,1],yerr=e2,lw=1.0,capsize=1,errorevery=5,markeredgewidth=0.5,markersize='3',elinewidth=0.5,label=n2.split('.')[0])
    plt.title(r'$\xi$ vs PMF')
    plt.xlabel(r'$\xi(\AA)$ - distance between COMs',fontsize=14,weight='bold')
    plt.ylabel('PMF (kcal/mol)',fontsize=14,weight='bold')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig('%s.pdf'%pdfname, bbox_inches='tight')
    plt.show()
    plt.close()
    
def pmfer(loc, free1):
    os.chdir(loc)
    free1 = np.loadtxt(free1)
    f1, e1 = free1[:,0:2],  free1[:,2]
    plt.errorbar(f1[::,0], f1[::,1],yerr=e1,capsize=1,errorevery=5,markeredgewidth=0.5,markersize='3',elinewidth=0.5,linewidth=1.0,label='benchmark')
    plt.title(r'$\xi$ vs PMF')
    plt.xlabel(r'$\xi(\AA)$ - distance between COMs',fontsize=14,weight='bold')
    plt.ylabel('PMF (kcal/mol)',fontsize=14,weight='bold')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.show()
    plt.close()
 
def pmfarea(loc, gfile, gfile2, savefile) :

   os.chdir(loc)
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
   
   print("\nFor %s curve:\n\tThe area between the curves is:\t%f between (%.4f, %.4f)\n\t\tAbove : %f\n\t\tBelow : %f\n\tThe delta pmf value is:\t%f\n\tThe dist-norm area is:\t%f"%(c2, area, curve1[0,0], curve1[-1,0], areaa, areab, deltapmf, newarea))
   plt.plot(x_y_curve1[:,0],x_y_curve1[:,1],c='b',lw=1,marker='+',markevery=5,markersize=5,label="%s"%c1)
   plt.plot(x_y_curve2[:,0],x_y_curve2[:,1],c='k',lw=1,marker='^',markevery=5,markersize=5,label="%s"%c2)      
   plt.fill_between(x_y_curve1[:,0], x_y_curve1[:,1],x_y_curve2[:,1],color='r',label='area-diff = {:.4f}'.format(area))
   
   plt.title(r'%s-%s - $\xi$ vs PMF'%(c1,c2))
   plt.xlabel(r'$\xi(\AA)$ - distance between COMs',fontsize=14,weight='bold')
   plt.ylabel(r'$\frac{\Delta PMF (kcal/mol)} {\Delta \xi (\AA)}$',fontsize=14,weight='bold')
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
    json = j.dumps(d)
    f = open("%s.json"%filename,"w")
    f.write(json)
    f.close()

def pickle_dump(d, filename):
    with open(filename+'.p','wb') as f:
        p.dump(d,f)

def pickle_load(filename):
    with open(filename,'rb') as f:
        dic = p.load(f)
    return dic


# loc = '/media/sf_dive/Research_work/afinalpaperrun/analysis/OPT/test/algorithm/pmf-calculation/'
# os.chdir(loc)

# ps = ['area_pmf_1','area_pmf_2','area_pmf_3','area_grad_1','area_grad_2','area_grad_3','area_pmf_ref','area_grad_ref']

# # # ps = ['pmf_s1p1','pmf_s1p2','pmf_s1p3','pmf_ref','pmf_s1p3a', 'grad_s1p1','grad_s1p2','grad_s1p3','grad_ref','grad_s1p3a']

# area        = {'%s'%f      :   pickle_load('%s.p'%f)['area']            for f in ps}
# normarea    = {'%s'%g      :   pickle_load('%s.p'%g)['norm_area']       for g in ps}



def distribution(loc, wins):
    
    os.chdir(loc)
    for i in range(wins):
        try:
            x = sorted(np.genfromtxt(loc+'/100mc%d.colvars.traj'%i)[:,1])#[start*1000:stop*1000,1])
        except:
            x = sorted(np.genfromtxt(loc+'/100mc%d.colvars.split.traj'%i)[:,1])#[start*1000:stop*1000,1])
        sns.distplot(x,hist=False)
        plt.xlabel(r'$\xi$')
        plt.ylabel('pdf')      
        plt.savefig('distribution.pdf',bbox_inches='tight')
    plt.show()

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
    
    fig, ax = plt.subplots()
    sns.distplot(hist1,label='win %d'%i,color='r')
    sns.distplot(hist2,label='win %d'%j,color='b')
    plt.legend()
    plt.show()
    
    output = 1.0 - (fabs(cdf(y,x1) - cdf(x,x1)) + fabs(cdf(y,x2) - cdf(x,x2)))
    return round(output,4)


def OVLap (loc, wins, name):
    os.chdir(loc)

    OVL = np.zeros((wins,wins))
    
    for i in range(0,wins):
        try:
            d1=np.genfromtxt('100mc%d.colvars.traj'%i)
        except:
            d1=np.genfromtxt('100mc%d.colvars.split.traj'%i) 
        print('\n')
        for j in range(i,wins):
            if i == j:
                OVL[i][j]=1.0
                continue
            try:
                d2=np.genfromtxt('100mc%d.colvars.traj'%j)
            except:
                d2=np.genfromtxt('100mc%d.colvars.split.traj'%j)

            hist1=np.array(sorted(d1[::,1]))    
            fit1 = stats.norm.pdf(hist1, np.mean(hist1), np.std(hist1))
    
            hist2=np.array(sorted(d2[::,1]))
            fit2= stats.norm.pdf(hist2, np.mean(hist2), np.std(hist2))
            
            # fig, ax = plt.subplots()
            # ax = sns.distplot(hist1,label='window %d'%i)
            # ax = sns.distplot(hist2,label='window %d'%j)
    
            # plt.legend()
            
            # plt.show()
            
            output = overlap(hist1,hist2,i,j)
            #output2 = overlap(hist2,hist1,i,j)
            
            OVL[j][i] = output
            OVL[i][j] = output
            if output == 0.00000: 
                break
                
            
            print('\n\tOverlap coefficient between Window %d and %d is: %f'%(i,j,output))
            #print('Overlap coefficient between Window %d and %d is: %f'%(j,i,output2))
    
    
    np.savetxt('OVL-%s.dat'%name,OVL,fmt='%.6f')
    
    OVL2=np.loadtxt('OVL-%s.dat'%name)
    plt.imshow(OVL2, cmap='plasma', interpolation='nearest')
    plt.title('OVL - %s'%name)
    plt.xlim(0,wins-0.5)
    plt.ylim(0,wins-0.5)
    plt.colorbar()
    plt.clim(0,1)
    plt.xticks(range(wins), rotation='vertical')
    plt.yticks(range(wins), rotation='horizontal')
    plt.savefig(loc+'/OVL-%s.pdf'%name,bbox_inches='tight')
    images = convert_from_path('OVL-%s.pdf'%name,dpi=300)
    images[0].save('OVL-%s.jpg'%name)
    plt.show()

#loc = '/media/sf_dive/Research_work/afinalpaperrun/analysis/OPT/test/algorithm/pmf-calculation/benchmark/cut'

def trajcut(s,per,loc,fol):
    ll = loc
    loc = loc+'/{}'.format(fol)
    slfol = '/{}cut{}ns'.format(fol,per*(8)/100.0)
    if os.path.isdir(loc+slfol):
        os.chdir(loc+slfol)
    else:
        os.mkdir(loc+slfol)
        os.chdir(loc+slfol)
    wins = len(glob.glob1(loc,"*.traj"))
    for i in range(0,wins):
        try:    
            f=loc+'/100mc%d.colvars.split.traj'%(i)
            data=np.genfromtxt(f)
        except:    
            f=loc+'/100mc%d.colvars.traj'%(i)
            data=np.genfromtxt(f)
        start = s*1000
        stop = start+int(len(data)*per/100)
        datanew=data[start:stop]
        newfile='100mc%d.colvars.split.traj'%(i)
        np.savetxt(newfile,datanew, fmt='%d %.14e %.1f %d')
    shutil.copy('../list.dat','./')
    print(slfol)
    return slfol

# for f in range(1,10):
#     print(f)
#     for pe in [25,50,75]:
#         #slfol = trajcut(0,pe,loc,str(f))
#         slfol = '/{}cut{}ns'.format(f,pe*(8)/100.0)
#         print(slfol.strip('/'))
#         auto(loc, slfol.strip('/'))
        

def winvariance(loc, name):
    
    os.chdir(os.path.join(loc,name))
    wins = len(glob.glob('./*.traj'))
    bvnloc = os.path.join('/media/sf_dive/Research_work/afinalpaperrun/analysis/OPT/test/algorithm/bvn',name)
    txtname = np.loadtxt(glob.glob(bvnloc+'/*.txt')[0])
    left, right = txtname[:,0], txtname[:,2]
    wid = right - left
    variance = []
    for w in range(wins):
        
        data = sorted(np.loadtxt('100mc{}.colvars.split.traj'.format(w))[:,1])
        var = np.var(data)*2
        variance.append(var)
    variance = np.array(variance)
    pdata = np.c_[wid, variance]
    pdata = pdata[np.argsort(pdata[:,0])]
    
    fig, ax = plt.subplots()
    ax.plot(pdata[:,0],pdata[:,1],label = name)
    # lims = [
    # np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
    # np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    # ]
    # ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    
    ax.set_title('point {}'.format(name))
    ax.set_xlabel('window width')
    ax.set_ylabel('distribution width')
    #plt.savefig('{}-winvariance.pdf'.format(name), bbox_inches='tight')
    #plt.show()

# for i in range(1,8):
#     name=  str(i)
    
#     os.chdir(os.path.join(loc,name))
#     wins = len(glob.glob('./*.traj'))
#     bvnloc = os.path.join('/media/sf_dive/Research_work/afinalpaperrun/analysis/OPT/test/algorithm/bvn',name)
#     txtname = np.loadtxt(glob.glob(bvnloc+'/*.txt')[0])
#     left, right = txtname[:,0], txtname[:,2]
#     wid = right - left
#     variance = []
#     for w in range(wins):
        
#         data = sorted(np.loadtxt('100mc{}.colvars.split.traj'.format(w))[:,1])
#         var = np.var(data)*2
#         variance.append(var)
#     variance = np.array(variance)
#     pdata = np.c_[wid, variance]
#     pdata = pdata[np.argsort(pdata[:,0])]
    
#     #fig, ax = plt.subplots()
#     plt.plot(pdata[:,0],pdata[:,1],label = name)
#     # lims = [
#     # np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
#     # np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
#     # ]
#     # ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    
# plt.title('win width vs dist width'.format(name))
# plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# plt.xlabel('window width')
# plt.ylabel('distribution width')
# plt.savefig('winvariance.pdf',bbox_inches='tight')


    

# loc = '/media/sf_dive/Research_work/afinalpaperrun/analysis/OPT/test/algorithm/pmf-calculation/benchmark/cut'
# os.chdir(loc)

# def pplot(loc, free1, free2, pdfname):
#     os.chdir(loc)
#     free1, free2 = np.loadtxt(free1), np.loadtxt(free2)
#     f1, f2, e1, e2 = free1[:,0:2], free2[:,0:2], free1[:,2], free2[:,2]
#     plt.errorbar(f1[::,0], f1[::,1],yerr=e1,lw=0.5,capsize=1,markeredgewidth=0.5,markersize='3',elinewidth=0.5,label='benchmark')
#     plt.errorbar(f2[::,0], f2[::,1],yerr=e2,lw=0.5,capsize=1,markeredgewidth=0.5,markersize='3',elinewidth=0.5,label='test')
#     plt.title(pdfname)
#     plt.xlabel(r'$\xi(\AA)$ - distance between COMs',fontsize=14,weight='bold')
#     plt.ylabel('PMF (kcal/mol)',fontsize=14,weight='bold')
#     plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#     plt.xticks(fontsize=14)
#     plt.yticks(fontsize=14)
#     plt.savefig('%s.pdf'%pdfname, bbox_inches='tight')
#     plt.show()
#     plt.close()
    
# for start in range(0,39):
#     stop = start+2
    
#     trajcut(start,2,loc)
#     mywhamer(24, '{}-{}.dat'.format(start,stop), loc+'/sliced', 50)
#     pplot(loc+'/sliced', 'b.dat', '{}-{}.dat'.format(start,stop), '{}-{}'.format(start,stop))
    

# wins = 24
# gloc = '/project/6003277/vasudevn/backup/graham/afinalpaperrun/pmf_run/100mc/pmf_final/benchmark/16'
# print('module load nixpkgs/16.09  intel/2018.3  openmpi/3.1.2\nmodule load lammps-omp/20190807')
# for w in range(wins):
#     print('lmp_icc_openmpi -r2data {}/{}/restarts/ps.restart.20000000 {}.data;'.format(gloc,w,w))
#     #print('mv {}/{}/restarts/{}.data ../../datafiles;'.format(gloc,w,w))
    
#################### cut trajectories to a certain percentage and then calculate the rest ##################
    
# def trajpercut(per,loc):
   
#     os.chdir(loc)
#     wins = len(glob.glob1(loc,"*.traj"))
#     for i in range(0,wins):
            
#         try:    
#             f='./100mc%d.colvars.traj'%(i)
#             data=np.genfromtxt(f)
#         except: 
#             f='./100mc%d.colvars.split.traj'%(i)
#             data=np.genfromtxt(f)*
#         length = len(data)
#         datanew=data[0:int(length*per)]
#         newfile='100mc%d.colvars.split.traj'%(i)
#         np.savetxt(newfile,datanew, fmt='%d %.14e %.1f %d')
#     return wins

# for s in range(1,12):
#     loc = '/media/sf_dive/Research_work/afinalpaperrun/analysis/OPT/test/algorithm/pmf-calculation'
#     w = trajpercut(1.0, loc+'/{}'.format(s))
#     auto(loc,'{}'.format(s),w)
    



# array = np.empty((11,3))

# for c,f in enumerate([50,75,100]):
#     os.chdir('/media/sf_dive/Research_work/afinalpaperrun/analysis/OPT/test/algorithm/pmf-calculation/{}-percentage'.format(f))
#     for s in range(1,12):
#         feed = pickle_load('area_pmf_{}.p'.format(s))
#         #print(feed)
#         array[s-1, c] = feed['area']
        
# index = range(1,12)        
# df = pd.DataFrame(data=array, index=index, columns=['50%-1ns','75%-1.5ns','100%-2ns'])
# print('\n\n',df)
# df.plot.bar(rot=0)
# plt.xlabel('evaluated points')
# plt.ylabel('area difference or fitness')
# plt.savefig('percentage-barplot.pdf',bbox_inches='tight')
