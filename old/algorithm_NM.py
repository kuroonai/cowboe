#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 12:29:22 2020

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


'''
Simplex Nelder-Mead Optimization (Amoeba Search)
Pseudocode


initialize search area (simplex) using random starting parameter values
while not done loop
    compute centroid
    compute reflected
    
    if reflected is better than best solution then
        compute expanded
        replace worst solution with better of reflected and expanded
        
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
            shirnk the search area

    else
        replace worst solution with reflected
    end if
    if the solution is within tolerance, exit loop
end loop
return best solution found

'''    

'''

initialize search area (simplex) using random starting parameter values
while not done loop
    compute centroid
    compute reflected
    if reflected is better than best solution then
        compute expanded
        replace worst solution with better of reflected and expanded
    else if reflected is worse than all but worst then
        if reflected is better than worst solution then
            replace worst solution with reflected
        end if
        compute contracted
        if contracted is worse than worst
            shrink the search area
        else
            replace worst solution with contracted
        end if
    else
        replace worst solution with reflected
    end if
    if the solution is within tolerance, exit loop
end loop
return best solution found

'''    

import numpy as np
from matplotlib import pyplot as plt
import os
import inspect

loc = '/media/sf_dive/Research_work/afinalpaperrun/analysis/OPT/test/algorithm/pmf-calculation'
os.chdir(loc)


#8ns / A runs

# Iteration - 1 - starting points 1,2,3
logA        = {1:np.log(2.0000)    ,    2:np.log(2.9000)   ,  3:np.log(3.5000) } 
V           = {1:0.75            ,    2:0.8700           ,  3:0.8000}
area_old = {'area_pmf_1': 21.2646,  'area_pmf_2': 17.6376,  'area_pmf_3': 24.5880  }


# Iteration - 2 - starting points  
logA        = {1:np.log(2.0000)    ,    2:np.log(2.9000)   ,  3:np.log(1.6571)} 
V           = {1:0.75            ,    2:0.8700           ,  3:0.8200}
area_old = {'area_pmf_1': 21.2646,  'area_pmf_2': 17.6376,  'area_pmf_3': 7.8664  }

# Iteration - 3 - starting points 
logA        = {1:np.log(2.4028)    ,    2:np.log(2.9000)   ,  3:np.log(1.6571)} 
V           = {1:0.9400            ,    2:0.8700           ,  3:0.8200}
area_old = {'area_pmf_1': 4.9780 ,  'area_pmf_2': 17.6376,  'area_pmf_3': 7.8664  }



# # Iteration - 3 - New starting points  - Replace wor with ref and compute conc (outer)
# logA        = {1:np.log(2.4028)    ,    2:np.log(1.3729)   ,  3:np.log(1.6571)} 
# V           = {1:0.9400            ,    2:0.8899           ,  3:0.8200}
# area_old = {'area_pmf_1': 4.9780 ,  'area_pmf_2': 8.4804,  'area_pmf_3': 7.8664  }

# # Iteration - 4 - New starting point
# logA        = {1:np.log(2.4028)    ,    2:np.log(2.4056)   ,  3:np.log(1.6571)} 
# V           = {1:0.9400            ,    2:0.8750           ,  3:0.8200}
# area_old = {'area_pmf_1': 4.9780 ,  'area_pmf_2': 5.9778,  'area_pmf_3': 7.8664  }

# # Iteration - 5 - New starting point
# logA        = {1:np.log(2.4028)    ,    2:np.log(2.4056)   ,  3:np.log(3.4881)} 
# V           = {1:0.9400            ,    2:0.8750           ,  3:0.9950}
# area_old = {'area_pmf_1': 4.9780 ,  'area_pmf_2': 5.9778,  'area_pmf_3': 11.9382  }

# # Iteration - 6 - New starting point
# logA        = {1:np.log(2.4028)    ,    2:np.log(2.4056)   ,  3:np.log(1.6571)} 
# V           = {1:0.9400            ,    2:0.8750           ,  3:0.8200}
# area_old = {'area_pmf_1': 4.9780 ,  'area_pmf_2': 5.9778,  'area_pmf_3': 7.8664  }

# # Iteration - 7 - New starting point
# logA        = {1:np.log(2.4028)    ,    2:np.log(2.4041)   ,  3:np.log(1.9954)} 
# V           = {1:0.9400            ,    2:0.9075           ,  3:0.8799}
# area_old = {'area_pmf_1': 4.9780 ,  'area_pmf_2': 6.2411,  'area_pmf_3': 19.9722  }


area        =   list(area_old.values())
F = {1:area[0],      2:area[1],      3:area[2]} #area-pmf

fitfunc = {}

def Kgiven(v):
    return v*2/0.5**2

def Vgiven(k):
    return 0.5*k*0.5**2

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
        
        # if l+1 == best: 
        #     label = 'best-{}-({:.4f},{:.4f})'.format(l+1,np.exp(x),y)
        #     fontc = 'g'
        # elif l+1 ==worst : 
        #     label = 'worst-{}-({:.4f},{:.4f})'.format(l+1,np.exp(x),y)
        #     fontc = 'r'

        # else : 
        #     label = 'other-{}-({:.4f},{:.4f})'.format(l+1,np.exp(x),y)
        #     fontc = 'grey'
        
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

def Expansion(cen, ref, best, worst, other, gamma):

    [xr,yr] = ref
    [xc, yc] = cen
    
    [xe,ye] = [xc + gamma*(xr - xc), yc + gamma*(yr - yc)]
    
    step_plot(best, worst, other)
    x,y = xe,ye
    
    exp = [x,y]
    
    plt.plot(xe,ye, 'g*',markersize='10')
    
    wor = [logA[worst], V[worst]]
    plt.plot([exp[0],wor[0]], [exp[1],wor[1]], 'k--')

    
    plt.annotate('Expanded', # this is the text
             (x,y), # this is the point to label
             textcoords="offset points", # how to position the text
             xytext=(0,10), # distance from text to points (x,y)
             ha='center',bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3))
    
    plt.xlim((min(bes[0],wor[0],oth[0],exp[0])-0.1 , max(bes[0],wor[0],oth[0],exp[0])+0.1))
    plt.ylim((min(bes[1],wor[1],oth[1],exp[1])-0.1 , max(bes[1],wor[1],oth[1],exp[1])+0.1))

    plt.savefig('expanded.pdf',bbox_inches='tight')
    plt.show()
    
    return exp

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

def A_convertor(ref, exp, in_con, out_con, s_wor, s_oth):
    
    c_ref       = convertor(ref)
    c_exp       = convertor(exp)
    c_in_con    = convertor(in_con)
    c_out_con   = convertor(out_con)
    c_s_wor     = convertor(s_wor)
    c_s_oth     = convertor(s_oth)
    
    return {
        'ref':c_ref, 
        'exp':c_exp, 
        'incon':c_in_con, 
        'outcon':c_out_con, 
        'swor':c_s_wor, 
        'soth':c_s_oth
        }
    

alpha = 1
gamma = 2
beta = 0.5
delta = 0.5

cen, ref, best, worst, other, bes, wor, oth = Reflection(logA, V, F, alpha)
exp = Expansion(cen, ref, best, worst, other, gamma)
in_con = inner_contraction(cen, wor, best, worst, other, beta)
out_con = outer_contraction(ref, cen, wor, best, worst, other, beta )
s_wor, s_oth = Shrink(wor, oth, best, worst, other, delta)

conv = A_convertor(ref, exp, in_con, out_con, s_wor, s_oth)

print(conv)

'''

initialize search area (simplex) using random starting parameter values
while not done loop
    compute centroid
    compute reflected
    if reflected is better than best solution then
        compute expanded
        replace worst solution with better of reflected and expanded
    else if reflected is worse than all but worst then
        if reflected is better than worst solution then
            replace worst solution with reflected
        end if
        compute contracted
        if contracted is worse than worst
            shrink the search area
        else
            replace worst solution with contracted
        end if
    else
        replace worst solution with reflected
    end if
    if the solution is within tolerance, exit loop
end loop
return best solution found

'''    

# psuedocode steps

# cen, ref, best, worst, other, bes, wor, oth = Reflection(logA, V, F, alpha)
# fitfunc.update({'ref':float(input('Enter fitness function value of Reflected: '))})

# exp = Expansion(cen, ref, best, worst, other, gamma)
# fitfunc.update({'exp':float(input('Enter fitness function value of Expanded: '))})

# if fitfunc['ref'] < fitfunc['best']:
#     exp = Expansion(cen, ref, best, worst, other, gamma)
#     fitfunc.update({'exp':float(input('Enter fitness function value of Expanded: '))})
#     if fitfunc['ref'] > fitfunc['exp']: wor = exp
#     else: wor = ref
    

# if max(fitfunc['best'], fitfunc['other']) < fitfunc['ref'] < fitfunc['worst']:
#     if fitfunc['ref'] < fitfunc['worst']:
#         wor = ref
        
#     in_con = inner_contraction(cen, wor, best, worst, other, beta)
#     out_con = outer_contraction(ref, cen, wor, best, worst, other, beta )
#     fitfunc.update({'in_con':float(input('Enter fitness function value of innter contraced: '))})
#     fitfunc.update({'out_con':float(input('Enter fitness function value of outer contraced: '))})
#     fitfunc.update({'con':min(fitfunc['in_con'], fitfunc['out_con'])})
#     if fitfunc['in_con'] > fitfunc['out_con']: con = out_con
#     else: con = in_con
    
#     if fitfunc['con'] > fitfunc['worst']:
#         s_wor, s_oth = Shrink(wor, oth, best, worst, other, delta)
#         wor, oth = s_wor, s_oth
        
#     else:
#         wor = con

# else:
#     wor = ref


#2 


    