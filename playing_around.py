#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 13:41:12 2022

@author: chris
"""

from model.interface import load_model, run_model, save_model
from model.plotting.plotting import *

#o = run_model('mbh', 'quasar')

#o = run_model('mbh', 'quasar', fitting_method='mcmc', num_walker=10, min_chain_length=0, parallel = True,
#              prior_name='successive')

#m = run_model('mstar','changing', fitting_method='mcmc',num_walker=10, autocorr_chain_multiple=1,
#              redshift=[4,5], tolerance = 0.01, parallel=True, min_chain_length=20000, parameter_calc=False,
#              autocorr_discard=True)
#save_model('mstar', 'changing', num_walker=50)

#mstar = load_model('mstar','changing')
#muv   = load_model('Muv','changing')
#lbol   = load_model('Lbol','quasar',prior_name='successive')
#mbh    = load_model('mbh','quasar',prior_name='successive')

#Plot_best_fit_ndfs(o)


# abundance matching
import numpy as np
import matplotlib.pyplot as plt
from model.analysis.calculations import calculate_best_fit_ndf

lbol   = load_model('Lbol','quasar',prior_name='successive')
mbh    = load_model('mbh','quasar',prior_name='successive')

## BLACK HOLE MASS - LAMBDA RELATION (ABBUNDANCE MATCHING)
# lfs = lbol.log_ndfs.dict
# mfs = calculate_best_fit_ndf(mbh, mbh.redshift, quantity_range=np.linspace(0,20,10000))

# match_ms = {}
# for z in mbh.redshift:
#     lf_phi      = np.sort(lfs[z][:,1])
#     ind         = np.argsort(mfs[z][:,1])
#     mf_phi      = mfs[z][:,1][ind]
#     mf_m        = mfs[z][:,0][ind]
#     match_ms[z] = np.interp(lf_phi,mf_phi,mf_m)
    
# for z in  mbh.redshift:
#     lum = lfs[z][:,0][::-1]
#     m   = match_ms[z]
#     plt.plot(lum,m)

## MORE COMPLICATED EDD RATIO DIST
from scipy.special import hyp2f1
from scipy.integrate import quad
from model.helper import make_list

best_fit_params = mbh.parameter.at_z(0)
bhmf            = calculate_best_fit_ndf(mbh, 0)[0]


print('Formulate everything in terms of log_lambda instead of lam, will make fitting much easier')
def erdf(lam, lambda_star, p):
    #if lam>1:
    #    return(0)
    if lam<0:
        return(0)
    else:
        print('You\'re evaluating hyp at 1 now, but need to do at at inf we you use full power law')
        x_0 = 1/(hyp2f1(1,1/p,1+1/p,-(1/lambda_star)**p)) # normalise to 1
        return(x_0/(1+(lam/lambda_star)**p))
    
def help_L(log_L, lam):
    log_mbh = log_L - np.log10(lam) - 38.1
    if log_mbh<0:
        print('whoops')
        log_mbh=0
    phi_bh  = np.power(10, mbh.calculate_log_abundance(log_mbh, 0, best_fit_params))
    contribution = phi_bh*erdf(lam,0.01,2.5)/lam
    if np.any(phi_bh == 0) and np.any(log_mbh<3):
        print('outside of complete model range where values can be reliably calculated')
        print('have to find better condition for that though')
        print(log_L)
    
    return(contribution)
    
def phi_bol(log_L):
    log_L = make_list(log_L)
    phi = []
    for L in log_L:
        func = lambda l: help_L(L, l)
        phi.append(quad(func,0,np.inf,limit=50)[0])
    phi = np.log10(phi)
    return(np.array([log_L,phi]).T)

log_L = np.linspace(40,60,100)
p = phi_bol(log_L)
print('probably has problems at low L bc phi becomes very large in linear space')

plt.figure()
plt.plot(p[:,0],p[:,1])
plt.plot(bhmf[:,0]+38.1, bhmf[:,1])
