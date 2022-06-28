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

import cProfile

#lbol   = load_model('Lbol','quasar',prior_name='successive')
#mbh    = load_model('mbh','quasar',prior_name='successive')
# def run():  
#     lbol = run_model('Lbol', 'eddington', prior_name='successive')
#     return(lbol)
# cProfile.run('run()')

print('fits look better if you fix ERDF to values at z=0, so do that')
print('then you can run the calibration, but you should also finish up the rest of the model'
      '(functions + doc strings)')

print('value of C parameter are close to the 38.1 you get from eddington calculation, '
      'kinda makes sense since the m^gamma model is similar to bh model (esp at high masses)'
      'so you effectively put in the black hole mass as in the original calculation')

lbol = save_model('Lbol', 'eddington', prior_name='successive',
                  num_walker=10, min_chain_length=0)

#Plot_best_fit_ndfs(lbol)

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
#%%
# from scipy.special import hyp2f1
# from scipy.integrate import quad
# from model.helper import make_list

# import warnings


# best_fit_params = mbh.parameter.at_z(0)
# bhmf            = calculate_best_fit_ndf(mbh, 0)[0]


# print('Formulate everything in terms of log_lambda instead of lam, will make fitting much easier')
# print('keep P(lambda>1)=0, helps with low mass end problems + integration/normalisation issues and still gives double power law over sensible range')
# def erdf(lam, lambda_star, p):
#     if lam>1:
#         return(0)
#     def func(lam):    
#         if lam<lambda_star:
#             return(0)
#         else:
#             return(1/((lam/lambda_star)**p))
        
#     normalisation = 1/quad(func, 0, 1)[0] # you can calculate this analytically
#     return(normalisation*func(lam))
    
# def help_L(log_L, lam):
#     log_mbh = log_L - np.log10(lam) - 38.1
#     if log_mbh<0:
#         print('whoops')
#         log_mbh=0
#     log_phi_bh = mbh.calculate_log_abundance(log_mbh, 0, best_fit_params)
#     contribution = np.power(10, log_phi_bh)*erdf(lam,0.01,25.5)
#     if np.any(np.isnan(log_phi_bh)):
#         print('outside of complete model range where values can be reliably calculated')
#         print('have to find better condition for that though')
#         print(log_L)
    
#     return(contribution)
    
# def phi_bol(log_L):
#     log_L = make_list(log_L)
#     phi, phi2 = [], []
#     lam_sample = np.linspace(1e-10,1,100)
#     for L in log_L:
#         func = lambda l: help_L(L, l)
#         func_vals =[]
#         for la in lam_sample:
#             func_vals.append(func(la)[0])
#         #breakpoint()
#         func_vals = np.array(func_vals)
#         integral = np.sum(func_vals*(lam_sample[1]-lam_sample[0]))/len(lam_sample)
#         phi.append(integral)
#         with warnings.catch_warnings():
#             warnings.simplefilter('error')
#             try:
#                 int_alt = quad(func,0,1,limit=500)[0]
#             except:
#                 int_alt = np.nan
#             phi2.append(int_alt)
#     #breakpoint()
#     phi = np.log10(phi)
#     phi2 = np.log10(phi2)
#     return(np.array([log_L,phi]).T, np.array([log_L,phi2]).T)

# log_L = np.linspace(40,60,100)
# p,p2 = phi_bol(log_L)
# print('probably has problems at low L bc phi becomes very large in linear space')

# plt.figure()
# plt.plot(p[:,0],p[:,1])
# plt.plot(p2[:,0],p2[:,1])
# plt.plot(bhmf[:,0]+38.1, bhmf[:,1])
