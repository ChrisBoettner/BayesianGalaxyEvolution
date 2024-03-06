#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 18:28:54 2022

@author: chris
"""
#%%
from model.interface import load_model, run_model, save_model
from model.plotting.plotting import *

print('distributions in log space or linear space?')    
print('read wechsler review') # https://ui.adsabs.harvard.edu/abs/2018ARA%26A..56..435W/abstract

# =============================================================================
#%%

z=0  

mstar  = load_model('mstar','stellar_blackhole', redshift=z)  
par = mstar.parameter.at_z(z)

x = mstar.quantity_options['quantity_range']
sigmas = [1,0.6,0.3,0.2,0.1,0.01]

plt.close()

# raise ValueError('Maybe before you go on, implement two different sigmas for '
#                   'mstar and muv and see if you get similar asymmetric distribution '
#                   'as data suggests. (see paper on how to get q_1|q_2 distribution')

for s in sigmas:
    y = mstar.calculate_log_abundance(x, z, par, scatter_name='lognormal',
                                          scatter_parameter=s, num=int(1e+3))
    plt.plot(x,y,label=str(s), linewidth=3, alpha=0.7)
y = mstar.calculate_log_abundance(x, z, par)
plt.plot(x,y,label='delta', linewidth=8,color='black', zorder=0)
plt.legend()
plt.xlim(mstar.quantity_options['quantity_range'][0],
         mstar.quantity_options['quantity_range'][-1])
plt.ylim(mstar.quantity_options['ndf_y_axis_limit'])
        
# =============================================================================
#%%
import numpy as np
from model.analysis.calculations import calculate_q1_q2_relation
from model.scatter import Joint_distribution, calculate_q1_q2_conditional_pdf
from scipy.integrate import trapezoid

z = 4
mstar  = load_model('mstar','stellar_blackhole', redshift=z)
muv    = load_model('Muv','stellar_blackhole', redshift=z)

scatter_name = 'lognormal'

mstar_sigma = 0.05
muv_sigma   = 0.25
muv_skew    = -4

mstar_joint = Joint_distribution(mstar, scatter_name, mstar_sigma)
muv_joint   = Joint_distribution(muv, scatter_name, muv_sigma)

muv_joint_skew   = Joint_distribution(muv, 'skewlognormal', 
                                      scatter_parameter=muv_sigma,
                                      skew_parameter=muv_skew)

mstar_par = mstar.parameter.at_z(z)
muv_par   = muv.parameter.at_z(z)

muv_val     = -20

log_q_space = np.linspace(7.5,11,500)

probs = calculate_q1_q2_conditional_pdf(mstar_joint, muv_joint, 
                                        log_q_space, muv_val, 
                                        mstar_par, muv_par, z)
probs_skew = calculate_q1_q2_conditional_pdf(mstar_joint, muv_joint_skew, 
                                             log_q_space, muv_val, 
                                             mstar_par, muv_par, z)

no_scatter = calculate_q1_q2_relation(muv, mstar, z, [muv_val])[1][0,1]

m = np.argmax(probs)
w = 100
# plt.plot(log_q_space[m-w:m+w]-log_q_space[m], (probs[m-w:m+w]))
plt.plot(log_q_space, probs)   
plt.axvline(no_scatter)
plt.plot(log_q_space, probs_skew) 
    
# %%
