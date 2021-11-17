# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 18:33:15 2021

@author: boettner
"""

from matplotlib import rc_file
rc_file('plots/settings.rc')  # <-- the file containing your settings

import numpy as np
import matplotlib.pyplot as plt

from smhr import calculate_SHMR, abundance_matching
from data_processing import group, z_ordered_data

def load():
    # get z=1,2,3,4 for Davidson, z=1,2,3 for Ilbert
    davidson    = np.load('data/Davidson2017SMF.npz'); davidson = {i:davidson[j] for i,j in [['0','2'],['1','4'],['2','6'],['3','8']]}
    ilbert      = np.load('data/Ilbert2013SMF.npz');   ilbert = {i:ilbert[j] for i,j in [['0','2'],['1','4'],['2','6']]}
    duncan      = np.load('data/Duncan2014SMF.npz')      
    song        = np.load('data/Song2016SMF.npz')       
    bhatawdekar = np.load('data/Bhatawdekar2018SMF.npz')
    stefanon    = np.load('data/Stefanon2021SMF.npz')
    hmfs        = np.load('data/HMF.npz'); hmfs = [hmfs[str(i)] for i in range(20)]   
    
    ## TURN DATA INTO GROUP OBJECTS, INCLUDING PLOT PARAMETER
    davidson    = group(davidson,    [1,2,3,4]   ).plot_parameter('black', 'o', 'Davidson2017')
    ilbert      = group(ilbert,      [1,2,3]     ).plot_parameter('black', 'H', 'Ilbert2013')
    duncan      = group(duncan,      [4,5,6,7]   ).plot_parameter('black', 'v', 'Duncan2014')
    song        = group(song,        [6,7,8]     ).plot_parameter('black', 's', 'Song2016')
    bhatawdekar = group(bhatawdekar, [6,7,8,9]   ).plot_parameter('black', '^', 'Bhatawdekar2019')
    stefanon    = group(stefanon,    [6,7,8,9,10]).plot_parameter('black', 'X', 'Stefanon2021')
    groups      = [davidson, ilbert, duncan, song, bhatawdekar, stefanon]
    
    ## DATA SORTED BY REDSHIFT
    smfs = z_ordered_data(groups)
    # undo log for easier fitting
    raise10 = lambda list_log: [10**list_log[i] for i in range(len(list_log))]
    smfs = raise10(smfs)
    hmfs = raise10(hmfs)
    
    ##
    i=0
    smf = smfs[i][smfs[i][:,1]>1e-6] # cut unreliable values
    hmf = hmfs[i+1]  
    return(smf,hmf)

smf, hmf = load()

from scipy.interpolate import interp1d

hmf_func = interp1d(*hmf.T)

m_c = 5e+10

scaled = lambda x, A, alpha, beta: hmf_func(x)*A * 1/((x/m_c)**(-alpha)+(x/m_c)**(beta))

from scipy.optimize import curve_fit, least_squares

def cost(params, x, y):
    return(np.log10(y)-np.log10(scaled(x,*params)))

par1 = least_squares(cost, [0.01, 1, 0.4], args = [smf[:,0], smf[:,1]]).x

par2, pcov = curve_fit(scaled, smf[:,0], smf[:,1],p0=[0.01,0.5,1.5])

# maybe optimize in log space instead?

plt.close('all')
plt.scatter(*smf.T)
plt.loglog(hmf[:,0],scaled(hmf[:,0],*par1))
plt.loglog(hmf[:,0],scaled(hmf[:,0],*par2))