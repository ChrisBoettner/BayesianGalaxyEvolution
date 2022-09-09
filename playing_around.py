#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 13:41:12 2022

@author: chris
"""

from model.interface import load_model, run_model, save_model
from model.plotting.plotting import *

#mstar  = load_model('mstar','changing')
# muv    = load_model('Muv','changing')
mbh    = load_model('mbh','quasar', prior_name='successive')
lbol   = load_model('Lbol', 'eddington', prior_name='successive')

#%%

log_edd_const  = np.log10(1.26*1e+38)
edd_ratios     = np.array([0.001,0.01,0.1,1]) 
edd_ratio_labels = [f'Eddington ratio: {e}' for e in edd_ratios]

# scale by 3 most sensible since QLF is build for type 1 and type 2 AGN
# and while BHMF only for type 1, so need to correct for that
#plot = Plot_q1_q2_relation(mbh, lbol,
#                            log_slopes = log_edd_const+np.log10(edd_ratios),
#                            log_slope_labels = edd_ratio_labels, datapoints=True,
#                            scaled_ndf=(mbh,[3,30]))

plt.close()

plot = Plot_q1_q2_relation(mbh, lbol, datapoints=True,
                            scaled_ndf=(mbh,[3,30]))

import numpy as np
lbols = np.linspace(33,49,100)

erdfs = lbol.calculate_conditional_ERDF(lbols,0,lbol.parameter.at_z(0))

ms = []
for log_L in erdfs.keys():
    erdf = erdfs[log_L]
    erdf[:,0] = log_L - erdf[:,0] - log_edd_const 
    # plot.axes.scatter(erdf[:,0],np.repeat(log_L,len(erdf)),
    # alpha = erdf[:,1]/np.amax(erdf[:,1]))

    dx = np.abs(erdf[1,0]-erdf[0,0])
    ms.append(np.sum(10**erdf[:,0]*erdf[:,1]*dx))

plot.axes.plot(np.log10(ms), lbols, label='conditional ERDF mean')
plt.legend()

#%%
from model.data.load import load_data_points
import numpy as np
import matplotlib.pyplot as plt

data = load_data_points('mbh_Lbol')
data = data[~np.isnan(data[:,0])]

mean_L = np.mean(data[:,1])

log_edd_const  = np.log10(1.26*1e+38)

cond_erdf     = lbol.calculate_conditional_ERDF(mean_L,0,lbol.parameter.at_z(0),
                                                eddington_ratio_space=np.linspace(-6, 31,10000))[mean_L]
log_m_bhs     = mean_L - cond_erdf[:,0] - log_edd_const
m_bh_dist     = np.copy(cond_erdf); m_bh_dist[:,0] = log_m_bhs 

m_bh_hist     = np.histogram(data[:,0], density=True)

lower_limit = mean_L-log_edd_const
upper_limit = mean_L-log_edd_const+2

lower_ind = np.argmin(m_bh_dist[:,0]>lower_limit)
upper_ind = np.argmin(m_bh_dist[:,0]>upper_limit)

from scipy.interpolate import interp1d

dx = np.abs(m_bh_dist[1,0]-m_bh_dist[0,0])
m_bh_pmf = np.copy(m_bh_dist)[::-1]; m_bh_pmf[:,1] = m_bh_pmf[:,1]*dx

m_bh_cdf = interp1d(m_bh_pmf[:,0], np.cumsum(m_bh_pmf[:,1]))

from scipy.stats import kstest

    
norm = m_bh_cdf(upper_limit) - m_bh_cdf(lower_limit)

plt.figure()
plt.hist(data[:,0], bins=15, density=True, label = 'observed black hole mass distribution')
plt.plot(m_bh_dist[:,0], m_bh_dist[:,1], label='predicted black hole mass distribution')
plt.axvline(lower_limit, color='black', label = 'Eddington ratio = 1')
plt.axvline(upper_limit, color='black', label = 'Eddington ratio = 0.01')
plt.plot(m_bh_dist[upper_ind:lower_ind,0], 
         m_bh_dist[upper_ind:lower_ind,1]/norm,
         label='adjusted model distribution')
plt.xlim([6,10])
plt.title(f'mean bolometric luminosity = {mean_L}')
plt.xlabel('log black hole mass (in stellar masses)')
plt.ylabel('probability')
plt.legend()



ks_result = kstest(np.sort(data[:,0]), m_bh_cdf)
print('pvalue = ' + str(ks_result.pvalue))

#%%
# alternative approach summing all erdfs instead of using mean
from scipy.interpolate import interp1d


edd_space = np.linspace(-6, 31,10000)

#calculate all erdfs
bh_distributions     = lbol.calculate_conditional_ERDF(np.unique(data[:,1]),0,lbol.parameter.at_z(0),
                                                eddington_ratio_space=edd_space,
                                                black_hole_mass_distribution=True)

# calculate number of occurences for every luminosity
unique, counts = np.unique(data[:,1], return_counts=True)
lum_occ = dict(zip(unique, counts))

for l in bh_distributions.keys():
    dist = bh_distributions[l] 
    bh_distributions[l] = interp1d(dist[:,0], dist[:,1])
    
def prob(m_bh):
    p=0
    for l in bh_distributions.keys():
        p = p + bh_distributions[l](m_bh)*lum_occ[l]
    p = p/len(data[:,1])
    return(p)
    

m_bh_x = np.linspace(6,10,100)
ps     = [prob(m) for m in m_bh_x]

plt.plot(m_bh_x, ps, label='sum of dists')