#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 18:28:54 2022

@author: chris
"""

from model.interface import load_model, run_model, save_model
from model.plotting.plotting import *

mstar  = load_model('mstar','changing', redshift=0)
# muv    = load_model('Muv','changing')
# mbh    = load_model('mbh','quasar')
# lbol   = load_model('Lbol', 'eddington')


print('distributions in log space or linear space?')    
print('read wechsler review') # https://ui.adsabs.harvard.edu/abs/2018ARA%26A..56..435W/abstract

z=0    
par = mstar.parameter.at_z(z)

# log_q = np.array([9,1])
# sigma = 0.1
# print(mstar.calculate_log_abundance(log_q, z, par))
# print(mstar.calculate_log_abundance(log_q, z, par, scatter_name='normal',
#                                     scatter_parameter=sigma))
# print(mstar.calculate_log_abundance(log_q, z, par, scatter_name='cauchy',
#                                     scatter_parameter=sigma))

x = mstar.quantity_options['quantity_range']
sigmas = [1.1,1,0.9,0.8,0.6,0.4,0.3,0.2,0.1,0.01,0.001]

plt.close()

for s in sigmas:
    y = mstar.calculate_log_abundance(x, z, par, scatter_name='lognormal',
                                          scatter_parameter=s, num=int(1e+4))
    plt.plot(x,y,label=str(s), linewidth=3, alpha=0.7)
y = mstar.calculate_log_abundance(x, z, par)
plt.plot(x,y,label='delta', linewidth=8,color='black', zorder=0)
plt.legend()
#plt.ylim(mstar.quantity_options['ndf_y_axis_limit'])
        
