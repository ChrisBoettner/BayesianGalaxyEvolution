#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 13:41:12 2022

@author: chris
"""
from model.interface import load_model, run_model, save_model
from model.plotting.plotting import *

#%%
mstar = load_model('mstar', 'stellar_blackhole')
muv   = load_model('Muv', 'stellar_blackhole')
mbh   = load_model('mbh', 'quasar')
lbol  = load_model('Lbol', 'eddington')

#%%
num = 10000
plt.close('all')

Plot_quantity_density_evolution(mstar, num_samples=num, 
                                log_q_space = mstar.quantity_options['density_bounds'],
                                columns='single', rasterized=False).save('pdf')
Plot_quantity_density_evolution(muv, num_samples=num,
                                log_q_space = muv.quantity_options['density_bounds'],
                                columns='single')
#Plot_quantity_density_evolution(lbol,num_samples=num,  columns='single')

Plot_stellar_mass_density_evolution(mstar, muv, num_samples=num,
                                    columns='single')
