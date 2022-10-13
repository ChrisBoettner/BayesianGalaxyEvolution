#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 13:41:12 2022

@author: chris
"""

from model.interface import load_model, run_model, save_model
from model.plotting.plotting import *

#mstar  = load_model('mstar','changing')
#muv    = load_model('Muv','changing')
#mbh    = load_model('mbh','quasar', prior_name='successive')
#lbol   = load_model('Lbol', 'eddington', prior_name='successive')


# redshift = 0
# mstar_no = run_model('mstar', 'none', redshift=redshift)
# mstar_st = run_model('mstar', 'stellar', redshift=redshift)
# mstar_sb = run_model('mstar', 'stellar_blackhole', redshift=redshift)
# Plot_best_fit_ndf([mstar_no, mstar_st, mstar_sb], columns='single').save()

#Plot_ndf_sample(mstar, sigma=[1,2,3])

# muv = save_model(quantity_name = 'Muv', physics_name='changing',
#                    prior_name='successive', min_chain_length=5000, 
#                    num_walker=50)
# mstar = save_model(quantity_name = 'mstar', physics_name='changing',
#                    prior_name='successive', min_chain_length=5000, 
#                    num_walker=50)
# mbh = save_model(quantity_name = 'mbh', physics_name='quasar',
#                    prior_name='successive', min_chain_length=5000, 
#                    num_walker=50)
lbol = save_model(quantity_name = 'Lbol', physics_name='eddington',
                   prior_name='successive', min_chain_length=5000, 
                   num_walker=10)