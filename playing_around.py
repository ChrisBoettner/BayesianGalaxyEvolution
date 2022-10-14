#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 13:41:12 2022

@author: chris
"""

from model.interface import load_model, run_model, save_model
from model.plotting.plotting import *

# mstar  = load_model('mstar','changing')
# muv    = load_model('Muv','changing')
# mbh    = load_model('mbh','quasar')
lbol   = load_model('Lbol', 'eddington')


# redshift = 0
# mstar_no = run_model('mstar', 'none', redshift=redshift)
# mstar_st = run_model('mstar', 'stellar', redshift=redshift)
# mstar_sb = run_model('mstar', 'stellar_blackhole', redshift=redshift)
# Plot_best_fit_ndf([mstar_no, mstar_st, mstar_sb], columns='single').save()

# Plot_conditional_ERDF(lbol, parameter = [40 ,  2,
#                                           -2,  2], columns='single').save()


# Plot_ndf_intervals(mstar, sigma=[1,2,3]).save()
# Plot_ndf_intervals(muv, sigma=[1,2,3]).save()
# Plot_ndf_intervals(mbh, sigma=[1,2,3]).save()

lbol_free = run_model('Lbol', 'eddington_free_ERDF')
Plot_ndf_intervals(lbol, sigma=[1,2,3], num = 1000,
                   additional_models=lbol_free).save()