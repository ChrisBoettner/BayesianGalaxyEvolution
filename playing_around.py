#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 13:41:12 2022

@author: chris
"""

from model.interface import load_model, run_model, save_model
from model.plotting.plotting import *

mstar  = load_model('mstar','changing')
#muv    = load_model('Muv','changing')
#mbh    = load_model('mbh','quasar', prior_name='successive')
#lbol   = load_model('Lbol', 'eddington', prior_name='successive')


# redshift = 0
# mstar_no = load_model('mstar', 'none', redshift=redshift)
# mstar_st = load_model('mstar', 'stellar', redshift=redshift)
# mstar_sb = load_model('mstar', 'stellar_blackhole', redshift=redshift)
# Plot_best_fit_ndf([mstar_no, mstar_st, mstar_sb], columns='single').save()

Plot_ndf_sample(mstar, sigma=[1,2])