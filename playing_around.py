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
# lbol   = load_model('Lbol', 'eddington')

from make_plots import save_plots

z=5
mstar  = load_model('mstar','changing', redshift=z)
muv    = load_model('Muv','changing', redshift=z)
Plot_q1_q2_relation(muv, mstar, z=z, datapoints=True, sigma=[1],
                    quantity_range=np.linspace(-22.24,-17.23,100),
                    y_lims=(7.5,11.9), columns='single')

# save_plots('Muv_mstar')

print('very asymmetric distribution in mstar. that not included in the model (which assumes direct relation between halo and stellar mass).'
'we get away with that if distribution is symmetric (so ignore scatter). if it was infact something more asymmetrically distributed, maybe this leads to this offeset')