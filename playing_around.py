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

log_edd_const  = np.log10(1.26*1e+38)
edd_ratios     = np.array([0.001,0.01,0.1,1]) 
edd_ratio_labels = [f'Eddington ratio: {e}' for e in edd_ratios]

plot = Plot_q1_q2_relation(mbh, lbol,
                           log_slopes = log_edd_const+np.log10(edd_ratios),
                           log_slope_labels = edd_ratio_labels, datapoints=True,
                           scaled_ndf=(mbh,[3,30]))

print('makes sense that scaling doesnt work since AGN luminosity population is'
      ' same sample as AGN mass population, both underestimated in same way basically'
      ' (or is it? check again if its not type1+type2 in QLF)')