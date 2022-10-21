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
mbh    = load_model('mbh','quasar')
lbol   = load_model('Lbol', 'eddington')

print('finish Lbol - mbh plot: make legend work and bigger, bring shaded area to foreground')

Plot_q1_q2_relation(lbol,mbh)

#Plot_black_hole_mass_distribution(lbol, columns='single').save()