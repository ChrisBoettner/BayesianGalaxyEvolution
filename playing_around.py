#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 13:41:12 2022

@author: chris
"""
from model.interface import load_model, run_model, save_model
from model.plotting.plotting import *

mstar  = load_model('mstar','changing')
muv    = load_model('Muv','changing')
# mbh    = load_model('mbh','quasar')
# lbol   = load_model('Lbol', 'eddington')

# for z in range(4,9):
#     Plot_q1_q2_relation(muv, mstar, z=z, datapoints=True, sigma=[1,2,3])

plot = Plot_q1_q2_relation(muv, mstar, z=4, datapoints=True, sigma=[1,2,3],
                           quantity_range=np.linspace(-22.24,-16.23,100))

x = np.linspace(-22.24,-16.23,100)

num =10000
a_draw = 0.03 * np.random.randn(num) - 0.54 
b_draw = 0.02 * np.random.randn(num) + 9.7

y = a_draw * (x[:,np.newaxis]+21) + b_draw

for i in range(num):
    plot.axes.plot(x,y[:,i], linewidth = 2, color = 'grey', alpha = 0.8)