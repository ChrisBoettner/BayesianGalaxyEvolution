#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  1 16:06:00 2023

@author: chris
"""

import numpy as np
import matplotlib.pyplot as plt
from model.plotting.plotting import *
from model.model import ModelResult
from model.helper import calculate_percentiles

mstar = load_model('mstar', 'stellar_blackhole', redshift=0)

total_bhmf = np.load('model/data/BHMF/Schulze2010total.npz')
total_bhmf = {0:total_bhmf['0']}

model = ModelResult(0, total_bhmf,
              'mbh', 'quasar', 'successive',
              fitting_method='mcmc',
              saving_mode='temp',
              fixed_m_c=False)



x = np.linspace(6,9.2,100)
phi_sample = np.array(model.get_ndf_sample(z=0, num=1000, 
                                           quantity_range=x))[:,:,1]
ndf = calculate_percentiles(phi_sample).T

plt.close()

plt.fill_between(x, ndf[:,1], ndf[:,2], alpha=0.2, color='purple', 
                 label='black hole mass model')
#plt.plot(x, ndf[:,0], color='purple', linewidth= 5, label='Model')

plt.scatter(total_bhmf[0][:,0],total_bhmf[0][:,1], color='black', 
            label = 'Schulze2010 total BHMF')

plt.xlabel('log (black hole mass)')
plt.ylabel('log phi (black hole mass)')

plt.legend()

plt.savefig('totalBHMF.png')

# p =Plot_q1_q2_relation(mstar, model, datapoints=True,
#                     quantity_range=np.linspace(8.7,11.9,100),
#                     sigma=2,
#                     columns='single')