#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 15:27:02 2022

@author: chris
"""
from matplotlib import rc_file
rc_file('plots/settings.rc')  # <-- the file containing your settings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from modelling import model, calculate_schechter_parameter

################## LOAD MODELS ################################################
#%%
print('Loading models..')
mstar_model = model('mstar')
lum_model   = model('lum')
print('Loading done')

#%%
redshift = range(11)
m_star   = np.logspace(7,11,100)

d = calculate_schechter_parameter(mstar_model,m_star, redshift, num = 100)

