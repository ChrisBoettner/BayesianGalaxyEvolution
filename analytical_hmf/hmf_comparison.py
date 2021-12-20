#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 15:10:40 2021

@author: boettner
"""

import numpy as np
import matplotlib.pyplot as plt

from hmf_analytical import hmf_approx

hmf_log  = np.load('data/HMF.npz'); 
hmfs = {str(i):np.power(10,hmf_log[str(i)]) for i in range(20)}

z = 19
mass_range = np.array([1e+1,1e+14])

hmf = hmfs[str(z)]
hmf_full = hmf[np.where(np.logical_and(hmf[:,0]>=mass_range[0], hmf[:,0]<=mass_range[1]))]

masses = hmf_full[:,0]

hmf_app2  = hmf_approx(M = masses, value = z, mode = 'z', order = 2)
hmf_app0  = hmf_approx(M = masses, value = z, mode = 'z', order = 0)

plt.close('all')
fig, ax = plt.subplots()
ax.set_xscale('log'); ax.set_yscale('log')

ax.plot(masses, hmf_full[:,1])
ax.plot(masses, hmf_app2)
ax.plot(masses, hmf_app0)