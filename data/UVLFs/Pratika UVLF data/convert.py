#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 12:35:47 2022

@author: chris
"""
import numpy as np

#dictionary of LF parameters (phi*, M*, alpha)
LFpars = {
0.05:[10**-2.37, -18.04, -1.22],  # Wyder+05         https://ui.adsabs.harvard.edu/abs/2005ApJ...619L..15W/abstract
0.5 : [1.69e-3, -19.49, -1.55],   # Arnouts+05       https://ui.adsabs.harvard.edu/abs/2005ApJ...619L..43A/abstract
2.3: [2.75e-3, -20.70, -1.73],    # Reddy&Steidel 09 https://ui.adsabs.harvard.edu/abs/2009ApJ...692..778R/abstract
3:   [1.71e-3, -20.97, -1.73],    # Reddy&Steidel 09
10:  [10**-4.89, -20.6, -2.28],   # Oesch+18         https://ui.adsabs.harvard.edu/abs/2018ApJ...855..105O/abstract
}

LFgrid = {3:
[[-22.77, 0.003e-3, 0.001e-3, np.nan],
[-22.27, 0.030e-3, 0.013e-3, np.nan],
[-21.77, 0.085e-3, 0.032e-3, np.nan],
[-21.27, 0.240e-3, 0.104e-3, np.nan],
[-20.77, 0.686e-3, 0.249e-3, np.nan],
[-20.27, 1.530e-3, 0.273e-3, np.nan],
[-19.77, 2.934e-3, 0.333e-3, np.nan],
[-19.27, 4.296e-3, 0.432e-3, np.nan],
[-18.77, 5.536e-3, 0.601e-3, np.nan]]
, 2.3:
[[-22.58, 0.004e-3, 0.003e-3, np.nan],
[-22.08, 0.035e-3, 0.007e-3, np.nan],
[-21.58, 0.142e-3, 0.016e-3, np.nan],
[-21.08, 0.341e-3, 0.058e-3, np.nan],
[-20.58, 1.246e-3, 0.083e-3, np.nan],
[-20.08, 2.030e-3, 0.196e-3, np.nan],
[-19.58, 3.583e-3, 0.319e-3, np.nan],
[-19.08, 7.171e-3, 0.552e-3, np.nan],
[-18.58, 8.188e-3, 0.777e-3, np.nan],
[-18.08, 12.62e-3, 1.778e-3, np.nan]],
0.05 :
[[-19.899, 5.083e-06, 4.458e-06, np.nan],
[-19.546, 7.774e-05, 2.430e-05, np.nan],
[-18.991, 2.738e-04, 3.911e-05, np.nan],
[-18.511, 7.037e-04, 6.384e-05, np.nan],
[-18.032, 1.587e-03, 1.250e-04, np.nan],
[-17.552, 2.544e-03, 1.647e-04, np.nan],
[-17.098, 2.977e-03, 2.345e-04, np.nan],
[-16.517, 2.900e-03, 3.715e-04, np.nan],
[-16.013, 5.164e-03, 6.791e-04, np.nan],
[-15.558, 6.369e-03, 1.006e-03, np.nan],
[-15.079, 7.653e-03, 1.654e-03, np.nan],
[-14.625, 6.539e-03, 2.439e-03, np.nan],
[-13.994, 9.195e-03, 3.815e-03, np.nan],
[-13.539, 8.499e-03, 5.071e-03, np.nan],
[-13.060, 1.327e-02, 8.756e-03, np.nan],
[-12.025, 4.674e-02, 3.424e-02, np.nan]],
0.5:
[[-20.323, 3.006e-04, 1.357e-04, np.nan],
[-20.167, 4.055e-05, 3.872e-05, np.nan],
[-19.702, 4.721e-04, 1.383e-04, np.nan],
[-19.295, 4.677e-04, 1.584e-04, np.nan],
[-18.986, 1.230e-03, 3.447e-04, np.nan],
[-18.573, 1.770e-03, 4.275e-04, np.nan],
[-18.249, 2.153e-03, 6.172e-04, np.nan],
[-17.843, 2.992e-03, 6.784e-04, np.nan],
[-17.476, 2.944e-03, 8.549e-04, np.nan],
[-17.180, 4.634e-03, 1.421e-03, np.nan]],
10:
[[-21.25,   0.0095e-4, 0.0079e-4,  0.022e-4],
[-20.25,   0.098e-4,  0.053e-4,   0.095e-4],
[-19.25,   0.34e-4,   0.22e-4, 0.45e-4],
[-18.25,   1.9e-4,   1.2e-4, 2.5e-4],
[-17.25,   6.3e-4,   5.2e-4, 14.9e-4]]
}
    
for i in [0.05,0.5,2.3,3,10]:
    LFgrid[i]      = np.array(LFgrid[i])
    phi            = np.copy(LFgrid[i][:,1])
    lower          = LFgrid[i][:,1] - LFgrid[i][:,2] 
    if i != 10:
        LFgrid[i][:,3] = LFgrid[i][:,2]
    upper      = LFgrid[i][:,1] + LFgrid[i][:,3] 
    
    phi            = np.log10(phi)
    lower_err      = phi - np.log10(lower)
    upper_err      = np.log10(upper) - phi
    
    LFgrid[i][:,1] = phi
    LFgrid[i][:,2] = lower_err    
    LFgrid[i][:,3] = upper_err 
    
Wyder   = {'0': LFgrid[0.05]}; np.savez('Wyder2005UVLF.npz', **Wyder)
Arnouts = {'0': LFgrid[0.5]}; np.savez('Arnouts2005UVLF.npz', **Arnouts)
Reddy   = {'0': LFgrid[2.3],  '1': LFgrid[3]}; np.savez('Reddy2009UVLF.npz', **Reddy)
Oesch18 = {'0': LFgrid[10]}; np.savez('Oesch2018UVLF.npz', **Oesch18)