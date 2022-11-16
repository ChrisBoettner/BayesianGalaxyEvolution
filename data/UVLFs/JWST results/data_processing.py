#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 19:02:49 2022

@author: chris
"""

import numpy as np

Donnan_z13 = np.array([[-20.35,-19],
                       [1.03e-5,2.74e-5],
                       [5.9e-6 ,1.37e-5],
                       [5.9e-6 ,1.37e-5]]).T

Harrikane_z12 = np.array([[-23.21,-22.21,-21.21,-20.21,-19.21,-18.21],
                          [5.91e-6,6.51e-6,4.48e-6  ,1.29e-5 ,1.46e-5 ,1.03e-4],
                          [5.91e-6    ,6.51e-6 ,3.82e-6  ,8.7e-6  ,9.9e-6  ,6.7e-5],
                          [0.     ,0.     ,1.034e-5 ,1.72e-5 ,1.95e-5 ,1.06e-4]]).T

Harrikane_z17 = np.array([[-23.59,-20.59],
                          [2.15e-6,5.39e-6], 
                          [2.15e-6,3.57e-6],
                          [0.,7.16e-6]]).T

dicto = {}
dicto['12'] = Harrikane_z12
dicto['13'] = Donnan_z13
dicto['17'] = Harrikane_z17

np.savez('JWST_UVLF.npz', **dicto)
