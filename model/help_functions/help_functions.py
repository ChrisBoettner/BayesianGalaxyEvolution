#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 14:16:21 2022

@author: chris
"""
import numpy as np

# convert between luminosity and abosulte magntidue (luminosity given in 
#  ergs s^-1 Hz^-1)
def lum_to_mag(L_nu):
    d    = 3.086e+19                # 10pc in cm
    flux = L_nu/(4*np.pi*d**2)
    M_uv = -2.5*np.log10(flux)-48.6 # definition in AB magnitude system
    return(M_uv)

def mag_to_lum(M_uv):
    d     = 3.086e+19                # 10pc in cm
    log_L = (M_uv + 48.6)/(-2.5) + np.log10(4*np.pi*d**2) 
    return(np.power(10,log_L))