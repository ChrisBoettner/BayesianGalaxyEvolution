#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 11:48:17 2022

@author: chris
"""

import astropy.units as u
from astropy.cosmology import Planck18, z_at_value

def t_to_z(t):
    '''
    Convert lookback time (in Gyr) to redshift in Planck18 cosmology.
    '''
    z = np.array(
        [z_at_value(Planck18.lookback_time, k * u.Gyr).value for k in t])
    return(z)


import numpy as np
import matplotlib.pyplot as plt

# analytical solution to halo mass growth from Salcido2020 paper Eq. 12

def f(t, tL=17.33, A=0.16, B=-0.01):
    tn = t/tL
    return(1+A*tn**2+B*tn**4)

def mh(t, M0, gamma=0.3, tm=26.04):
    ex = -gamma/2
    term_one = (M0/10**12)**ex
    term_two = 0.31*gamma*((t/tm)**(-2/3)*f(t)-1.67)
    return((term_one+term_two)**(1/ex) * 10**12)

t  = np.linspace(0.5,Planck18.age(0).value*0.99,1000)
M0 = 10**12

Mt = mh(t,M0) 

z = t_to_z(Planck18.age(0).value-t)

plt.close()

plt.plot(z,np.log10(Mt), label=f'log M(z=0) = {np.log10(M0)} [solar masses]')
plt.xlabel('redshift')
plt.ylabel('log halo mass [solar masses]')
plt.legend()