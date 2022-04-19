#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 13:58:49 2021

@author: boettner
"""

import numpy as np
from astropy.units import Gyr
from astropy.cosmology import FlatLambdaCDM, z_at_value

# quick test: for more information, see Salcido paper and notes

# cosmological parameter
H_0         = 70                # H_0 in km/s/Mpc
H_0_gyr     = H_0*0.001022      # 70 km/s/Mpc in 1/Gyr
Omega_m0    = 0.3               # matter density parameter
Omega_L0    = 0.7               # dark energy density parameter
Omega_c     = 1.3e+11           # critical density in solar masses/Mpc^3

rho_0    = Omega_m0*Omega_c     # matter density in solar masses per Mpc^-3 

cosmo = FlatLambdaCDM(H_0, Omega_m0) # astropy cosmology instance for easily converting between z and t
t_0   = cosmo.age(z=0).value

# time scales
t_m = 1/(H_0_gyr*np.sqrt(Omega_m0))
t_L = 1/(H_0_gyr*np.sqrt(Omega_L0)) 

# matter power spectrum parametrization
S_0   = 3.98 
gamma = 0.3  
S     = lambda M: S_0*(M/1e+12)**(-gamma) # in solar masses

# extended Press-Schechter parameter
delta_c0 = 1.68
A        = 0.322
q        = 0.3
c        = 0.84

# help function f_Lambda
def f_Lambda(t, A, B, order = 2):
    t_Lscale = t/t_L
    
    if order==0:
        return(1)
    if order==1:
        return(1+A*t_Lscale**2)
    if order==2:
        return(1+A*t_Lscale**2+B*t_Lscale**4)

# value of growth factor at different powers taylor expanded 
def D_pow(t, power, **kwargs):
    
    K_D = 1/((3/2*t_0/t_m)**(2/3) * 2/5 * t_m**2 * f_Lambda(t_0, -0.16, 0.04, **kwargs))
    
    if power == -2:
        return(1/(K_D*(3/2*t/t_m)**(2/3) * 2/5 * t_m**2)**2 * f_Lambda(t, 0.32, 0, **kwargs))          
    if power == -1:
        return(1/(K_D*(3/2*t/t_m)**(2/3) * 2/5 * t_m**2) * f_Lambda(t, 0.16, -0.01, **kwargs))    
    if power == 1:
        return(K_D*(3/2*t/t_m)**(2/3) * 2/5 * t_m**2 * f_Lambda(t, -0.16, 0.04, **kwargs))
    if power == 2:
        return((K_D*(3/2*t/t_m)**(2/3) * 2/5 * t_m**2)**2 * f_Lambda(t, -0.32, 0.11, **kwargs))        
    
def hmf_approx(M, value, mode = 'z', **kwargs):
    
    if mode == 't':
        t = value
    if mode == 'z':
        t = cosmo.age(z=value).value        
    C      = A*rho_0* delta_c0 * gamma*c/np.sqrt(2*np.pi*S(M))
    nu     = ((c*delta_c0)**2/S(M))*D_pow(t,-2,**kwargs)
    nu_inv = S(M)/(c*delta_c0)**2*D_pow(t,2,**kwargs)
    
    dndm = C/M**2 * D_pow(t,-1) * np.exp(-nu/2) * (1 + nu_inv**q)
    
    dndlogm = dndm*M*np.log(10)
    
    return(dndlogm)
    
    
    
    
    
