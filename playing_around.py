#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 13:41:12 2022

@author: chris
"""
from model.interface import load_model, run_model, save_model
from model.plotting.plotting import *

mstar  = load_model('mstar','changing', redshift=0)
# muv    = load_model('Muv','changing')
# mbh    = load_model('mbh','quasar')
# lbol   = load_model('Lbol', 'eddington')

import numpy as np
from scipy.stats import norm
from scipy.integrate import trapezoid

def calc_abundance(model, z, log_q, parameter, sigma):

    upper_bound = model.physics_model.at_z(z).calculate_log_halo_mass(
                                    log_q+4*sigma, *parameter)
    lower_bound = model.physics_model.at_z(z).calculate_log_halo_mass(
                                    log_q-4*sigma, *parameter)
    
    m_h_space = np.linspace(lower_bound, upper_bound, int(1e+5))
    
    dx=m_h_space[1]-m_h_space[0]
    
    print('WELL, NOT ITS ACTUALLY LOG NORMALLY DISTRRIBUTED '
          'YOU PROBABLY WANT A NORMAL DISTRIBUTION IN LIN SPACE')
    norm_dist = norm(loc=log_q, scale=sigma)
    
    q_vals = model.physics_model.at_z(z).calculate_log_quantity(
                                    m_h_space, *parameter)
    
    vals = norm_dist.pdf(q_vals)*10**model.calculate_log_hmf(m_h_space, z)
    
    phi = np.log10(trapezoid(vals,dx=dx))
    return(phi)
    
print('read wechsler review') # https://ui.adsabs.harvard.edu/abs/2018ARA%26A..56..435W/abstract

z=0    
log_q = 9
par = mstar.parameter.at_z(0)

print(calc_abundance(mstar, z, log_q, par, 0.1))
print(mstar.calculate_log_abundance(log_q, z, par))