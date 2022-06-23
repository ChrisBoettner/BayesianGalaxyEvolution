#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 16:23:34 2022

@author: chris
"""

from scipy.integrate import quad
import numpy as np
import matplotlib.pyplot as plt

from model.interface import load_model
from model.eddington import ERDF
mbh    = load_model('mbh','quasar',prior_name='successive')

#%%
def schechter_function(log_quantity, log_phi_star, log_quant_star, alpha):
    x = log_quantity - log_quant_star

    # normalisation
    norm = np.log10(np.log(10)) + log_phi_star
    # power law
    power_law = (alpha + 1) * x
    # exponential
    if np.any(np.abs(x) > 80):  # deal with overflow
        exponential = np.sign(x) * np.inf
    else:
        exponential = -np.power(10, x) / np.log(10)
    return(np.power(10,norm + power_law + exponential))

def unnorm_erdf(edd, edd_star, p1, p2):
    x = edd/edd_star
    broken_pl = 1/(x**p1 + x**p2)
    return(broken_pl)

def erdf(edd, A, edd_star, p1, p2):
    return(A*unnorm_erdf(edd, edd_star, p1, p2))

def bh_mass(L, edd):
    return(L/10**38.1 * 1/edd)
 
def contribution(edd, L):
    m_bh      = bh_mass(L, edd)
    edd_star = 0.001
    p1       = 1
    p2       = 2.5
    #A        = 1/quad(unnorm_erdf, 0, np.inf, args=(edd_star,p1,p2))[0]

    #xi       = erdf(edd, A, edd_star, p1, p2)
    xi = ERDF(np.log10(edd_star), p1, p2).pdf(np.log10(edd))
    
    log_phi_star   = -3
    log_quant_star = 11
    alpha          = -2
    
    #phi_bh   = schechter_function(np.log10(mbh), log_phi_star, 
    #                              log_quant_star, alpha)
    
    #p        = mbh.parameter.at_z(0)
    #phi_bh   = 10**mbh.calculate_log_abundance(np.log10(m_bh), 0, p)
    
    phi_bh   = 10**mbh.calculate_log_hmf(np.log10(m_bh)+6, 0)
    return(phi_bh*xi)   
    
def lum_func(L):
    return(quad(contribution, 0, np.inf, args=(L))[0])

L   = np.logspace(40, 50, 100)
phi = np.array([lum_func(l) for l in L])

l = np.log10(L)
p = np.log10(phi)

    