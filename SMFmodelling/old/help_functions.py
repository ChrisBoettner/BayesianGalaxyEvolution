#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 16:14:43 2021

@author: boettner
"""
import numpy as np

## Fit Functions

def schechter(m, phi, m_star, alpha, mode = 'dndlogm'):
    if mode == 'dndm':
        return(phi/m_star*(m/m_star)**alpha*np.exp(-m/m_star))
    if mode == 'dndlogm':
        return(phi/m_star*(m/m_star)**alpha*np.exp(-m/m_star) *m*np.log(10))

def log_schechter(m, A, m_star, alpha, mode = 'dndlogm'):
    '''
    A = log10(phi/m_star)
    '''
    if mode == 'dndm':
        return((A + alpha* (np.log10(m/m_star))  - np.log(10)*m/m_star))
    if mode == 'dndlogm':
        print('Error: dndlogm not yet implemented')

def power_law(m, A, alpha, mode = 'dndlogm'):
    if mode == 'dndm':
        return(A*m**alpha)
    if mode == 'dndlogm':
        return(A*m**alpha *m*np.log(10))

def log_power_law(m, logA, alpha, mode = 'dndlogm'):
    '''
    logA = log10(A)
    '''
    if mode == 'dndm':
        return(logA + alpha * np.log10(m))
    if mode == 'dndlogm':
        print('Error: dndlogm not yet implemented')

## Plot Functions

def plot_schechter(phi, m_star, alpha, **kwargs):
    x = np.logspace(7, 12, 1000)
    y = schechter(x,phi, m_star, alpha, **kwargs)
    return(np.array([x,y]))

def plot_powerlaw(logA, alpha, **kwargs):
    x = np.logspace(6, 11, 1000)
    y = power_law(x, logA, alpha, **kwargs)
    return(np.array([x,y]))