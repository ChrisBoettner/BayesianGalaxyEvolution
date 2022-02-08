#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 11:49:01 2022

@author: chris
"""

import numpy as np

from data_processing import load_mcmc_data

class model():
    '''
    Determined model that related halo mass to stellar mass and UV luminosity. 
    To select model, call function with either 'mstar' or 'lum'.
    '''
    def __init__(self, quantity_name):
        parameter, distribution = load_mcmc_data(quantity_name)
        
        self.quantity_name = quantity_name
        
        self.parameter     = dataset(parameter)
        self.distribution  = dataset(distribution)        
        
    def calculate_quantity(self, m_h, z, num = int(1e+6), m_c = 1e+12):
        '''
        Calcuate value of the quantity at a given halo mass and redshift. To do
        this, random samples are drawn from the parameter distribution and 
        percentiles are calculated for resulting quantity distribution.
        '''
        
        # randomly draw from parameter distribution at z 
        random_draw = np.random.choice(range(self.distribution.at_z(z).shape[0]),
                                       size = num, replace = False)
        parameter_draw = self.distribution.at_z(z)[random_draw]
        
        # seperate the values to easily call function
        if parameter_draw.shape[1] == 3: # meaning sn + bh feedback
            A       = parameter_draw[:,0]
            alpha   = parameter_draw[:,1]
            beta    = parameter_draw[:,2]
        if parameter_draw.shape[1]== 2: # just sn feedback
            A       = parameter_draw[:,0]
            alpha   = parameter_draw[:,1]
            beta    = np.zeros(num)     # set beta to zero
         
        # correct for the change in variables done in the fit originally    
        if self.quantity_name == 'lum':
            A = A*1e+18
        
        # calculate quantity distribution using model function
        quantity_dist = _quantity_from_snbh_model_function(m_h, A, alpha, beta, m_c)
        
        # calculate percentiles
        median     = np.percentile(quantity_dist, 50)
        lower      = np.percentile(quantity_dist, 16)
        upper      = np.percentile(quantity_dist, 84)      
        return(np.array([median, lower, upper])) 
    
class dataset():
    # easily retrieve data at certain redshift
    def __init__(self, data):
        self.data = data
    def at_z(self, redshift):
        return(self.data[redshift])


def lum_to_sfr(L_nu):
    '''
    Converts between UV luminosity (in ergs s^-1 Hz^-1) and star formation rate
    (in solar masses yr^-1), using the Kennicutt relation.
    Kennicutt parameter chosen for Chabrier IMF and mass limits of 0.01 and 100 
    solar masses.
    See
    https://astronomy.stackexchange.com/questions/39806/relation-between-absolute-magnitude-of-uv-and-star-formation-rate
    '''
    C_Kennicutt = 7.8e-29
    return(C_Kennicutt*L_nu)

## HELP FUNCTIONS    
def _quantity_from_snbh_model_function(m_h, A, alpha, beta, m_c):
    '''
    Modelled relation of quantity to halo mass, including stellar and black hole
    feedback. (For stellar feedback only: beta = 0)
    '''
    if np.isnan(m_h).any() or np.any(m_h<=0):
        return(np.nan)
    sn = (m_h/m_c)**(-alpha)
    bh = (m_h/m_c)**beta
    return(A * m_h/(sn + bh))   
    