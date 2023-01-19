#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 19 12:44:16 2022

@author: chris
"""
import numpy as np
from model.interface import load_model
from model.analysis.parameter import tabulate_parameter
from model.analysis.surveys import tabulate_expected_number

def make_parameter_tables():
    '''
    Create parameter tables for galaxy and black hole properties
    '''
    mstar = load_model('mstar', 'stellar_blackhole')
    muv   = load_model('Muv', 'stellar_blackhole')
    mbh   = load_model('mbh', 'quasar')
    lbol  = load_model('Lbol', 'eddington')
    
    gal_table = tabulate_parameter([mstar, muv])
    bh_table  = tabulate_parameter([mbh, lbol], redshift=np.arange(8))
    return(gal_table, bh_table)

def make_survey_tables():
    '''
    Calculate expected upper limits of galaxies for different surveys and
    redshifts.
    '''
    muv   = load_model('Muv', 'stellar_blackhole')
    
    JWST_table   = tabulate_expected_number(muv, 'JWST')
    Euclid_table = tabulate_expected_number(muv, 'Euclid')
    return(JWST_table, Euclid_table)

if __name__ == '__main__':
    #gal_table, bh_table = make_parameter_tables()
    
    JWST_table, Euclid_table = make_survey_tables()
    