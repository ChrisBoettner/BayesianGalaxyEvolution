#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 19 12:44:16 2022

@author: chris
"""
import numpy as np
from model.interface import load_model
from model.analysis.parameter import tabulate_parameter


if __name__ == '__main__':
    mstar = load_model('mstar', 'stellar_blackhole')
    muv   = load_model('Muv', 'stellar_blackhole')
    mbh   = load_model('mbh', 'quasar')
    lbol  = load_model('Lbol', 'eddington')
    
    gal_table = tabulate_parameter([mstar, muv])
    bh_table  = tabulate_parameter([mbh, lbol], redshift=np.arange(8))
    
    tables = gal_table+bh_table