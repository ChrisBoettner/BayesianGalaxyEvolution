#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 19 12:44:16 2022

@author: chris
"""

from model.interface import load_model
from model.analysis.reference_parametrization import tabulate_reference_parameter

def make_tables(quantity):
    '''
    Make Latex table of parameters for parametrization.
    '''
    if quantity in ['mstar', 'Muv']:
        feedback = 'changing'
    elif quantity=='Lbol':
        feedback = 'quasar'
    else:
        raise ValueError('quantity_name not known.')        
        
    model = load_model(quantity, feedback, prior_name='successive')
    table = tabulate_reference_parameter(model, model.redshift)
    return(table)

if __name__ == '__main__':
    quantities = ['mstar', 'Muv', 'Lbol']
    tables     = ''
    for quantity in quantities:
        tables += make_tables(quantity)