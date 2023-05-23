#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  1 13:36:00 2023

@author: chris
"""
from model.interface import load_model, run_model, save_model
from corner import corner
#%%
mstar = load_model('mstar', 'stellar_blackhole')
muv   = load_model('Muv', 'stellar_blackhole')
mbh   = load_model('mbh', 'quasar')
lbol  = load_model('Lbol', 'eddington')

#%%

ModelResults = [mstar, muv, mbh, lbol]

for ModelResult in ModelResults:
    param_num = ModelResult.distribution.at_z(ModelResult.
                                              redshift[0]).shape[1] 
    param_labels =  ModelResult.quantity_options['param_y_labels']
    if param_num > ModelResult.quantity_options['model_param_num']:
        m_c_label = (r'$\log M_\mathrm{c}^'
                    + ModelResult.quantity_options[
                                    'quantity_subscript'] + r'$')
        param_labels =  [m_c_label] + param_labels
    
    figure = corner(
        ModelResult.distribution.at_z(0),
        labels= param_labels,
        label_kwargs={"fontsize": 32})
    
    figure.tight_layout()
    
    figure.savefig(f'corner_plots/{ModelResult.quantity_name}_corner.pdf')