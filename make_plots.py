#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  4 13:09:05 2022

@author: chris
"""

from model.interface import load_model, run_model, save_model
from model.plotting.plotting import *
 
quantities = ['mstar', 'Muv']

for quantity in quantities:
    print(quantity)
    # successive prior
    model = load_model(quantity, 'changing')
    
    Plot_best_fit_ndfs(model).save()
    Plot_marginal_pdfs(model).save()
    Plot_parameter_sample(model).save()
    if quantity == 'mstar':
        Plot_qhmr(model).save()
    Plot_schechter_comparison(model).save()
        
    # marginal pdfs for uniform priors
    models = [load_model(quantity,'none'), load_model(quantity,'stellar'), 
              load_model(quantity,'stellar_blackhole')]
    Plot_best_fit_ndfs(models).save()
    Plot_marginal_pdfs(models).save()