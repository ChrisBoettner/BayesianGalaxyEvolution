#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  4 13:09:05 2022

@author: chris
"""

from model.interface import load_model
from model.plotting.plotting import *

def make_plots(quantity, show = False, file_format='pdf'):
    '''
    Make and save all plots for a given quantity. 
    Choose if plots are displayed using show parameter. Default is False.
    Choose file format with file_format. Default is 'pdf'.
    '''
    if show:
        plt.ion() 
    else:
        plt.ioff()
    
    if quantity in ['mstar' , 'Muv']: 
        # successive prior
        model = load_model(quantity, 'changing')
        
        Plot_best_fit_ndfs(model).save(file_format)
        Plot_marginal_pdfs(model).save(file_format)
        Plot_parameter_sample(model).save(file_format)
        if quantity == 'mstar':
            Plot_qhmr(model).save()
        Plot_reference_comparison(model).save(file_format)
        del model
        
        # uniform priors
        models = [load_model(quantity,'none'), load_model(quantity,'stellar'), 
                  load_model(quantity,'stellar_blackhole')]
        Plot_best_fit_ndfs(models).save(file_format)
        Plot_marginal_pdfs(models).save(file_format)
    elif quantity in ['Lbol', 'mbh']:
        # successive prior
        model = load_model(quantity, 'quasar', prior_name='successive')
        
        Plot_best_fit_ndfs(model).save(file_format)
        Plot_marginal_pdfs(model).save(file_format)
        Plot_parameter_sample(model).save(file_format)
        Plot_reference_comparison(model).save(file_format)
        del model
        
        # uniform priors
        models = [load_model(quantity,'none'), load_model(quantity,'quasar')]
        Plot_best_fit_ndfs(models).save(file_format)
        Plot_marginal_pdfs(models).save(file_format)
    else:
        raise ValueError('quantity_name not known.')
        
    if not show:
        plt.close('all')
        plt.ion()
    return

if __name__ == '__main__':
    quantities = ['mstar', 'Muv', 'Lbol']
    [make_plots(quantity) for quantity in quantities]
    