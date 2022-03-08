#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 15:27:02 2022

@author: chris
"""
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from scipy.optimize import minimize_scalar,dual_annealing

from modelling import model, calculate_schechter_parameter, lum_to_mag

################## LOAD MODELS ################################################
#%%
print('Loading models..')
mstar_model = model('mstar')
lum_model   = model('lum')
print('Loading done')

################## CALCULATE SCHECHTER PARAMETER ##############################
#%%
# redshifts
redshift = range(11)

# SMF Schechter parameters
num = 100
m_star   = np.logspace(6,12,100)
params_m, lower_m, upper_m, dist_m = calculate_schechter_parameter(mstar_model, m_star,
                                                                   redshift, num = num)
# UVLF Schechter parameters
lum   = np.logspace(24,30,100)
params_l, lower_l, upper_l, dist_l = calculate_schechter_parameter(lum_model, lum,
                                                                   redshift, num = num)

# # calculate mode instead of median since that is a better 'best fit' value
# def find_mode(dist, lower_bound, upper_bound, median):
#     '''
#     Find Mode of distribution by approximating function using a Gaussian kernal
#     denisity estimate and then minimizing that function using dual annealing.
#     Use 16th and 84th percentile as proper bounds for minimization
#     '''
#     mode = []
#     for i in range(len(dist)):
#         d_func     = gaussian_kde(dist[i].T)
#         d_func_neg = lambda x: (-1)*d_func(x) # negative, since we search for minimum
#         bounds = list(zip(lower_bound[i],upper_bound[i]))
#        # if i ==10:
#        #     import pdb; pdb.set_trace()
#         mode.append(dual_annealing(d_func_neg,bounds, x0 = median[i]).x)
#     return(np.array(mode))

# # change params from median to mode, use 95 percentile as bounds
# bounds_lower_m = [np.percentile(d,  2.5, axis = 0) for d in dist_m]
# bounds_upper_m = [np.percentile(d, 97.5, axis = 0) for d in dist_m]
# params_m = find_mode(dist_m, bounds_lower_m, bounds_upper_m, params_m) 
# bounds_lower_l = [np.percentile(d,  2.5, axis = 0) for d in dist_l]
# bounds_upper_l = [np.percentile(d, 97.5, axis = 0) for d in dist_l]
# params_l = find_mode(dist_l, bounds_lower_l, bounds_upper_l, params_l)

# for UVLF, turn L* to M^UV_* since this is the more commonly cited value
params_mag = np.copy(params_l)
lower_mag  = np.copy(lower_l)
upper_mag  = np.copy(upper_l)
params_mag[:,1] = lum_to_mag(10**params_l[:,1])
lower_mag[:,1]  = lum_to_mag(10**upper_l[:,1]) # upper and lower bound switch in magnitude system
upper_mag[:,1]  = lum_to_mag(10**lower_l[:,1]) # upper and lower bound switch in magnitude system

# calculate errors
lower_err_m    = params_m   - lower_m
upper_err_m    = upper_m    - params_m
lower_err_mag  = params_mag - lower_mag
upper_err_mag  = upper_mag  - params_mag

################## SAVE FOR LATEX TABLE #######################################
#%%
print('Also maybe Modes instead of Medians? same for main sequence?')

def format_to_string(parameter, lower_err, upper_err, redshift, columns,
                     magnitude = False):
    '''
    Creates an array of strings with all the parameters, that's already formatted
    to correctly display parameter values and errors.
    '''
    rows = range(parameter.shape[0])
    cols = range(parameter.shape[1]) 
    
    formatted_DataFrame = pd.DataFrame(index   = rows, columns = cols)
    
    for c in cols:
        for r in rows:
            pres = 1
            param_val = np.format_float_scientific(parameter[r,c], precision = pres)
            param_sig_fig, exponent = param_val.split('e')
            exponent = int(exponent)
            
            if c == 1:  # should be the log_obs_*, looks nicer like this
                exponent       += -1
                param_sig_fig =  np.around(parameter[r,c]/10**exponent, pres)
            
            l_err = np.around(lower_err[r,c]/10**exponent, pres)
            u_err = np.around(upper_err[r,c]/10**exponent, pres)
            
            p_str = str(param_sig_fig)
            l_str = str(l_err)
            u_str = str(u_err)
            
            if len(p_str.split('.')[1]) < pres:
                p_str = p_str + '0'
            if len(l_str.split('.')[1]) < pres:
                l_str = l_str + '0'
            if len(u_str.split('.')[1]) < pres:
                u_str = u_str + '0'
            
            string = r'$\left(' + p_str +r'^{+' + u_str + r'}_{-' + l_str + r'}\right)$'
            if exponent != 0:
                string = string[:-1] + r' \cdot 10^{' + str(exponent) + r'}$'
            else:
                string = string.replace(r'\left(','')
                string = string.replace(r'\right)','') 
                     
            formatted_DataFrame.iloc[r,c] = string
    formatted_DataFrame.insert(0, 'z', redshift)
    formatted_DataFrame.columns = columns
    return(formatted_DataFrame)            


header_m = [r'$z$', r'$\phi_*$ [cMpc$^{-1}$ dex$^{-1}$]', r'$\log M_*$ [$M_\odot$]',
            r'$\alpha$']
header_l = [r'$z$', r'$\phi_*$ [cMpc$^{-1}$ dex$^{-1}$]',
            r'$M^\mathrm{UV}_{*}$ (mag)', r'$\alpha$']
#header_l = [r'$z$', r'$\phi_*$ [cMpc$^{-1}$ dex$^{-1}$]',
#            r'$\log L_*$ [ergs s${^-1}$ Hz$^{-1}$]', r'$\alpha$']

mstar_table    = format_to_string(params_m, lower_err_m, upper_err_m, redshift, header_m)
lum_table      = format_to_string(params_mag, lower_err_mag, upper_err_mag, redshift, header_l)

# turn into text that can be used as latex code
mstar_latex = mstar_table.to_latex(index = False, escape=False, column_format = 'rrrr',
                                caption = r'Schechter parameters for SMF (median of distribution). Errors are 16th and 84th percentile.')
lum_latex   = lum_table.to_latex(  index = False, escape=False, column_format = 'rrrr',
                                 caption = r'Schechter parameters for UVLF (median of distribution). Errors are 16th and 84th percentile.')

all_tables_latex  = mstar_latex + lum_latex