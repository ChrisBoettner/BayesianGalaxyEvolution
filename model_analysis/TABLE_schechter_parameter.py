#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 15:27:02 2022

@author: chris
"""
import numpy as np
import pandas as pd

from modelling import model, calculate_schechter_parameter, lum_to_mag, find_mode

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
num = 500
m_star   = np.logspace(6,12,100)
params_m, lower_m, upper_m, dist_m = calculate_schechter_parameter(mstar_model, m_star,
                                                                   redshift, num = num)
# UVLF Schechter parameters
lum   = np.logspace(24,30,100)
params_l, lower_l, upper_l, dist_l = calculate_schechter_parameter(lum_model, lum,
                                                                   redshift, num = num)

# calculate mode instead of median since that is a better 'best fit' value
# change params from median to mode, use 95 percentile as bounds
# bounds_lower_m = [np.percentile(d,  2.5, axis = 0) for d in dist_m]
# bounds_upper_m = [np.percentile(d, 97.5, axis = 0) for d in dist_m]
# params_m       = np.array([find_mode(dist_m[i], bounds_lower_m[i], bounds_upper_m[i]) for i in range(len(dist_m))]) 
# bounds_lower_l = [np.percentile(d,  2.5, axis = 0) for d in dist_l]
# bounds_upper_l = [np.percentile(d, 97.5, axis = 0) for d in dist_l]
# params_l       = np.array([find_mode(dist_l[i], bounds_lower_l[i], bounds_upper_l[i]) for i in range(len(dist_l))])

# for UVLF, turn L* to M^UV_* since this is the more commonly cited value
params_mag = np.copy(params_l)
lower_mag  = np.copy(lower_l)
upper_mag  = np.copy(upper_l)
params_mag[:,1] = lum_to_mag(10**params_l[:,1])
lower_mag[:,1]  = lum_to_mag(10**upper_l[:,1]) # upper and lower bound switch in magnitude system
upper_mag[:,1]  = lum_to_mag(10**lower_l[:,1]) # upper and lower bound switch in magnitude system

# undo log for normalisation
params_m[:,0] = 10**params_m[:,0]; lower_m[:,0] = 10**lower_m[:,0]; upper_m[:,0] = 10**upper_m[:,0]
params_l[:,0] = 10**params_l[:,0]; lower_l[:,0] = 10**lower_l[:,0]; upper_l[:,0] = 10**upper_l[:,0]
params_mag[:,0] = 10**params_mag[:,0]; lower_mag[:,0] = 10**lower_mag[:,0]; upper_mag[:,0] = 10**upper_mag[:,0]

# calculate errors
lower_err_m    = params_m   - lower_m
upper_err_m    = upper_m    - params_m
lower_err_mag  = params_mag - lower_mag
upper_err_mag  = upper_mag  - params_mag

################## SAVE FOR LATEX TABLE #######################################
#%%
def format_to_string(parameter, lower_err, upper_err, redshift, columns):
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
                exponent     += -1
                param_sig_fig = np.around(parameter[r,c]/10**exponent, pres)
            
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

def format_to_string_v2(parameter, lower_err, upper_err, redshift, columns):
    '''
    Alternative version: Just gives ranges instead of values    
    '''
    rows = range(parameter.shape[0])
    cols = range(parameter.shape[1]) 
    
    formatted_DataFrame = pd.DataFrame(index   = rows, columns = cols)
    
    for c in cols:
        for r in rows:
            pres = 1
            range_lower = parameter[r,c]-lower_err[r,c]
            range_upper = parameter[r,c]+upper_err[r,c]            
            
            exponent    = np.format_float_scientific(range_upper, precision = pres)
            _, exponent = exponent.split('e')
            exponent = int(exponent)
            
            if c == 1:  # should be the log_obs_*, looks nicer like this
                exponent     += -1
            
            range_u = np.around(range_upper/10**exponent, pres)
            range_l = np.around(range_lower/10**exponent, pres)
            if range_l == 0:
                range_l = np.around(range_lower/exponent, pres+1)  

            l_str = str(range_l)
            u_str = str(range_u)
            #import pdb; pdb.set_trace()
            
            if len(l_str.split('.')[1]) < pres:
                l_str = l_str + '0'
            if len(u_str.split('.')[1]) < pres:
                u_str = u_str + '0'
            
            if (range_u<0) and (range_l<0):
                string = r'$-\left[' + l_str[1:] + ' \text{-} ' + u_str[1:] + '\right]$'  
            elif (range_u>0) and (range_l>0):
                string = r'$\left[' + l_str + ' \text{-} ' + u_str + '\right]$' 
            else:
                raise ValueError('Whoops, I was to lazy to implement that case')
            if exponent != 0:
                string = string[:-1] + r' \cdot 10^{' + str(exponent) + r'}$'
                     
            formatted_DataFrame.iloc[r,c] = string
    formatted_DataFrame.insert(0, 'z', redshift)
    formatted_DataFrame.columns = columns
    return(formatted_DataFrame)           


header_m = [r'$z$', r'$\phi_*$ [cMpc$^{-1}$ dex$^{-1}$]', r'$\log M_*$ [$M_\odot$]',
            r'$\alpha$']
header_l = [r'$z$', r'$\phi_*$ [cMpc$^{-1}$ dex$^{-1}$]',
            r'$\mathcal{M}^\mathrm{UV}_{*}$ (mag)', r'$\alpha$']
#header_l = [r'$z$', r'$\phi_*$ [cMpc$^{-1}$ dex$^{-1}$]',
#            r'$\log L_*$ [ergs s${^-1}$ Hz$^{-1}$]', r'$\alpha$']

mstar_table    = format_to_string_v2(params_m, lower_err_m, upper_err_m, redshift, header_m)
lum_table      = format_to_string_v2(params_mag, lower_err_mag, upper_err_mag, redshift, header_l)

# turn into text that can be used as latex code
mstar_latex = mstar_table.to_latex(index = False, escape=False, column_format = 'rrrr',
                                   caption = r'Likely Schechter parameters ranges for SMF given by 16th and 84th percentile.')
lum_latex   = lum_table.to_latex(  index = False, escape=False, column_format = 'rrrr',
                                   caption = r'Likely Schechter parameters ranges for UVLF given by 16th and 84th percentile.')

all_tables_latex  = mstar_latex + lum_latex
