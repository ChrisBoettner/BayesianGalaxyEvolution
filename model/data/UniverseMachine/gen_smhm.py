#!/usr/bin/python
import sys
import math
import re

import numpy as np
from scipy.interpolate import interp1d

path = 'model/data/UniverseMachine/'

def universemachine_smf(mstar_model, log_mstar, z):
    log_m_h = universemachine_mstar_mh(log_mstar, z)
    
    log_phi_contr = mstar_model.calculate_log_hmf(log_m_h, z)
    log_feedback_contr = np.log10(universemachine_mstar_mh(log_mstar, z,
                                                           der=True))
    return(log_phi_contr-log_feedback_contr)
    
    

def universemachine_mstar_mh(log_mstar, z, der=False):
    '''
    Calculate the stellar mass - halo mass relation m_halo(_star) based on the 
    data obtained by Universe Machine.
    
    Parameters
    ----------
    log_quantity : float or array
        Input (log of) observable quantity.
    z : int
        Redshift at which value is calculated.
    der : int, optional
        Degree of derivative at which function is evaluated. der=False gives 
        spline itself, der=True first derivative.

    Returns
    -------
    out : float or array
        Calculated halo mass or derivative of halo mass - stellar mass relation.
    
    
    '''
    # calculate stellar mass - halo mass relation from UniverseMachine data
    data = gen_smhm(z)
    
    log_m_h_f = interp1d(data[:,1], data[:,0])
    
    if not der:
        log_m_h = log_m_h_f(log_mstar)
        return(log_m_h)
    if der:
        dx = 0.001
        log_m_h_der = (log_m_h_f(log_mstar+dx)-log_m_h_f(log_mstar))/dx
        return(log_m_h_der)

def gen_smhm(z):
    # if (len(sys.argv) < 3):
    #     print("Usage: %s z smhm_parameter_file.txt" % sys.argv[0])
    #     quit()
    
    #z = sys.argv[1]
    
    # if (re.search('^uncertainties', 'smhm_true_med_params.txt')):
    #     #print("Use gen_smhm_uncertainties.py instead for uncertainties_* files.");
    #     quit()
    
    #Load params
    param_file = open(path + 'smhm_true_med_params.txt', "r")
    param_list = []
    allparams = []
    for line in param_file:
        param_list.append(float((line.split(" "))[1]))
        allparams.append(line.split(" "))
    
    if (len(param_list) != 20):
        #print("Parameter file not correct length.  (Expected 20 lines, got %d)." % len(param_list))
        quit()
    
    names = "EFF_0 EFF_0_A EFF_0_A2 EFF_0_Z M_1 M_1_A M_1_A2 M_1_Z ALPHA ALPHA_A ALPHA_A2 ALPHA_Z BETA BETA_A BETA_Z DELTA GAMMA GAMMA_A GAMMA_Z CHI2".split(" ");
    params = dict(zip(names, param_list))
    
    #Decide whether to print tex or evaluate SMHM parameter
    try:
        z = float(z)
    except:
        ##print TeX
        for x in allparams[0:10:1]:
            x[3] = -float(x[3])
            sys.stdout.write('& $%.3f^{%+.3f}_{%+.3f}$' % tuple(float(y) for y in x[1:4]))
        sys.stdout.write("\\\\\n & & & ")
        for x in allparams[10:19:1]:
            x[3] = -float(x[3])
            sys.stdout.write('& $%.3f^{%+.3f}_{%+.3f}$' % tuple(float(y) for y in x[1:4]))
    #    sys.stdout.write("\\\\\n & & & ")    
    #    for x in allparams[16:19:1]:
    #        x[3] = -float(x[3])
    #        sys.stdout.write('& $%.3f^{%+.3f}_{%+.3f}$' % tuple(float(y) for y in x[1:4]))
        sys.stdout.write(' & %.0f' % float(allparams[19][1]))
        if (float(allparams[19][1])>200):
            sys.stdout.write('$\dag$')
        #print('\\\\[2ex]')
        quit()

    #Print SMHM relation
    a = 1.0/(1.0+z)
    a1 = a - 1.0
    lna = math.log(a)
    zparams = {}
    zparams['m_1'] = params['M_1'] + a1*params['M_1_A'] - lna*params['M_1_A2'] + z*params['M_1_Z']
    zparams['sm_0'] = zparams['m_1'] + params['EFF_0'] + a1*params['EFF_0_A'] - lna*params['EFF_0_A2'] + z*params['EFF_0_Z']
    zparams['alpha'] = params['ALPHA'] + a1*params['ALPHA_A'] - lna*params['ALPHA_A2'] + z*params['ALPHA_Z']
    zparams['beta'] = params['BETA'] + a1*params['BETA_A'] + z*params['BETA_Z']
    zparams['delta'] = params['DELTA']
    zparams['gamma'] = 10**(params['GAMMA'] + a1*params['GAMMA_A'] + z*params['GAMMA_Z'])
    
    smhm_max = 14.5-0.35*z
    #print('#Log10(Mpeak/Msun) Log10(Median_SM/Msun) Log10(Median_SM/Mpeak)')
    #print('#Mpeak: peak historical halo mass, using Bryan & Norman virial overdensity.')
    #print('#Overall fit chi^2: %f' % params['CHI2'])
    if (params['CHI2']>200):
        print('#Warning: chi^2 > 200 implies that not all features are well fit.  Comparison with the raw data (in data/smhm/median_raw/) is crucial.')
    
    data = []
    for m in np.arange(5,20,0.01):
        dm = m-zparams['m_1'];
        dm2 = dm/zparams['delta'];
        sm = zparams['sm_0'] - math.log10(10**(-zparams['alpha']*dm) + 10**(-zparams['beta']*dm)) + zparams['gamma']*math.exp(-0.5*(dm2*dm2));
        #print("%.2f %.6f %.6f" % (m,sm,sm-m))
        data.append([m,sm,sm-m])
        
    return(np.array(data))

    
