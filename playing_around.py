#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 13:41:12 2022

@author: chris
"""
from model.interface import load_model, run_model, save_model
from model.plotting.plotting import *

# mstar  = load_model('mstar','changing')
# muv    = load_model('Muv','changing')
# mbh    = load_model('mbh','quasar')
# lbol   = load_model('Lbol', 'eddington')


#%%
#models = [mstar]

# plt.close('all')
# for m in models:
#     Plot_parameter_sample(m, columns='single')
# # path = 'model/data/HMF/'
# # mnl  = np.load(path + 'turnover_mass.npz')

# # for z in mstar.redshift:
# #     qc  = mstar.calculate_quantity_distribution(mstar.quantity_options['log_m_c'], z)
# #     qnl = mstar.calculate_quantity_distribution(mnl[str(z)], z) 
# #     print(np.percentile(qc, 50), np.percentile(qnl, 50), mstar.log_ndfs.at_z(z)[-1,0])

# Plot_ndf_intervals(mstar)

# deltas = [0, 0.1, 0.2, 0.3, 0.4]

# plt.figure()
# for d in deltas:
#     ndfs = mstar.calculate_ndf(0, [-2, 1, d])
#     plt.plot(*ndfs, label=f'{d}')
#     plt.legend()

# plt.figure()
# mh = np.linspace(8,15)

# for z in range(4):
    
#     q  = mstar.physics_model.at_z(z).calculate_log_quantity(mh, *mstar.parameter.at_z(z))
    
#     plt.plot(mh, q-mh, label=str(z))
#     plt.legend()


#%%
# from model.data.UniverseMachine.gen_smhm import universemachine_smf, universemachine_mstar_mh

# data = np.load('model/data/UniverseMachine/UniverseMachine_SMF_z0.npy')
# plt.close('all')
# log_mstar = np.linspace(7.8,12,100)

# #smhm rel
# my = mstar.physics_model.at_z(0).calculate_log_halo_mass(log_mstar, *mstar.parameter.at_z(0))
# um = universemachine_mstar_mh(log_mstar, 0)

# plt.semilogy(my, 10**(log_mstar-my), label='my')
# plt.semilogy(um, 10**(log_mstar-um), label='um')
# plt.legend()


# # smf
# mysmf = mstar.calculate_ndf(0, mstar.parameter.at_z(0), quantity_range=log_mstar)
# umsmf = universemachine_smf(mstar, log_mstar, 0)

# plt.figure()
# plt.plot(log_mstar, mysmf[1])
# plt.plot(log_mstar, umsmf)
# plt.scatter(data[:,0], np.log10(data[:,1]), color='black')

# o = universemachine_smf(mstar, log_mstar, 1)
# i = mstar.calculate_ndf(2, mstar.parameter.at_z(2))

#%%
# mstar = save_model('mstar', 'stellar_blackhole', fixed_m_c=False)
# muv   = save_model('Muv', 'stellar_blackhole', fixed_m_c=False)
# mbh   = save_model('mbh', 'quasar')
# lbol  = save_model('Lbol', 'eddington')

# from make_plots import make_all_plots
# make_all_plots()

# print('make plots, see if stuff still works, especially for bhs')

# Plot_parameter_sample(mstar, 
#                       marginalise=(mstar.quantity_options['feedback_change_z'],
#                       [1,2]))
# Plot_qhmr(mstar, redshift=[0,1,2], sigma=2, , columns='single')
# Plot_qhmr(mstar, redshift=[4,5,6,7,8], sigma=2, columns='single',
#           only_data=True, m_halo_range=np.linspace(10, 13.43, 1000))

mstar = load_model('mstar', 'stellar_blackhole')
muv   = load_model('Muv', 'stellar_blackhole')
mbh   = load_model('mbh', 'quasar')
lbol  = load_model('Lbol', 'eddington')

#%%
# vals where agn becomes important, put in table

# model = mstar

# for z in model.redshift:
#     if z<model.quantity_options['feedback_change_z']:
#         z_i = z
#         par  = model.draw_parameter_sample(z_i, 1000)
#     else:
#         z_i = model.quantity_options['feedback_change_z']-1
#         par = model.draw_parameter_sample(z, 1000)
#         # par[:,0] = model.parameter.at_z(z_i)[0]  
#         # par[:,-1] = model.parameter.at_z(z_i)[-1] 
#     q_c = []
#     phi_c = []
#     for p in par:
#         q = model.calculate_feedback_regimes(z, log_epsilon=-1, 
#                                              parameter=p)[-1]
#         q_c.append(q)
#         phi_c.append(model.calculate_log_abundance(q, z, p, hmf_z=z))
#     q_median = np.nanmedian(q_c)
#     phi_median = np.nanmedian(phi_c)  
#     print(q_median, phi_median)

# print('changed feedback_z for mstar and muv')
# print('https://academic.oup.com/mnras/article/357/1/82/1039256')

#%%
# from model.analysis.calculations import calculate_expected_black_hole_mass_from_ERDF

# for z in range(8):
#     mbh_dict = calculate_expected_black_hole_mass_from_ERDF(lbol, np.linspace(43,52,100),
#                                                             z, sigma=1)
#     plt.plot(mbh_dict[1][:,0],mbh_dict[1][:,1], label=str(z), linewidth=5)

#%%
# model = muv

# vals = []

# for z in model.redshift:
#     par_sample = model.draw_parameter_sample(z,500)
#     o = []
#     for p in par_sample:
#         o.append(model.calculate_total_number_density(z, p, num=500))
#     vals.append(np.median(o))
    
# #vals = np.log10(np.array(vals) * 1.4 * 1e-28)
# vals = np.log10(np.array(vals))

# from model.helper import z_to_t
# plt.plot(model.redshift, np.log10(vals))


#%%
print('make ylimits work')
print('choose smart way to select Lbol limits')
print('num=int(1e+4) for all')
Plot_quantity_density_evolution(mstar, columns='single')
Plot_quantity_density_evolution(muv,   columns='single')
Plot_quantity_density_evolution(mbh,   columns='single')
Plot_quantity_density_evolution(lbol,  columns='single')

Plot_stellar_mass_density_evolution(mstar, muv, num_samples=int(1e+4),
                                    columns='single')
