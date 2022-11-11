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
# muv = save_model('Muv', 'stellar_blackhole', fixed_m_c=False)
# mbh = save_model('mbh', 'quasar')
# lbol = save_model('Lbol', 'eddington', num_walker=10)

# print('make plots, see if stuff still works, especially for bhs')

# Plot_parameter_sample(mstar, 
#                       marginalise=(mstar.quantity_options['feedback_change_z'],
#                       [1,2]))
# Plot_qhmr(mstar, redshift=[0,1,2], sigma=2, , columns='single')
# Plot_qhmr(mstar, redshift=[4,5,6,7,8], sigma=2, columns='single',
#           only_data=True, m_halo_range=np.linspace(10, 13.43, 1000))

# https://academic.oup.com/mnras/article/357/1/82/1039256

mstar = load_model('mstar', 'stellar_blackhole')
muv  = load_model('Muv', 'stellar_blackhole')
mbh  = load_model('mbh', 'quasar')
# lbol = load_model('Lbol', 'eddington')

#%%
# vals where agn becomes important, put in table
for z in mstar.redshift:
    par  = mstar.draw_parameter_sample(z, 1000)
    q_c = []
    phi_c = []
    for p in par:
        q = mstar.physics_model.at_z(z).calculate_log_quantity(p[0], *p)
        q_c.append(q)
        phi_c.append(mstar.calculate_log_abundance(q, z, p))
    q_median = np.median(q_c)
    phi_median = np.median(phi_c)  
    print(q_median, phi_median)