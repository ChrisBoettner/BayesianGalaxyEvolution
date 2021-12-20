#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 15:05:09 2021

@author: boettner
"""
# little backup with mc fixed

 
class supernova_blackhole_feedback():
    def __init__(self, feedback_name, m_crit):
        self.name          = feedback_name
        self.m_c           = m_crit
        self.initial_guess = [0.01, 1, 0.3]
        
    def calculate_m_star(self, m_h, A, alpha, beta):
        sn = (m_h/self.m_c)**(-alpha)
        bh = (m_h/self.m_c)**beta
        return(A * m_h/(sn + bh))
    
    def calculate_m_h(self, m_star, A, alpha, beta):
        m_star_func = lambda m_halo : self.calculate_m_star(m_halo, A, alpha, beta)
        fprime      = lambda m_halo : self._calculate_dmstar_dmh(m_halo, A, alpha, beta)
        fprime2     = lambda m_halo : self._calculate_d2mstar_dmh2(m_halo, A, alpha, beta)
        m_h = invert_function(m_star_func, fprime, fprime2, m_star) 
        return(m_h)
    def calculate_dlogmstar_dlogmh(self, m_h, A, alpha, beta):
        sn = (m_h/self.m_c)**(-alpha)
        bh = (m_h/self.m_c)**beta
        return(1 - np.log(10) * (-alpha*sn + beta * bh)/(sn + bh))
    def _calculate_dmstar_dmh(self, m_h,A,alpha,beta): # first derivative, just used to calc inverse
        if m_h<0:
            return(np.nan)
        sn = (m_h/self.m_c)**(-alpha)
        bh = (m_h/self.m_c)**beta
        return(A * ((1+alpha)*sn+(1-beta)*bh)/(sn + bh)**2)
    def _calculate_d2mstar_dmh2(self, m_h,A,alpha,beta): # second derivative, just used to calc inverse
        if m_h<0:
            return(np.nan)
        x = m_h/self.m_c; a = alpha; b = beta
        denom = x**(-a) + x**b
        first_num  = (1+a)*(-a)*x**(-a-1)+(1-b)*b*x**(b-1)
        second_num = -2*((1+a)* x**(-a)+(1+b)*x**b)*(-a*x**(-a-1)+b*x**(b-1))
        return(A/self.m_c * (first_num/denom**2 + second_num/denom**3))