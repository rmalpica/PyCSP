#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 09:30:10 2020

@author: Riccardo Malpica Galassi, Sapienza University, Roma, Italy
"""

import numpy as np
import pyjacob as pj
import cantera as ct

class CanteraCSP(ct.Solution):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)        
        self.rhs = []
        self.jac = []
        self.eval = []
        self.Revec = []
        self.Levec = []
        self.f = []
        self._changed = False

    def __setattr__(self, key, value):
            if key != '_changed':
                self._changed = True
            super(CanteraCSP, self).__setattr__(key, value)

    def is_changed(self):
            return self._changed
                
    def update_kernel(self):
        self.evals,self.Revec,self.Levec,self.f = kernel_pyJac(self)
        self._changed = False
        
    def get_kernel(self):
        if self.is_changed(): 
            self.update_kernel()
            self._changed = False
        return [self.evals,self.Revec,self.Levec,self.f]
        

def kernel_pyJac(gas):
    """Computes CSP kernel, using PyJac to get RHS and analytic Jacobian.
    Returns [evals,Revec,Levec,amplitudes]"""
    
    ydot = rhs_const_p_pyJac(gas)
    
    jac2D = jac_pyJac(gas)
    
    #eigensystem
    evals,Revec,Levec = eigsys(jac2D)
    f = np.matmul(Levec,ydot)
    
    #rotate eigenvectors such that amplitudes are positive    
    Revec,Levec,f = evec_pos_ampl(Revec,Levec,f)
    
    return[evals,Revec,Levec,f]


def jac_pyJac(gas):
    """Computes analytic Jacobian, using PyJac.
    Returns a 2D array [jac]"""
    #setup the state vector
    y = np.zeros(gas.n_species)
    y[0] = gas.T
    y[1:] = gas.Y[:-1]
    
    #create a jacobian vector
    jac = np.zeros(gas.n_species * gas.n_species)
    
    #evaluate the Jacobian
    pj.py_eval_jacobian(0, gas.P, y, jac)
    
    #reshape as 2D array
    jac2D = jac.reshape(gas.n_species,gas.n_species).transpose()
    
    return jac2D


def eigsys(jac):        
    """Returns eigensystem (evals, Revec, Levec). Input must be a 2D array"""
    
    evals, Revec = np.linalg.eig(jac)
    
    #sort 
    idx = np.argsort(abs(evals))[::-1]   
    evals = evals[idx]
    Revec = Revec[:,idx]

    #adjust complex conjugates
    cmplx = Revec.imag.any(axis=0)   #boolean indexing of complex eigenvectors (cols)
    icmplx = np.flatnonzero(cmplx)   #indexes of complex eigenvectors
    for i in icmplx[::2]:
        re = (Revec[:,i]+Revec[:,i+1])/2.0
        im = (Revec[:,i]-Revec[:,i+1])/(2.0j)
        Revec[:,i] = re
        Revec[:,i+1] = im
    Revec = Revec.real  #no need to carry imaginary part anymore  

    #compute left eigenvectors, amplitudes
    Levec = np.linalg.inv(Revec)

    
    return[evals,Revec,Levec]


def evec_pos_ampl(Revec,Levec,f):
    """changes sign to eigenvectors based on sign of corresponding mode amplitude"""
    idx = np.flatnonzero(f < 0)
    Revec[:,idx] = - Revec[:,idx]
    Levec[idx,:] = - Levec[idx,:]
    f[idx] = -f[idx]
    
    return[Revec,Levec,f]


def rhs_const_p(gas):
    """Computes chemical RHS"""
    
    ns = gas.n_species    
    ydot = np.zeros(ns+1)
    Wk = gas.molecular_weights
    R = ct.gas_constant
    
    wdot = gas.net_production_rates
    orho = 1./gas.density
    
    ydot[0] = - R * gas.T * np.dot(gas.standard_enthalpies_RT, wdot) * orho / gas.cp_mass
    ydot[1:] = wdot * Wk * orho
    
    return ydot


def rhs_const_p_pyJac(gas):
    """Retrieves chemical RHS from PyJac"""
    
    #setup the state vector
    y = np.zeros(gas.n_species)
    y[0] = gas.T
    y[1:] = gas.Y[:-1]
    
    #create ydot vector
    ydot = np.zeros_like(y)
    pj.py_dydt(0, gas.P, y, ydot)
       
    return ydot