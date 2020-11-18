#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
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
            super().__setattr__(key, value)

    def is_changed(self):
            return self._changed
                
    def update_kernel(self, jacobian):
        if jacobian == 'analytic':    
            self.evals,self.Revec,self.Levec,self.f = kernel_pyJac(self)            
        elif jacobian == 'numeric':
            self.evals,self.Revec,self.Levec,self.f = kernel(self)
        else:
            print("Invalid jacobian keyword")
            
        self._changed = False
        
    def get_kernel(self, jacobian):
        if self.is_changed(): 
            self.update_kernel(jacobian)
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


def kernel(gas):
    """Computes CSP kernel numerically.
    Returns [evals,Revec,Levec,amplitudes]"""

    ydot = rhs_const_p(gas)
    
    jac2D = jac(gas)
    
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

def jac(gas):
    """Computes numerical Jacobian.
    Returns a 2D array [jac]"""
    roundoff = np.finfo(float).eps
    sro = np.sqrt(roundoff)
    #setup the state vector
    T = gas.T
    y = gas.Y   #ns-long
    ydot = rhs_const_p(gas)   #ns-long (T,Y1,...,Yn-1)
    
    #create a jacobian vector
    jac = np.zeros(gas.n_species * gas.n_species)
    jac2D = jac.reshape(gas.n_species,gas.n_species)
    
    #evaluate the Jacobian
    for i in range(gas.n_species-1):
        dy = np.zeros(gas.n_species)
        dy[i] = max(sro*abs(y[i]),1e-8)
        gas.set_unnormalized_mass_fractions(y+dy)
        ydotp = rhs_const_p(gas)
        dydot = ydotp-ydot
        jac2D[:,i+1] = dydot/dy[i]
    
    gas.Y = y

    dT = max(sro*abs(T),1e-3)
    gas.TP = T+dT,gas.P
    ydotp = rhs_const_p(gas)
    dydot = ydotp-ydot
    jac2D[:,0] = dydot/dT
    
    gas.TP = T,gas.P
       
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
    """Computes chemical RHS [shape:(ns)]"""
    
    ns = gas.n_species    
    ydot = np.zeros(ns+1)
    Wk = gas.molecular_weights
    R = ct.gas_constant
    
    wdot = gas.net_production_rates
    orho = 1./gas.density
    
    ydot[0] = - R * gas.T * np.dot(gas.standard_enthalpies_RT, wdot) * orho / gas.cp_mass
    ydot[1:] = wdot * Wk * orho
    
    return ydot[:-1]


def rhs_const_p_pyJac(gas):
    """Retrieves chemical RHS from PyJac [shape(ns)]"""
    
    #setup the state vector
    y = np.zeros(gas.n_species)
    y[0] = gas.T
    y[1:] = gas.Y[:-1]
    
    #create ydot vector
    ydot = np.zeros_like(y)
    pj.py_dydt(0, gas.P, y, ydot)
       
    return ydot