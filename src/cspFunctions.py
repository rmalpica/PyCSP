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
        self.tau = []
        self._changed = False

    def __setattr__(self, key, value):
            if key != '_changed':
                self._changed = True
            super().__setattr__(key, value)

    def is_changed(self):
            return self._changed
                
    def update_kernel(self, jacobian):
        if jacobian == 'analytic':    
            self.evals,self.Revec,self.Levec,self.f = self.kernel_pyJac()            
        elif jacobian == 'numeric':
            self.evals,self.Revec,self.Levec,self.f = self.kernel()
        else:
            print("Invalid jacobian keyword")
        
        self.tau = timescales(self.eval)
            
        self._changed = False
        
    def get_kernel(self, jacobian):
        if self.is_changed(): 
            self.update_kernel(jacobian)
            self._changed = False
        return [self.evals,self.Revec,self.Levec,self.f]
        

        """ ~~~~~~~~~~~~ KERNEL ~~~~~~~~~~~~~
        """
    def kernel_pyJac(self):
        """Computes CSP kernel, using PyJac to get RHS and analytic Jacobian.
        Returns [evals,Revec,Levec,amplitudes].
        Input must be an instance of the CSPCantera class"""
        
        ydot = self.rhs_const_p_pyJac()
        
        jac2D = self.jac_pyJac()
        
        #eigensystem
        evals,Revec,Levec = eigsys(jac2D)
        f = np.matmul(Levec,ydot)
        
        #rotate eigenvectors such that amplitudes are positive    
        Revec,Levec,f = evec_pos_ampl(Revec,Levec,f)
        
        return[evals,Revec,Levec,f]
    
    
    def kernel(self):
        """Computes CSP kernel numerically.
        Returns [evals,Revec,Levec,amplitudes]. 
        Input must be an instance of the CSPCantera class"""
    
        ydot = self.rhs_const_p()
        
        jac2D = self.jac_numeric()
        
        #eigensystem
        evals,Revec,Levec = eigsys(jac2D)
        f = np.matmul(Levec,ydot)
        
        #rotate eigenvectors such that amplitudes are positive    
        Revec,Levec,f = evec_pos_ampl(Revec,Levec,f)
        
        return[evals,Revec,Levec,f]

    """ ~~~~~~~~~~~~ JACOBIAN ~~~~~~~~~~~~~
    """
    def jac_pyJac(self):
        """Computes analytic Jacobian, using PyJac.
        Returns a 2D array [jac]. 
        Input must be an instance of the CSPCantera class"""
        #setup the state vector
        y = np.zeros(self.n_species)
        y[0] = self.T
        y[1:] = self.Y[:-1]
        
        #create a jacobian vector
        jac = np.zeros(self.n_species * self.n_species)
        
        #evaluate the Jacobian
        pj.py_eval_jacobian(0, self.P, y, jac)
        
        #reshape as 2D array
        jac2D = jac.reshape(self.n_species,self.n_species).transpose()
        
        return jac2D

    def jac_numeric(self):
        """Computes numerical Jacobian.
        Returns a 2D array [jac]. Input must be an instance of the CSPCantera class"""
        roundoff = np.finfo(float).eps
        sro = np.sqrt(roundoff)
        #setup the state vector
        T = self.T
        y = self.Y   #ns-long
        ydot = self.rhs_const_p()   #ns-long (T,Y1,...,Yn-1)
        
        #create a jacobian vector
        jac = np.zeros(self.n_species * self.n_species)
        jac2D = jac.reshape(self.n_species,self.n_species)
        
        #evaluate the Jacobian
        for i in range(self.n_species-1):
            dy = np.zeros(self.n_species)
            dy[i] = max(sro*abs(y[i]),1e-8)
            self.set_unnormalized_mass_fractions(y+dy)
            ydotp = self.rhs_const_p()
            dydot = ydotp-ydot
            jac2D[:,i+1] = dydot/dy[i]
        
        self.Y = y

        dT = max(sro*abs(T),1e-3)
        self.TP = T+dT,self.P
        ydotp = self.rhs_const_p()
        dydot = ydotp-ydot
        jac2D[:,0] = dydot/dT
        
        self.TP = T,self.P
           
        return jac2D

    """ ~~~~~~~~~~~~ RHS ~~~~~~~~~~~~~
    """
    def rhs_const_p(self):
        """Computes chemical RHS [shape:(ns)]. 
        Input must be an instance of the CSPCantera class"""
        
        ns = self.n_species    
        ydot = np.zeros(ns+1)
        Wk = self.molecular_weights
        R = ct.gas_constant
        
        wdot = self.net_production_rates
        orho = 1./self.density
        
        ydot[0] = - R * self.T * np.dot(self.standard_enthalpies_RT, wdot) * orho / self.cp_mass
        ydot[1:] = wdot * Wk * orho
        
        return ydot[:-1]
    
    
    def rhs_const_p_pyJac(self):
        """Retrieves chemical RHS from PyJac [shape(ns)]. 
        Input must be an instance of the CSPCantera class"""
        
        #setup the state vector
        y = np.zeros(self.n_species)
        y[0] = self.T
        y[1:] = self.Y[:-1]
        
        #create ydot vector
        ydot = np.zeros_like(y)
        pj.py_dydt(0, self.P, y, ydot)
           
        return ydot
    
""" ~~~~~~~~~~~~ EIGEN ~~~~~~~~~~~~~
"""

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
    """changes sign to eigenvectors based on sign of corresponding mode amplitude."""
    idx = np.flatnonzero(f < 0)
    Revec[:,idx] = - Revec[:,idx]
    Levec[idx,:] = - Levec[idx,:]
    f[idx] = -f[idx]
    
    return[Revec,Levec,f]


def timescales(evals):
    tau = [1.0/abslam if abslam > 0 else 1e-20 for abslam in np.absolute(evals)]
    
    return tau



""" ~~~~~~~~~~~~ EXHAUSTED MODES ~~~~~~~~~~~~~
"""

"""
def findM(self):
    np = len(self.Revec)
    nel = self.n_elements - 1   #-1 accounts for removed N2 in jacobian calc
    nconjpairs = sum(1 for x in self.eval.imag if x != 0)/2
"""    
    
