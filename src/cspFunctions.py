#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Riccardo Malpica Galassi, Sapienza University, Roma, Italy
"""
import sys
import numpy as np
import cantera as ct
from .cspThermoKinetics import CanteraThermoKinetics


class CanteraCSP(CanteraThermoKinetics):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  
        self.nv = 0
        self.jacobiantype = 'numeric'
        self.rtol = 1.0e-2
        self.atol = 1.0e-8
        self.rhs = []
        self.jac = []
        self.evals = []
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

                
    def update_kernel(self, jacobiantype):
        """Computes the CSP kernel"""
        jacobiantype = self.jacobiantype
        if jacobiantype == 'analytic': 
            self.evals,self.Revec,self.Levec,self.f = self.kernel_pyJac()            
        elif jacobiantype == 'numeric':
            self.evals,self.Revec,self.Levec,self.f = self.kernel()
        else:
            print("Invalid jacobian keyword")
            sys.exit()
        self.tau = timescales(self.evals)
        self._changed = False
 
        
    def get_kernel(self, **kwargs):
        """Retrieves the stored CSP kernel.
        Optional argument is jacobiantype.
        If provided and different from stored value, kernel is recomputed"""
        jacobiantype = self.jacobiantype
        for key, value in kwargs.items():
            if (key == 'jacobiantype'): 
                jacobiantype = value
            else:
                sys.exit("unknown argument %s" %key)
        if (self.jacobiantype != jacobiantype): self.jacobiantype = jacobiantype
        if self.is_changed(): 
            self.update_kernel(self.jacobiantype)
            self._changed = False
        return [self.evals,self.Revec,self.Levec,self.f]
 
    
    def calc_exhausted_modes(self, **kwargs):
        """Computes number of exhausted modes (M). 
        Optional arguments are rtol and atol for the calculation of M.
        If not provided, uses default or previously set values"""
        rtol = self.rtol
        atol = self.atol
        for key, value in kwargs.items():
            if (key == 'rtol'): 
                rtol = value
            elif (key == 'atol'): 
                atol = value
            else:
                sys.exit("unknown argument %s" %key)
        M = self.findM(rtol,atol)
        return M
 
       
    def calc_TSR(self,**kwargs):
        """Computes number of exhausted modes (M) and the TSR. 
        Optional arguments are rtol and atol for the calculation of M.
        If not provided, uses default or previously set values.
        The calculated value of M can be retrieved by passing
        the optional argument getM=True"""
        rtol = self.rtol
        atol = self.atol
        getM = False
        for key, value in kwargs.items():
            if (key == 'rtol'): 
                rtol = value
            elif (key == 'atol'): 
                atol = value
            elif (key == 'getM'): 
                getM = value
            else:
                sys.exit("unknown argument %s" %key)
        M = self.findM(rtol, atol)
        TSR = self.findTSR(M)
        if getM:
            return [TSR, M]
        else:
            return TSR
        
 
    
        """ ~~~~~~~~~~~~ STATE ~~~~~~~~~~~~~
        """
    def stateTY(self):
        y = np.zeros(self.n_species+1)
        y[0] = self.T
        y[1:] = self.Y
        return y
    
    
    def set_stateTY(self,y):
        self.Y = y[1:]
        self.TP = y[0],self.P
        

        """ ~~~~~~~~~~~~ KERNEL ~~~~~~~~~~~~~
        """
    def kernel_pyJac(self):
        """Computes CSP kernel, using PyJac to get RHS and analytic Jacobian.
        Returns [evals,Revec,Levec,amplitudes].
        Input must be an instance of the CSPCantera class"""
        
        self.nv = self.n_species 
        
        self.rhs = self.rhs_const_p_pyJac()
        
        self.jac = self.jac_pyJac()
        
        #eigensystem
        evals,Revec,Levec = eigsys(self.jac)
        f = np.matmul(Levec,self.rhs)
        
        #rotate eigenvectors such that amplitudes are positive    
        Revec,Levec,f = evec_pos_ampl(Revec,Levec,f)
        
        return[evals,Revec,Levec,f]
    
    
    def kernel(self):
        """Computes CSP kernel numerically.
        Returns [evals,Revec,Levec,amplitudes]. 
        Input must be an instance of the CSPCantera class"""
    
        self.nv = self.n_species     
        
        self.rhs = self.rhs_const_p()
        
        self.jac = self.jac_numeric()
        
        #eigensystem
        evals,Revec,Levec = eigsys(self.jac)
        f = np.matmul(Levec,self.rhs)
        
        #rotate eigenvectors such that amplitudes are positive    
        Revec,Levec,f = evec_pos_ampl(Revec,Levec,f)
        
        return[evals,Revec,Levec,f]


    
    """ ~~~~~~~~~~~~ EXHAUSTED MODES ~~~~~~~~~~~~~
    """
    def findM(self,rtol,atol):
        nv = len(self.Revec)
        nEl = self.n_elements - 1   #-1 accounts for removed inert in jacobian calc
        #nconjpairs = sum(1 for x in self.eval.imag if x != 0)/2
        imPart = self.evals.imag!=0
        nModes = nv - nEl
        ewt = setEwt(self.stateTY(),rtol,atol)
        
        delw = np.zeros(nv)
        for j in range(nModes-1):          #loop over eligible modes (last excluded)
            taujp1 = self.tau[j+1]              #timescale of next (slower) mode
            Aj = self.Revec[j]                  #this mode right evec
            fj = self.f[j]                      #this mode amplitued
            lamj = self.evals[j].real           #this mode eigenvalue (real part)
            
            for i in range(nv):
                Aji = Aj[i]                     #i-th component of this mode Revec
                delw[i] = delw[i] + modeContribution(Aji,fj,taujp1,lamj)    #contribution of j-th mode to i-th var            
                if np.abs(delw[i]) > ewt[i]:
                    if j==0:
                        M = 0
                    else:
                        M = j-1 if (imPart[j] and imPart[j-1]) else j    #if j is the second of a pair, move back by 2                    
                    return M
    
        #print("All modes are exhausted")
        M = nModes - 1   #if criterion is never verified, all modes are exhausted. Leave 1 active mode.
        return M
        

    """ ~~~~~~~~~~~~~~ TSR ~~~~~~~~~~~~~~~~
    """
    
    def findTSR(self,M):
        n = len(self.Revec)
        nEl = self.n_elements - 1  #-1 accounts for removed inert in jacobian calc
        #deal with amplitudes of cmplx conjugates
        fvec = self.f.copy()
        imPart = self.evals.imag!=0
        for i in range(1,n):
            if (imPart[i] and imPart[i-1]):
                fvec[i] = np.sqrt(fvec[i]**2 + fvec[i-1]**2)
                fvec[i-1] = fvec[i]
        fnorm = fvec / np.linalg.norm(self.rhs)
        #deal with zero-eigenvalues (if any)
        fvec[self.evals==0.0] = 0.0
        weights = fnorm**2
        weights[0:M] = 0.0 #excluding fast modes
        weights[n-nEl:n] = 0.0 #excluding conserved modes
        normw = np.sum(weights)
        weights = weights / normw if normw > 0 else np.zeros(n)
        TSR = np.sum([weights[i] * np.sign(self.evals[i].real) * np.abs(self.evals[i]) for i in range(n)])
        return TSR
          


def setEwt(y,rtol,atol):   
    ewt = [rtol * absy + atol if absy >= 1.0e-6 else absy + atol for absy in np.absolute(y)]
    return ewt


def modeContribution(a,f,tau,lam):
    delwMi = a*f*(np.exp(tau*lam) - 1)/lam if lam != 0.0 else 0.0
    return delwMi
        
    
    
""" ~~~~~~~~~~~~ EIGEN ~~~~~~~~~~~~~
"""

def eigsys(jac):        
    """Returns eigensystem (evals, Revec, Levec). Input must be a 2D array.
    Both Revec (since it is transposed) and Levec (naturally) contain row vectors,
    such that an eigenvector can be retrieved using R/Levec[index,:] """
    
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
    
    #transpose Revec
    Revec = np.transpose(Revec)

    
    return[evals,Revec,Levec]


def evec_pos_ampl(Revec,Levec,f):
    """changes sign to eigenvectors based on sign of corresponding mode amplitude."""
    idx = np.flatnonzero(f < 0)
    Revec[idx,:] = - Revec[idx,:]
    Levec[idx,:] = - Levec[idx,:]
    f[idx] = -f[idx]
    
    return[Revec,Levec,f]


def timescales(evals):
    tau = [1.0/abslam if abslam > 0 else 1e+20 for abslam in np.absolute(evals)]
    
    return tau





""" ~~~~~~~~~~~~ OTHER JAC FORMULATIONS ~~~~~~~~~~~~~
"""

def jacThermal(gas):
    nv = len(gas.evals)
    R = ct.gas_constant
    hspec = gas.standard_enthalpies_RT
    Hspec = hspec * R * gas.T
    Wk = gas.molecular_weights
    cp = gas.cp_mass
    TJrow = Hspec[:-1] / ( Wk[:-1] * cp)
    TJcol = gas.jac[1:nv,0]
    JacThermal = np.outer(TJcol,TJrow)
    return JacThermal

def jacKinetic(gas):
    nv = len(gas.evals)
    jacKinetic = gas.jac[1:nv,1:nv] 
    return jacKinetic
        
def kernel_kinetic_only(gas):
    nv = len(gas.evals)
    kineticjac = jacKinetic(gas)       
    #eigensystem
    evals,Revec,Levec = eigsys(kineticjac)
    f = np.matmul(Levec,gas.rhs[1:nv])
    #rotate eigenvectors such that amplitudes are positive    
    Revec,Levec,f = evec_pos_ampl(Revec,Levec,f)
    return[evals,Revec,Levec,f]
    
def kernel_constrained_jac(gas):
    nv = len(gas.evals)
    kineticjac = jacKinetic(gas)  
    thermaljac = jacThermal(gas)   
    jac = kineticjac - thermaljac
    #eigensystem
    evals,Revec,Levec = eigsys(jac)
    f = np.matmul(Levec,gas.rhs[1:nv])
    #rotate eigenvectors such that amplitudes are positive    
    Revec,Levec,f = evec_pos_ampl(Revec,Levec,f)
    return[evals,Revec,Levec,f]