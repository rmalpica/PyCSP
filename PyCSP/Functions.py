#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Riccardo Malpica Galassi, Sapienza University, Roma, Italy
"""

import numpy as np
from .ThermoKinetics import CanteraThermoKinetics


class CanteraCSP(CanteraThermoKinetics):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  
        self.jacobiantype = 'full'
        self.rtol = 1.0e-2
        self.atol = 1.0e-8
        self._rhs = []
        self._jac = []
        self._evals = []
        self._Revec = []
        self._Levec = []
        self._f = []
        self._tau = []
        self._nUpdates = 0
        self._changed = False
    
    @property
    def jacobiantype(self):
        return self._jacobiantype
          
    @jacobiantype.setter
    def jacobiantype(self,value):
        if value == 'full' or value == 'kinetic' or value == 'constrained':
            self._jacobiantype = value
        else:
            raise ValueError("Invalid jacobian type --> %s" %value)
            
    @property
    def rtol(self):
        return self._rtol
          
    @rtol.setter
    def rtol(self,value):
        self._rtol = value
        
    @property
    def atol(self):
        return self._atol
          
    @atol.setter
    def atol(self,value):
        self._atol = value
        
    @property
    def rhs(self):
        return self._rhs
    
    @property
    def jac(self):
        return self._jac
    
    @property
    def evals(self):
        return self._evals
    
    @property
    def Revec(self):
        return self._Revec
    
    @property
    def Levec(self):
        return self._Levec
    
    @property
    def f(self):
        return self._f
    
    @property
    def tau(self):
        return self._tau
    
    @property
    def nUpdates(self):
        return self._nUpdates
    
    def __setattr__(self, key, value):
        if key != '_changed':
            self._changed = True
        super().__setattr__(key, value)


    def is_changed(self):
        return self._changed

            
    def update_kernel(self):
        if self.jacobiantype == 'full':
            self._evals,self._Revec,self._Levec,self._f = self.kernel()
        elif self.jacobiantype == 'constrained':
            self._evals,self._Revec,self._Levec,self._f = self.kernel_constrained_jac()
        elif self.jacobiantype == 'kinetic':
            self._evals,self._Revec,self._Levec,self._f = self.kernel_kinetic_only()
        self._tau = timescales(self.evals)
        self._changed = False
 
        
    def get_kernel(self, **kwargs):
        """Retrieves the stored CSP kernel.
        Optional argument is jacobiantype.
        If provided and different from stored value, kernel is recomputed"""
        for key, value in kwargs.items():
            if (key == 'jacobiantype'): 
                if (self.jacobiantype != value): self.jacobiantype = value
            else:
                raise ValueError("unknown argument --> %s" %key)
        if self.is_changed(): 
            self.update_kernel()
            self._changed = False
        return [self.evals,self.Revec,self.Levec,self.f]
 
    
    def calc_exhausted_modes(self, **kwargs):
        """Computes number of exhausted modes (M). 
        Optional arguments are rtol and atol for the calculation of M.
        If not provided, uses default or previously set values"""
        for key, value in kwargs.items():
            if (key == 'rtol'):                 
                if (self.rtol != value): self.rtol = value
            elif (key == 'atol'): 
                if (self.atol != value): self.atol = value
            else:
                raise ValueError("unknown argument --> %s" %key)
        if self.is_changed(): 
            self.update_kernel()
            self._changed = False
        M = findM(self.n_elements,self.stateYT(),self.evals,self.Revec,self.tau,self.f,self.rtol,self.atol)
        return M
 
       
    def calc_TSR(self,**kwargs):
        """Computes number of exhausted modes (M) and the TSR. 
        Optional arguments are rtol and atol for the calculation of M.
        If not provided, uses default or previously set values.
        The calculated value of M can be retrieved by passing
        the optional argument getM=True"""
        getM = False
        for key, value in kwargs.items():
            if (key == 'rtol'): 
                if (self.rtol != value): self.rtol = value
            elif (key == 'atol'): 
                if (self.atol != value): self.atol = value
            elif (key == 'getM'): 
                getM = value
            else:
                raise ValueError("unknown argument --> %s" %key)
        if self.is_changed(): 
            self.update_kernel()
            self._changed = False
        M = findM(self.n_elements,self.stateYT(),self.evals,self.Revec,self.tau,self.f,self.rtol,self.atol)
        TSR, weights = findTSR(self.n_elements,self.rhs,self.evals,self.Revec,self.f,M)
        if getM:
            return [TSR, M]
        else:
            return TSR

    def calc_TSRindices(self,**kwargs):
        """Computes number of exhausted modes (M), the TSR and its indices. 
        Optional argument is type, which can be timescale or amplitude.
        Default value is amplitude.
        Other optional arguments are rtol and atol for the calculation of M.
        If not provided, uses default or previously set values.
        The calculated value of M can be retrieved by passing
        the optional argument getM=True"""
        useTPI = False
        for key, value in kwargs.items():
            if (key == 'rtol'): 
                if (self.rtol != value): self.rtol = value
            elif (key == 'atol'): 
                if (self.atol != value): self.atol = value
            elif (key == 'type'): 
                if(value == 'timescale'):
                    useTPI = True
                elif(value != 'amplitude'):
                    raise ValueError("unknown type --> %s" %value)
            else:
                raise ValueError("unknown argument --> %s" %key)
        if self.is_changed(): 
            self.update_kernel()
            self._changed = False
        Smat = self.generalized_Stoich_matrix
        rvec = self.R_vector
        M = findM(self.n_elements,self.stateYT(),self.evals,self.Revec,self.tau,self.f,self.rtol,self.atol)
        TSR, weights = findTSR(self.n_elements,self.rhs,self.evals,self.Revec,self.f,M)
        TSRidx = TSRindices(weights, self.evals)
        if (useTPI):
            JacK = self.jac_contribution()
            CSPidx = CSP_timescale_participation_indices(self.n_reactions, JacK, self.evals, self.Revec, self.Levec)
        else:
            CSPidx = CSP_amplitude_participation_indices(self.Levec, Smat, rvec)
        TSRind = TSR_participation_indices(TSRidx, CSPidx)
        return TSRind
        
        
    def calc_CSPindices(self,**kwargs):
        """Computes number of exhausted modes (M) and the CSP Indices. 
        Optional arguments are rtol and atol for the calculation of M.
        If not provided, uses default or previously set values.
        The calculated value of M can be retrieved by passing
        the optional argument getM=True"""
        getM = False
        getAPI = False
        getImpo = False
        getspeciestype = False
        getTPI = False
        API = None
        Ifast = None
        Islow = None
        species_type = None
        TPI = None
        for key, value in kwargs.items():
            if (key == 'rtol'): 
                if (self.rtol != value): self.rtol = value
            elif (key == 'atol'): 
                if (self.atol != value): self.atol = value
            elif (key == 'getM'): 
                getM = value
            elif (key == 'API'): 
                getAPI = value
            elif (key == 'Impo'): 
                getImpo = value
            elif (key == 'species_type'): 
                getspeciestype = value
            elif (key == 'TPI'): 
                getTPI = value
            else:
                raise ValueError("unknown argument --> %s" %key)
        if self.is_changed(): 
            self.update_kernel()
            self._changed = False
        M = findM(self.n_elements,self.stateYT(),self.evals,self.Revec,self.tau,self.f,self.rtol,self.atol)
        Smat = self.generalized_Stoich_matrix
        rvec = self.R_vector
        if getAPI: API = CSP_amplitude_participation_indices(self.Levec, Smat, rvec)
        if getImpo: Ifast,Islow = CSP_importance_indices(self.Revec,self.Levec,M,Smat,rvec)
        if getspeciestype: 
            pointers = CSP_pointers(self.Revec,self.Levec)
            species_type = classify_species(self.stateYT(), self.rhs, pointers, M)
        if getTPI:
            JacK = self.jac_contribution()
            TPI = CSP_timescale_participation_indices(self.n_reactions, JacK, self.evals, self.Revec, self.Levec)
        if getM:
            return [API, TPI, Ifast, Islow, species_type, M]
        else:
            return [API, TPI, Ifast, Islow, species_type]
        

        

        """ ~~~~~~~~~~~~ KERNEL ~~~~~~~~~~~~~
        """    
   
    def kernel(self):
        """Computes CSP kernel. Its dimension is Nspecies + 1.
        Returns [evals,Revec,Levec,amplitudes]. 
        Input must be an instance of the CSPCantera class"""
    
        self.nv = self.n_species + 1     
        self._rhs = self.source.copy()
        self._jac = self.jacobian.copy()
        #eigensystem
        evals,Revec,Levec = eigsys(self.jac)
        f = np.matmul(Levec,self.rhs)
        self._nUpdates = self._nUpdates + 1
        #rotate eigenvectors such that amplitudes are positive    
        Revec,Levec,f = evec_pos_ampl(Revec,Levec,f)
        
        return[evals,Revec,Levec,f]
    
    
    def kernel_kinetic_only(self):
        """Computes kinetic kernel. Its dimension is Nspecies.
        Returns [evals,Revec,Levec,amplitudes]. 
        Input must be an instance of the CSPCantera class"""
        self.nv = self.n_species
        self._rhs = self.source.copy()[:self.nv]
        self._jac = self.jacKinetic().copy()       
        #eigensystem
        evals,Revec,Levec = eigsys(self.jac)
        f = np.matmul(Levec,self.rhs)
        self._nUpdates = self._nUpdates + 1
        #rotate eigenvectors such that amplitudes are positive    
        Revec,Levec,f = evec_pos_ampl(Revec,Levec,f)
        return[evals,Revec,Levec,f]
    
    
    def kernel_constrained_jac(self):
        """Computes constrained (to enthalpy) kernel. Its dimension is Nspecies .
        Returns [evals,Revec,Levec,amplitudes]. 
        Input must be an instance of the CSPCantera class"""
        self.nv = self.n_species
        self._rhs = self.source.copy()[:self.nv]
        kineticjac = self.jacKinetic()  
        thermaljac = self.jacThermal()   
        self._jac = kineticjac - thermaljac
        #eigensystem
        evals,Revec,Levec = eigsys(self.jac)
        f = np.matmul(Levec,self.rhs)
        self._nUpdates = self._nUpdates + 1
        #rotate eigenvectors such that amplitudes are positive    
        Revec,Levec,f = evec_pos_ampl(Revec,Levec,f)
        return[evals,Revec,Levec,f]


    
""" ~~~~~~~~~~~~ EXHAUSTED MODES ~~~~~~~~~~~~~
"""
def findM(n_elements,stateYT,evals,Revec,tau,f,rtol,atol):
    nv = len(Revec)
    nEl = n_elements 
    #nconjpairs = sum(1 for x in self.eval.imag if x != 0)/2
    imPart = evals.imag!=0
    nModes = nv - nEl
    ewt = setEwt(stateYT,rtol,atol)
    
    delw = np.zeros(nv)
    for j in range(nModes-1):          #loop over eligible modes (last excluded)
        taujp1 = tau[j+1]              #timescale of next (slower) mode
        Aj = Revec[j]                  #this mode right evec
        fj = f[j]                      #this mode amplitued
        lamj = evals[j].real           #this mode eigenvalue (real part)
        
        for i in range(nv):
            Aji = Aj[i]                     #i-th component of this mode Revec
            delw[i] = delw[i] + modeContribution(Aji,fj,taujp1,lamj)    #contribution of j-th mode to i-th var            
            if np.abs(delw[i]) > ewt[i]:
                if j==0:
                    M = 0
                else:
                    M = j-1 if (imPart[j] and imPart[j-1] and evals[j].real==evals[j-1].real) else j    #if j is the second of a pair, move back by 2                    
                return M

    #print("All modes are exhausted")
    M = nModes - 1   #if criterion is never verified, all modes are exhausted. Leave 1 active mode.
    return M



def setEwt(y,rtol,atol):   
    ewt = [rtol * absy + atol if absy >= 1.0e-6 else absy + atol for absy in np.absolute(y)]
    return ewt



def modeContribution(a,f,tau,lam):
    delwMi = a*f*(np.exp(tau*lam) - 1)/lam if lam != 0.0 else 0.0
    return delwMi        




""" ~~~~~~~~~~~~~~ TSR ~~~~~~~~~~~~~~~~
"""
    
def findTSR(n_elements,rhs,evals,Revec,f,M):
    n = len(Revec)
    nEl = n_elements 
    #deal with amplitudes of cmplx conjugates
    fvec = f.copy()
    imPart = evals.imag!=0
    for i in range(1,n):
        if (imPart[i] and imPart[i-1] and evals[i].real==evals[i-1].real):
            fvec[i] = np.sqrt(fvec[i]**2 + fvec[i-1]**2)
            fvec[i-1] = fvec[i]
    fnorm = fvec / np.linalg.norm(rhs)
    #deal with zero-eigenvalues (if any)
    fvec[evals==0.0] = 0.0
    weights = fnorm**2
    weights[0:M] = 0.0 #excluding fast modes
    weights[n-nEl:n] = 0.0 #excluding conserved modes
    normw = np.sum(weights)
    weights = weights / normw if normw > 0 else np.zeros(n)
    TSR = np.sum([weights[i] * np.sign(evals[i].real) * np.abs(evals[i]) for i in range(n)])
    return [TSR, weights]
          

def TSRindices(weights, evals):
    """Ns array containing participation index of mode i to TSR"""
    n = len(weights)
    Index = np.zeros((n))
    norm = np.sum([weights[i] * np.abs(evals[i]) for i in range(n)])
    for i in range(n):
        Index[i] = weights[i] * np.abs(evals[i])
        Index[i] = Index[i]/norm if norm > 1e-10 else 0.0
    return Index

def TSR_participation_indices(TSRidx, CSPidx):
    """2Nr array containing participation index of reaction k to TSR"""
    Index = np.matmul(np.transpose(CSPidx),np.abs(TSRidx))
    norm = np.sum(np.abs(Index))
    Index = Index/norm if norm > 0 else Index*0.0
    return Index
    
    
""" ~~~~~~~~~~~~ EIGEN FUNC ~~~~~~~~~~~~~
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
    
    return np.array(tau)




""" ~~~~~~~~~~~~ INDEXES FUNC ~~~~~~~~~~~~~
"""

def CSPIndices(Proj, Smat, rvec):
    ns = Smat.shape[0]
    nr = Smat.shape[1]
    Index = np.zeros((ns,nr))
    for i in range(ns):
        norm = 0.0
        for k in range(nr):
            Index[i,k] = np.dot(Proj[i,:],Smat[:,k]) * rvec[k]
            norm = norm + abs(Index[i,k])
        for k in range(nr):
            Index[i,k] = Index[i,k]/norm if (norm != 0.0) else 0.0 
    #np.sum(abs(Index),axis=1) check: a n-long array of ones
    return Index

def CSP_amplitude_participation_indices(B, Smat, rvec):
    """Ns x 2Nr array containing participation index of reaction k to variable i"""
    API = CSPIndices(B, Smat, rvec)
    return API

def CSP_importance_indices(A,B,M,Smat,rvec):
    """Ns x 2Nr array containing fast/slow importance index of reaction k to variable i"""
    fastProj = np.matmul(np.transpose(A[0:M]),B[0:M])
    Ifast = CSPIndices(fastProj, Smat, rvec)
    slowProj = np.matmul(np.transpose(A[M:]),B[M:])
    Islow = CSPIndices(slowProj, Smat, rvec)      
    return [Ifast,Islow]
                
def CSP_pointers(A,B):
    nv = A.shape[0]
    pointers = np.array([[np.transpose(A)[spec,mode]*B[mode,spec] for spec in range(nv)] for mode in range(nv)])            
    return pointers

def classify_species(stateYT, rhs, pointers, M):
    n = len(stateYT)
    ytol = 1e-20
    rhstol = 1e-13
    sort = np.absolute(np.sum(pointers[:,:M],axis=1)).argsort()[::-1]
    species_type = np.full(n,'slow',dtype=object)
    species_type[sort[0:M]] = 'fast'
    species_type[-1] = 'slow'  #temperature is always slow
    for i in range(1,n):
        if (stateYT[i] < ytol and abs(rhs[i]) < rhstol): species_type[i] = 'trace' 
    return species_type
        
def CSP_participation_to_one_timescale(i, nr, JacK, evals, A, B):
    imPart = evals.imag!=0
    if(imPart[i] and imPart[i-1] and evals[i].real==evals[i-1].real): i = i-1  #if the second of a complex pair, shift back the index by 1
    Index = np.zeros(2*nr)
    norm = 0.0
    if(imPart[i]):
        for k in range(2*nr):
            Index[k] = 0.5 * ( np.matmul(np.matmul(B[i],JacK[k]),np.transpose(A)[:,i]) -
                np.matmul(np.matmul(B[i+1],JacK[k]),np.transpose(A)[:,i+1]))
            norm = norm + abs(Index[k])
    else:
        for k in range(2*nr):
            Index[k] = np.matmul(np.matmul(B[i],JacK[k]),np.transpose(A)[:,i])
            norm = norm + abs(Index[k])
    for k in range(2*nr):
            Index[k] = Index[k]/norm if (norm != 0.0) else 0.0 
    return Index
    
def CSP_timescale_participation_indices(nr, JacK, evals, A, B):
    """Ns x 2Nr array containing TPI of reaction k to variable i"""
    nv = A.shape[0]
    TPI = np.zeros((nv,2*nr))
    for i in range(nv):
        TPI[i] = CSP_participation_to_one_timescale(i, nr, JacK, evals, A, B)
    return TPI    