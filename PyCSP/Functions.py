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
        self._classify_traces = True
        self._index_norm = 'abs'
        self._nUpdates = 0
        self._changed = False
    
    @property
    def jacobiantype(self):
        return self._jacobiantype
          
    @jacobiantype.setter
    def jacobiantype(self,value):
        if value == 'full':
            self.nv = self.n_species + 1   
            self._jacobiantype = value
        elif value == 'kinetic' or value == 'constrained':
            self.nv = self.n_species    
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
    def classify_traces(self):
        return self._classify_traces
          
    @classify_traces.setter
    def classify_traces(self,value):
        self._classify_traces = value
        
    @property
    def index_norm(self):
        return self._index_norm
          
    @index_norm.setter
    def index_norm(self,value):
        if value == 'abs' or value == 'none':  
            self._index_norm = value
        else:
            raise ValueError("Invalid index_norm --> %s" %value)
    
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
 
        
    def get_kernel(self):
        """Retrieves the stored CSP kernel.
        If any attributes changed before latest query, kernel is recomputed"""
        if self.is_changed():
            self.update_kernel()
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
                raise ValueError("unknown argument --> %s" %key)
        if self.is_changed(): 
            self.update_kernel()
            self._changed = False
        M = findM(self.n_elements,self.stateYT(),self.evals,self.Revec,self.tau,self.f,rtol,atol)
        return M
 
    def calc_subspaces(self, **kwargs):
        """Computes subspaces partitioning (M and H). 
        Optional arguments are rtol and atol for the calculation of M, and rtol and atol for the calculation of H.
        If not provided, uses default or previously set values"""
        rtolTail = self.rtol
        atolTail = self.atol
        rtolHead = self.rtol
        atolHead = self.atol
        for key, value in kwargs.items():
            if (key == 'rtolTail'):                 
                rtolTail = value
            elif (key == 'atolTail'): 
                atolTail = value
            elif (key == 'rtolHead'): 
                rtolHead = value
            elif (key == 'atolHead'): 
                atolHead = value
            else:
                raise ValueError("unknown argument --> %s" %key)
        if self.is_changed(): 
            self.update_kernel()
            self._changed = False
        M = findM(self.n_elements,self.stateYT(),self.evals,self.Revec,self.tau,self.f,rtolTail,atolTail)
        H = findH(self.n_elements,self.stateYT(),self.evals,self.Revec,self.tau[M],self.f,rtolHead,atolHead,M)
        return M, H
       
    def calc_TSR(self,**kwargs):
        """Computes number of exhausted modes (M) and the TSR. 
        Optional arguments are rtol and atol for the calculation of M.
        If not provided, uses default or previously set values.
        The calculated value of M can be retrieved by passing
        the optional argument getM=True"""
        getM = False
        rtol = self.rtol
        atol = self.atol
        for key, value in kwargs.items():
            if (key == 'rtol'): 
                rtol = value
            elif (key == 'atol'): 
                atol = value
            elif (key == 'getM'): 
                getM = value
            else:
                raise ValueError("unknown argument --> %s" %key)
        if self.is_changed(): 
            self.update_kernel()
            self._changed = False
        M = findM(self.n_elements,self.stateYT(),self.evals,self.Revec,self.tau,self.f,rtol,atol)
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
        getTSR = False
        useTPI = False
        rtol = self.rtol
        atol = self.atol
        for key, value in kwargs.items():
            if (key == 'rtol'): 
                rtol = value
            elif (key == 'atol'): 
                atol = value
            elif (key == 'getTSR'): 
                getTSR = value
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
        M = findM(self.n_elements,self.stateYT(),self.evals,self.Revec,self.tau,self.f,rtol,atol)
        TSR, weights = findTSR(self.n_elements,self.rhs,self.evals,self.Revec,self.f,M)
        TSRidx = TSRindices(weights, self.evals)
        if (useTPI):
            JacK = self.jac_contribution()
            CSPidx = CSP_timescale_participation_indices(self.n_reactions, JacK, self.evals, self.Revec, self.Levec, self.index_norm)
        else:
            CSPidx = CSP_amplitude_participation_indices(self.Levec, Smat, rvec, self.index_norm)
        TSRind = TSR_participation_indices(TSRidx, CSPidx, self.index_norm)
        if getTSR:
            return TSR, TSRind
        else:
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
        rtol = self.rtol
        atol = self.atol
        for key, value in kwargs.items():
            if (key == 'rtol'): 
                rtol = value
            elif (key == 'atol'): 
                atol = value
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
        M = findM(self.n_elements,self.stateYT(),self.evals,self.Revec,self.tau,self.f,rtol,atol)
        Smat = self.generalized_Stoich_matrix
        rvec = self.R_vector
        if getAPI: API = CSP_amplitude_participation_indices(self.Levec, Smat, rvec, self.index_norm)
        if getImpo: Ifast,Islow = CSP_importance_indices(self.Revec, self.Levec, M, Smat, rvec, self.index_norm)
        if getspeciestype: 
            pointers = CSP_pointers(self.Revec,self.Levec)
            species_type = classify_species(self.stateYT(), self.rhs, pointers, M, self.classify_traces)
        if getTPI:
            JacK = self.jac_contribution()
            TPI = CSP_timescale_participation_indices(self.n_reactions, JacK, self.evals, self.Revec, self.Levec, self.index_norm)
        if getM:
            return [API, TPI, Ifast, Islow, species_type, M]
        else:
            return [API, TPI, Ifast, Islow, species_type]
        
    def calc_extended_TSR(self,**kwargs):
        """Computes number of Extended exhausted modes (Mext) and the extended TSR. 
        Caller must provide either a diffusion rhs with the keywork rhs_diffYT
        or a convection rhs with the keywork rhs_convYT, or both.
        Optional arguments are rtol and atol for the calculation of Mext.
        If not provided, uses default or previously set values.
        The calculated value of Mext can be retrieved by passing
        the optional argument getMext=True"""
                        
        if self.is_changed(): 
            self.update_kernel()
            self._changed = False
        
        nv=len(self.Revec)
        getMext = False
        rtol = self.rtol
        atol = self.atol
        rhs_convYT = np.zeros(nv)
        rhs_diffYT = np.zeros(nv)
        for key, value in kwargs.items():
            if (key == 'rtol'): 
                rtol = value
            elif (key == 'atol'): 
                atol = value
            elif (key == 'getMext'): 
                getMext = value
            elif (key == 'conv'):
                rhs_convYT = value
            elif (key == 'diff'):
                rhs_diffYT = value
            else:
                raise ValueError("unknown argument --> %s" %key)
        if(len(rhs_convYT)!=nv):
            raise ValueError("Check dimension of Convection rhs. Should be %d", nv)
        if(len(rhs_diffYT)!=nv):
            raise ValueError("Check dimension of Diffusion rhs. Should be %d", nv)

            
        Smat = self.generalized_Stoich_matrix
        rvec = self.R_vector    
        rhs_ext, h, Smat_ext, rvec_ext = add_transport(self.rhs,self.Levec,Smat,rvec,rhs_convYT,rhs_diffYT)
        
        Mext = findM(self.n_elements,self.stateYT(),self.evals,self.Revec,self.tau,h,rtol,atol)
        TSR_ext, weights_ext = findTSR(self.n_elements,rhs_ext,self.evals,self.Revec,h,Mext)
        if getMext:
            return [TSR_ext, Mext]
        else:
            return TSR_ext

    def calc_extended_TSRindices(self,**kwargs):
        """Computes number of Extended exhausted modes (Mext), the extended TSR and its indices. 
        Caller must provide either a diffusion rhs with the keywork rhs_diffYT
        or a convection rhs with the keywork rhs_convYT, or both.
        Other optional arguments are rtol and atol for the calculation of Mext.
        If not provided, uses default or previously set values.
        The calculated value of Mext can be retrieved by passing
        the optional argument getMext=True"""
        if self.is_changed(): 
            self.update_kernel()
            self._changed = False
        getTSRext = False    
        nv=len(self.Revec)
        rtol = self.rtol
        atol = self.atol
        rhs_convYT = np.zeros(nv)
        rhs_diffYT = np.zeros(nv)
        for key, value in kwargs.items():
            if (key == 'rtol'): 
                rtol = value
            elif (key == 'atol'): 
                atol = value
            elif (key == 'getTSRext'): 
                getTSRext = value
            elif (key == 'conv'):
                rhs_convYT = value
            elif (key == 'diff'):
                rhs_diffYT = value
            else:
                raise ValueError("unknown argument --> %s" %key)
        
        if(len(rhs_convYT)!=nv):
            raise ValueError("Check dimension of Convection rhs. Should be %d", nv)
        if(len(rhs_diffYT)!=nv):
            raise ValueError("Check dimension of Diffusion rhs. Should be %d", nv)
            
        Smat = self.generalized_Stoich_matrix
        rvec = self.R_vector    
        rhs_ext, h, Smat_ext, rvec_ext = add_transport(self.rhs,self.Levec,Smat,rvec,rhs_convYT,rhs_diffYT)
        
        Mext = findM(self.n_elements,self.stateYT(),self.evals,self.Revec,self.tau,h,rtol,atol)
        TSR_ext, weights_ext = findTSR(self.n_elements,rhs_ext,self.evals,self.Revec,h,Mext)
        TSRidx = TSRindices(weights_ext, self.evals)
        CSPidx = CSP_amplitude_participation_indices(self.Levec, Smat_ext, rvec_ext, self.index_norm)
        TSRind_ext = TSR_participation_indices(TSRidx, CSPidx, self.index_norm)
        if getTSRext:
            return TSR_ext, TSRind_ext
        else:
            return TSRind_ext
        

        """ ~~~~~~~~~~~~ KERNEL ~~~~~~~~~~~~~
        """    
   
    def kernel(self):
        """Computes CSP kernel. Its dimension is Nspecies + 1.
        Returns [evals,Revec,Levec,amplitudes]. 
        Input must be an instance of the CSPCantera class"""
    
        #self.nv = self.n_species + 1     
        self._rhs = self.source.copy()
        self._jac = self.jacobian.copy()
        #eigensystem
        evals,Revec,Levec = eigsys(self.jac)
        self.clean_conserved(evals)
        f = np.matmul(Levec,self.rhs)
        self._nUpdates = self._nUpdates + 1
        #rotate eigenvectors such that amplitudes are positive    
        Revec,Levec,f = evec_pos_ampl(Revec,Levec,f)
        
        return[evals,Revec,Levec,f]
    
    
    def kernel_kinetic_only(self):
        """Computes kinetic kernel. Its dimension is Nspecies.
        Returns [evals,Revec,Levec,amplitudes]. 
        Input must be an instance of the CSPCantera class"""
        #self.nv = self.n_species
        self._rhs = self.source.copy()[:self.nv]
        self._jac = self.jacKinetic().copy()       
        #eigensystem
        evals,Revec,Levec = eigsys(self.jac)
        self.clean_conserved(evals)
        f = np.matmul(Levec,self.rhs)
        self._nUpdates = self._nUpdates + 1
        #rotate eigenvectors such that amplitudes are positive    
        Revec,Levec,f = evec_pos_ampl(Revec,Levec,f)
        return[evals,Revec,Levec,f]
    
    
    def kernel_constrained_jac(self):
        """Computes constrained (to enthalpy) kernel. Its dimension is Nspecies .
        Returns [evals,Revec,Levec,amplitudes]. 
        Input must be an instance of the CSPCantera class"""
        #self.nv = self.n_species
        self._rhs = self.source.copy()[:self.nv]
        kineticjac = self.jacKinetic()  
        thermaljac = self.jacThermal()   
        self._jac = kineticjac - thermaljac
        #eigensystem
        evals,Revec,Levec = eigsys(self.jac)
        self.clean_conserved(evals)
        f = np.matmul(Levec,self.rhs)
        self._nUpdates = self._nUpdates + 1
        #rotate eigenvectors such that amplitudes are positive    
        Revec,Levec,f = evec_pos_ampl(Revec,Levec,f)
        return[evals,Revec,Levec,f]


    def clean_conserved(self,evals):
        """Zero-out conserved modes eigenvalues"""
        #threshold = np.abs(evals[0] * np.finfo(float).eps) #smallest acceptable eigenvalue based on range precision
        #nCons = max( self.n_elements , np.sum(np.abs(evals) < threshold) )  #conserved subspace is n_elements or out-of-range eigenvalues if larger
        #i = self.nv - nCons
        i = self.nv-self.n_elements
        evals[i:] = 0.0
        
    
""" ~~~~~~~~~~~~ EXHAUSTED MODES ~~~~~~~~~~~~~
"""
def findM_original(n_elements,stateYT,evals,Revec,tau,f,rtol,atol):
    nv = len(Revec)
    #nEl = n_elements 
    threshold = np.abs(evals[0] * np.finfo(float).eps) #smallest acceptable eigenvalue based on range precision
    nEl = max( n_elements , np.sum(np.abs(evals) < threshold) )  #conserved subspace is n_elements or out-of-range eigenvalues if larger

    #nconjpairs = sum(1 for x in self.eval.imag if x != 0)/2
    imPart = evals.imag!=0
    nModes = nv - nEl    #removing conserved modes
    ewt = setEwt(stateYT,rtol,atol)    
    delw = np.zeros(nv)
    for j in range(nModes-1):          #loop over eligible modes (last excluded)
        taujp1 = tau[j+1]              #timescale of next (slower) mode
        Aj = Revec[j]                  #this mode right evec
        fj = f[j]                      #this mode amplitued
        lamj = evals[j].real           #this mode eigenvalue (real part)
        
        delwMi =  modeContribution(Aj,fj,taujp1,lamj)    #contribution of j-th mode to state variables    
        delw += delwMi
        #if np.any(np.abs(delw) > ewt):
        if np.max(np.abs(delw) / ewt) > 1:
            if j==0:
                M = 0
            else:
                M = j-1 if (imPart[j] and imPart[j-1] and evals[j].real==evals[j-1].real) else j    #if j is the second of a pair, move back by 2                    
            return M

    #print("All modes are exhausted")
    M = nModes - 1   #if criterion is never verified, all modes are exhausted. Leave 1 active mode.
    return M


def setEwt_original(y,rtol,atol):   
    ewt = [rtol * absy + atol if absy >= 1.0e-6 else absy + atol for absy in np.absolute(y)]
    return ewt


def modeContribution_original(a,f,tau,lam):
    #delwMi = a*f*(np.exp(tau*lam) - 1)/lam if lam != 0.0 else 0.0
    delwMi = a*f*(np.exp(tau*lam) - 1)/lam if lam != 0.0 and np.isfinite(np.exp(tau*lam)) else 1e+20
    return delwMi        

""" ~~~~~~~~~~~~~~ findM optimized ~~~~~~~~~~~~~~~~
"""

def findM(n_elements, stateYT, evals, Revec, tau, f, rtol, atol):
    nv = len(Revec)
    # Threshold for eigenvalue precision
    threshold = np.abs(evals[0] * np.finfo(float).eps)
    nEl = max(n_elements, np.sum(np.abs(evals) < threshold))  # Conserved subspace size
    
    # Identify imaginary part of eigenvalues
    imPart = evals.imag != 0
    nModes = nv - nEl  # Number of non-conserved modes
    
    # Set error weight and precompute its absolute value
    ewt = np.abs(setEwt(stateYT, rtol, atol))
    
    # Precompute reusable values to minimize redundant computations
    tau_shifted = tau[1:]  # Shifted tau values for efficiency
    real_evals = evals.real
    pair_check = imPart[:-1] & imPart[1:] & (real_evals[:-1] == real_evals[1:])
    pair_check_modes = pair_check[:nModes - 1]
    
    # Slice arrays to get only non-conserved modes
    Revec_modes = Revec[:nModes - 1]  # Shape: (nModes - 1, n_elements)
    f_modes = f[:nModes - 1]          # Shape: (nModes - 1,)
    tau_modes = tau_shifted[:nModes - 1]  # Shape: (nModes - 1,)
    lam_modes = real_evals[:nModes - 1]   # Shape: (nModes - 1,)
    
    # Compute mode contributions
    delwMi = modeContribution(Revec_modes, f_modes, tau_modes, lam_modes)  # Shape: (nModes - 1, n_elements)
    
    # Compute cumulative sums over modes
    delw_cumulative = np.cumsum(delwMi, axis=0)  # Shape: (nModes - 1, n_elements)
    
    # Compute the condition for each mode
    condition = np.any(np.abs(delw_cumulative) > ewt, axis=1)  # Shape: (nModes - 1,)
    
    # Find the first index where condition is True
    indices = np.where(condition)[0]
    
    if len(indices) == 0:
        # If no condition is met, return the last mode index
        return nModes - 1
    else:
        j = indices[0]
        if j == 0:
            return 0
        else:
            # Adjust index based on pair_check
            if pair_check_modes[j - 1]:
                return j - 1
            else:
                return j

def setEwt(y, rtol, atol):
    absy = np.abs(y)
    ewt = np.where(absy >= 1.0e-6, rtol * absy + atol, absy + atol)
    return ewt

def modeContribution(a, f, tau, lam):
    # Compute exponential term
    exp_term = np.exp(tau * lam)
    
    # Identify invalid cases
    invalid = (lam == 0.0) | (~np.isfinite(exp_term))
    valid = ~invalid
    
    # Initialize delwMi with default value
    delwMi = np.full_like(a, 1e+20, dtype=np.float64)
    
    # Compute valid contributions
    if np.any(valid):
        f_valid = f[valid][:, np.newaxis]
        lam_valid = lam[valid][:, np.newaxis]
        exp_term_valid = exp_term[valid][:, np.newaxis]
        a_valid = a[valid, :]
        delwMi[valid, :] = a_valid * f_valid * (exp_term_valid - 1) / lam_valid
    
    return delwMi



def findH(n_elements,stateYT,evals,Revec,dt,f,rtol,atol,Tail):
    nv = len(Revec)
    nEl = n_elements 
    #nconjpairs = sum(1 for x in self.eval.imag if x != 0)/2
    imPart = evals.imag!=0
    nModes = nv - nEl - 1   #removing conserved modes
    ewt = setEwt(stateYT,rtol,atol)    
    delw = np.zeros(nv)
    
    for j in range(nModes,Tail,-1):    #backwards loop over eligible modes (conserved excluded)
        Aj = Revec[j]                  #this mode right evec
        fj = f[j]                      #this mode amplitude
        lamj = evals[j].real           #this mode eigenvalue (real part)
        
        delw = delw + 0.5*dt*dt*Aj*fj*np.abs(lamj)    #contribution of j-th mode to all vars           
        if np.any(np.abs(delw) > ewt):
            if j==nModes:
                H = nModes  
            else:
                H = j+1 if (imPart[j] and imPart[j+1] and evals[j].real==evals[j+1].real) else j    #if j is the second of a pair, move fwd by 2                    
            return H

    #print("No modes are active")
    #H = Tail   #if criterion is never verified, no modes are active.
    H = Tail + 1  #if criterion is never verified, leave one active mode
    #print("-----------")
    return H
    

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

def TSR_participation_indices(TSRidx, CSPidx, normtype):
    """2Nr array containing participation index of reaction k to TSR"""
    Index = np.matmul(np.transpose(CSPidx),np.abs(TSRidx))
    if normtype == 'abs':
        norm = np.sum(np.abs(Index))
        Index = Index/norm if norm > 0 else Index*0.0
    return Index
    
    
""" ~~~~~~~~~~~~ EIGEN FUNC ~~~~~~~~~~~~~
"""

def eigsys(jac):        
    """Returns eigensystem (evals, Revec, Levec). Input must be a 2D array.
    Both Revec (since it is transposed) and Levec (naturally) contain row vectors,
    such that an eigenvector can be retrieved using R/Levec[index,:] """
    
    #ncons = len(jac) - np.linalg.matrix_rank(jac)
    evals, Revec = np.linalg.eig(jac)
    
    #sort 
    idx = np.argsort(abs(evals))[::-1]   
    evals = evals[idx]
    Revec = Revec[:,idx]
    
    #zero-out conserved eigenvalues (last ncons)
    #evals[-ncons:] = 0.0

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
    try:
        Levec = np.linalg.inv(Revec)
    except:
        print('Warning: looks like the R martrix is singular (rank[R] = %i).' %np.linalg.matrix_rank(Revec))
        print('         Kernel is zeroed-out.')
        return[np.zeros(len(jac)),np.zeros((len(jac),len(jac))),np.zeros((len(jac),len(jac)))]
    
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

def CSPIndices(Proj, Smat, rvec, normtype):
    """Returns a Nv x 2Nr matrix of indexes, computed as Proj S r """
    ns = Smat.shape[0]
    nr = Smat.shape[1]
    Index = np.zeros((ns,nr))
    for i in range(ns):
        Proj_i = Proj[i,:]
        Index[i,:] = CSPIndices_one_var(Proj_i, Smat, rvec, normtype)
    #np.sum(abs(Index),axis=1) check: a n-long array of ones
    return Index

def CSPIndices_one_var(Proj_i, Smat, rvec, normtype):
    """Given the i-th row of the projector, computes 2Nr indexes of reactions to variable i.
    Proj_i must be a nv-long array. Returns a 2Nr-long array"""
    nr = Smat.shape[1]
    PS = np.matmul(Proj_i,Smat) 
    Index = np.multiply(PS,rvec)
    if normtype == 'abs':
        norm = np.sum(abs(Index))
        if norm != 0.0:
            Index = Index/norm
        else:
            Index = np.zeros((nr))
        #np.sum(abs(Index),axis=1) check: a n-long array of ones
    return Index


def CSP_amplitude_participation_indices(B, Smat, rvec, normtype):
    """Ns x 2Nr array containing participation index of reaction k to variable i"""
    API = CSPIndices(B, Smat, rvec, normtype)
    return API

def CSP_importance_indices(A, B, M, Smat, rvec, normtype):
    """Ns x 2Nr array containing fast/slow importance index of reaction k to variable i"""
    fastProj = np.matmul(np.transpose(A[0:M]),B[0:M])
    Ifast = CSPIndices(fastProj, Smat, rvec, normtype)
    slowProj = np.matmul(np.transpose(A[M:]),B[M:])
    Islow = CSPIndices(slowProj, Smat, rvec, normtype)      
    return [Ifast,Islow]
                
def CSP_pointers(A,B):
    nv = A.shape[0]
    pointers = np.array([[np.transpose(A)[spec,mode]*B[mode,spec] for spec in range(nv)] for mode in range(nv)])            
    return pointers

def classify_species(stateYT, rhs, pointers, M, trace):
    """species classification
    """
    n = len(stateYT)
    ytol = 1e-20
    rhstol = 1e-13
    sort = np.absolute(np.sum(pointers[:,:M],axis=1)).argsort()[::-1]
    species_type = np.full(n,'slow',dtype=object)
    species_type[sort[0:M]] = 'fast'
    species_type[-1] = 'slow'  #temperature is always slow
    if trace:
        for i in range(n-1):
            if (stateYT[i] < ytol and abs(rhs[i]) < rhstol): species_type[i] = 'trace'
    return species_type
        
def CSP_participation_to_one_timescale(i, nr, JacK, evals, A, B, normtype):
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
            if normtype != 'none': 
                Index[k] = Index[k]/norm if (norm != 0.0) else 0.0 
    return Index
    
def CSP_timescale_participation_indices(nr, JacK, evals, A, B, normtype):
    """Ns x 2Nr array containing TPI of reaction k to variable i"""
    nv = A.shape[0]
    TPI = np.zeros((nv,2*nr))
    for i in range(nv):
        TPI[i] = CSP_participation_to_one_timescale(i, nr, JacK, evals, A, B, normtype)
    return TPI    


def CSPtiming(gas):
    import time
    ns = gas.n_species
    randY =  np.random.dirichlet(np.ones(ns),size=1)
    gas.TP = 1000,101325.0
    gas.Y = randY
    gas.constP = 101325.0
    gas.jacobiantype='full'
    gas.rtol=1.0e-3
    gas.atol=1.0e-10
    starttime = time.time()
    gas.update_kernel()
    endtime = time.time()
    timekernel = endtime - starttime
    starttime = time.time()
    M = gas.calc_exhausted_modes()
    endtime = time.time()
    timeM = endtime - starttime
    starttime = time.time()
    TSR = gas.calc_TSR()
    endtime = time.time()
    timeTSR = endtime - starttime
    starttime = time.time()
    api, tpi, ifast, islow, species_type = gas.calc_CSPindices(API=True,Impo=False,species_type=False,TPI=False)
    endtime = time.time()
    timeAPI = endtime - starttime
    starttime = time.time()
    api, tpi, ifast, islow, species_type = gas.calc_CSPindices(API=False,Impo=True,species_type=False,TPI=False)
    endtime = time.time()
    timeImpo = endtime - starttime
    starttime = time.time()
    api, tpi, ifast, islow, species_type = gas.calc_CSPindices(API=False,Impo=False,species_type=True,TPI=False)
    endtime = time.time()
    timeclassify = endtime - starttime
    starttime = time.time()
    api, tpi, ifast, islow, species_type = gas.calc_CSPindices(API=False,Impo=False,species_type=False,TPI=True)
    endtime = time.time()
    timeTPI = endtime - starttime
    print ('Time Kernel:      %10.3e' %timekernel)
    print ('Time findM:       %10.3e' %timeM)
    print ('Time TSR:         %10.3e' %timeTSR)
    print ('Time API indexes: %10.3e' %timeAPI)
    print ('Time Imp indexes: %10.3e' %timeImpo)
    print ('Time TPI indexes: %10.3e' %timeTPI)
    print ('Time class specs: %10.3e' %timeclassify)
    print ('*** all times in seconds')

    
"""  ~~~~~~~~~~~ EXTENDED FUNC ~~~~~~~~~~~~
"""

def add_transport(rhs,Levec,Smat,rvec,rhs_convYT,rhs_diffYT):
    nv=len(Levec)
    nr=Smat.shape[1]
    rhs_ext = rhs + rhs_convYT + rhs_diffYT
    h = np.matmul(Levec,rhs_ext)
    Smat_ext = np.zeros((nv,nr+2*nv))    
    Smat_ext[:,0:nr] = Smat
    Smat_ext[:,nr:nr+nv] = np.eye(nv)
    Smat_ext[:,nr+nv:nr+2*nv] = np.eye(nv)
    rvec_ext = np.zeros((nr+2*nv))
    rvec_ext[0:nr] = rvec
    rvec_ext[nr:nr+nv] = rhs_convYT
    rvec_ext[nr+nv:nr+2*nv] = rhs_diffYT
    
    #splitrhs_ext = np.dot(Smat_ext,rvec_ext)
    #checksplitrhs = np.isclose(rhs_ext, splitrhs_ext, rtol=1e-6, atol=0, equal_nan=False)
    #if(np.any(checksplitrhs == False)):
    #    raise ValueError('Mismatch between numerical extended RHS and S.r')
        
    return rhs_ext, h, Smat_ext, rvec_ext 
    