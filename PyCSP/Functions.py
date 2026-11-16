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
        TSR, _ = findTSR(self.n_elements,self.rhs,self.evals,self.Revec,self.f,M)
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
        The TSR value can be retrieved by passing getTSR=True, in which case
        the method returns (TSR, indices).
        Optional norm_cut (scalar or nv-long array, default None): with
        index_norm='abs', modes whose CSP indices have a sum of absolute
        contributions smaller than norm_cut get zero indices (units of that sum:
        1/s for type='timescale', the mode-amplitude units for type='amplitude')."""
        getTSR = False
        useTPI = False
        norm_cut = None
        rtol = self.rtol
        atol = self.atol
        for key, value in kwargs.items():
            if (key == 'rtol'): 
                rtol = value
            elif (key == 'atol'): 
                atol = value
            elif (key == 'getTSR'): 
                getTSR = value
            elif (key == 'norm_cut'): 
                norm_cut = value
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
            JacK = self._jac_contribution_for_TPI()
            CSPidx = CSP_timescale_participation_indices(self.n_reactions, JacK, self.evals, self.Revec, self.Levec, self.index_norm, norm_cut)
        else:
            CSPidx = CSP_amplitude_participation_indices(self.Levec, Smat, rvec, self.index_norm, norm_cut)
        TSRind = TSR_participation_indices(TSRidx, CSPidx, self.index_norm)
        if getTSR:
            return TSR, TSRind
        else:
            return TSRind
        
        
    def calc_CSPindices(self,**kwargs):
        """Computes number of exhausted modes (M) and the CSP Indices.
        Boolean keywords select the indices to compute (all default False):
        API (amplitude participation), TPI (timescale participation),
        Impo (fast/slow importance), species_type (fast/slow/trace species).
        Returns [API, TPI, Ifast, Islow, species_type] (None for those not
        requested), plus M if getM=True.
        Optional arguments are rtol and atol for the calculation of M.
        If not provided, uses default or previously set values.
        Optional norm_cut (scalar or nv-long array, default None): with
        index_norm='abs', rows (modes for API/TPI, variables for Ifast/Islow)
        whose sum of absolute contributions is smaller than norm_cut are set
        to zero instead of being normalized (units of that sum: the rate of
        the mode amplitude or variable, 1/s for TPI)."""
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
        norm_cut = None
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
            elif (key == 'norm_cut'): 
                norm_cut = value
            else:
                raise ValueError("unknown argument --> %s" %key)
        if self.is_changed(): 
            self.update_kernel()
            self._changed = False
        M = findM(self.n_elements,self.stateYT(),self.evals,self.Revec,self.tau,self.f,rtol,atol)
        Smat = self.generalized_Stoich_matrix
        rvec = self.R_vector
        if getAPI: API = CSP_amplitude_participation_indices(self.Levec, Smat, rvec, self.index_norm, norm_cut)
        if getImpo: Ifast,Islow = CSP_importance_indices(self.Revec, self.Levec, M, Smat, rvec, self.index_norm, norm_cut)
        if getspeciestype: 
            pointers = CSP_pointers(self.Revec,self.Levec)
            species_type = classify_species(self.stateYT(), self.rhs, pointers, M, self.classify_traces)
        if getTPI:
            JacK = self._jac_contribution_for_TPI()
            TPI = CSP_timescale_participation_indices(self.n_reactions, JacK, self.evals, self.Revec, self.Levec, self.index_norm, norm_cut)
        if getM:
            return [API, TPI, Ifast, Islow, species_type, M]
        else:
            return [API, TPI, Ifast, Islow, species_type]
        
    def calc_extended_TSR(self,**kwargs):
        """Computes number of Extended exhausted modes (Mext) and the extended TSR. 
        Caller must provide either a diffusion rhs with the keyword diff
        or a convection rhs with the keyword conv, or both (nv-long arrays).
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
            raise ValueError("Check dimension of Convection rhs. Should be %d" % nv)
        if(len(rhs_diffYT)!=nv):
            raise ValueError("Check dimension of Diffusion rhs. Should be %d" % nv)

            
        Smat = self.generalized_Stoich_matrix
        rvec = self.R_vector    
        rhs_ext, h, _, _ = add_transport(self.rhs,self.Levec,Smat,rvec,rhs_convYT,rhs_diffYT)
        
        Mext = findM(self.n_elements,self.stateYT(),self.evals,self.Revec,self.tau,h,rtol,atol)
        TSR_ext, _ = findTSR(self.n_elements,rhs_ext,self.evals,self.Revec,h,Mext)
        if getMext:
            return [TSR_ext, Mext]
        else:
            return TSR_ext

    def calc_extended_TSRindices(self,**kwargs):
        """Computes number of Extended exhausted modes (Mext), the extended TSR and its indices. 
        Caller must provide either a diffusion rhs with the keyword diff
        or a convection rhs with the keyword conv, or both (nv-long arrays).
        Other optional arguments are rtol and atol for the calculation of Mext.
        If not provided, uses default or previously set values.
        The extended TSR can be retrieved by passing the optional argument
        getTSRext=True, in which case the method returns (TSR_ext, indices).
        Indices refer to the 2Nr reactions, then the nv convection and the
        nv diffusion terms."""
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
            raise ValueError("Check dimension of Convection rhs. Should be %d" % nv)
        if(len(rhs_diffYT)!=nv):
            raise ValueError("Check dimension of Diffusion rhs. Should be %d" % nv)
            
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
        """
    
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
        """
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
        """Computes constrained kernel (constant enthalpy for const_p, constant internal energy for const_v). Its dimension is Nspecies.
        Returns [evals,Revec,Levec,amplitudes]. 
        """
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


    def _jac_contribution_for_TPI(self):
        """Per-reaction Jacobian contributions used by the timescale participation
        indices. They sum to the full or kinetic Jacobian only: the constrained
        Jacobian also contains the thermal correction, which is not split by
        reaction, so TPIs would not decompose its eigenvalues."""
        if self.jacobiantype == 'constrained':
            raise NotImplementedError("Timescale participation indices are not available "
                                      "for jacobiantype='constrained'. Use 'full' or 'kinetic'.")
        return self.jac_contribution()

    def clean_conserved(self,evals):
        """Zero-out conserved modes eigenvalues"""
        i = self.nv-self.n_elements
        evals[i:] = 0.0
        
    
""" ~~~~~~~~~~~~ EXHAUSTED MODES ~~~~~~~~~~~~~
"""
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
    ewt = np.abs(setEwt(stateYT, rtol, atol))[:nv]   #[:nv] required for constrained jacobian that has size smaller than Ns+1, while stateYT is always Ns+1
    
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
    ewt = rtol * absy + atol
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
    imPart = evals.imag!=0
    nModes = nv - nEl - 1   #removing conserved modes
    ewt = setEwt(stateYT,rtol,atol)[:nv]   #[:nv] required for kinetic/constrained jacobian (size Ns)
    delw = np.zeros(nv)
    
    for j in range(nModes,Tail,-1):    #backwards loop over eligible modes (conserved excluded)
        Aj = Revec[j]                  #this mode right evec
        fj = f[j]                      #this mode amplitude
        lamj = evals[j].real           #this mode eigenvalue (real part)
        
        delw = delw + 0.5*dt*dt*Aj*fj*np.abs(lamj)    #contribution of j-th mode to all vars           
        if np.any(np.abs(delw) > ewt):
            if j==nModes:
                H = nModes + 1  #in this case, H is nv - nEl. Hence, last available mode is slow. 
            else:
                H = j+2 if (imPart[j] and imPart[j+1] and evals[j].real==evals[j+1].real) else j+1    #if j is the second of a pair, move fwd by 2                    
            return H

    H = Tail + 1  #if criterion is never verified, leave one active mode
    return H


def detectFrozenModes(M, stateYT, Revec, Levec, Smat, rvec, rtol, atol):
    """
    Identifies frozen modes among the M exhausted modes (or all modes, if needed).
    A mode is frozen if the sum of absolute projected reaction rates is small.
    
    Parameters:
    -----------
    M : int
        Number of exhausted modes (returned by findM).
    stateYT : array
        State vector [Species... , T].
    Revec : array
        Right eigenvectors (rows).
    Levec : array
        Left eigenvectors (rows).
    Smat : array
        Stoichiometric matrix.
    rvec : array
        Reaction rates vector.
    rtol : float
        Relative tolerance.
    atol : float
        Absolute tolerance.
        
    Returns:
    --------
    frozen_indices : list
        Indices of modes that are frozen.
    """
    nv = len(Revec)
    
    # 1. Calculate ewt
    ewt = np.abs(setEwt(stateYT, rtol, atol))[:nv]
    
    frozen_modes = []
    
    # Iterate over exhausted modes (0 to M-1)
    for m in range(M):
        # 1. Construct Projection Matrix P_m = A_m * B_m
        # Revec[m] is the m-th eigenvector (column in theory, but stored as rows in PyCSP?)
        # So Revec[m] is the right eigenvector A_m.
        # Levec[m] is the left eigenvector B_m.
        
        A_m = Revec[m]
        B_m = Levec[m]
        P_m = np.outer(A_m, B_m) # Shape (nv, nv)
        
        # 2. Compute physical reaction rate vectors S_k * r_k
        # Smat is (nv, nr), rvec is (nr,)
        # We want a matrix where col k is S_k * r_k
        R_vecs = Smat * rvec[np.newaxis, :] # Shape (nv, nr)
        
        # 3. Project reaction rates: P_m @ (S_k * r_k)
        Proj_R_vecs = np.matmul(P_m, R_vecs) # Shape (nv, nr)
        
        # 4. Sum of absolute values of projected rates
        max_delw = np.sum(np.abs(Proj_R_vecs), axis=1) # Shape (nv,)
        
        # 5. Check condition
        if np.all(max_delw < ewt):
            frozen_modes.append(m)
            
    bool_frozen = np.isin(np.arange(len(stateYT)), frozen_modes)
    return bool_frozen  

""" ~~~~~~~~~~~~~~ TSR ~~~~~~~~~~~~~~~~
"""
    
def findTSR(n_elements,rhs,evals,Revec,f,M):
    n = len(Revec)
    # conserved subspace: n_elements or out-of-range eigenvalues if larger (as in findM)
    threshold = np.abs(evals[0] * np.finfo(float).eps)
    nEl = max(n_elements, np.sum(np.abs(evals) < threshold))
    #deal with amplitudes of cmplx conjugates
    fvec = f.copy()
    imPart = evals.imag!=0
    for i in range(1,n):
        if (imPart[i] and imPart[i-1] and evals[i].real==evals[i-1].real):
            fvec[i] = np.sqrt(fvec[i]**2 + fvec[i-1]**2)
            fvec[i-1] = fvec[i]
    fnorm = fvec / np.linalg.norm(rhs)
    #deal with zero-eigenvalues (if any)
    fnorm[evals==0.0] = 0.0
    weights = fnorm**2
    weights[0:M] = 0.0 #excluding fast modes
    weights[n-nEl:n] = 0.0 #excluding conserved modes
    normw = np.sum(weights)
    weights = weights / normw if normw > 0 else np.zeros(n)
    TSR = np.sum([weights[i] * np.sign(evals[i].real) * np.abs(evals[i]) for i in range(n)])
    return [TSR, weights]
          

def TSRindices(weights, evals):
    """Nv array containing participation index of mode i to TSR"""
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
    try:
        Levec = np.linalg.inv(Revec)
    except np.linalg.LinAlgError:
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

def _row_cut(norm_cut, i):
    """norm_cut may be None, a scalar (same cut for every row) or a per-row array."""
    if norm_cut is None or np.isscalar(norm_cut):
        return norm_cut
    return norm_cut[i]

def _normalize_index(Index, normtype, norm_cut=None):
    """Normalizes Index by the sum of its absolute values (normtype='abs').
    If norm_cut is given and that sum is smaller than norm_cut, the index is
    set to zero: the process contributions are negligible and their relative
    weights are not meaningful."""
    if normtype != 'abs':
        return Index
    norm = np.sum(np.abs(Index))
    if norm == 0.0 or (norm_cut is not None and norm < norm_cut):
        return np.zeros_like(Index)
    return Index/norm

def CSPIndices(Proj, Smat, rvec, normtype, norm_cut=None):
    """Returns a Nv x 2Nr matrix of indexes, computed as Proj S r.
    Optional norm_cut (scalar or Nv-long array): rows whose sum of absolute
    contributions is below the cut are set to zero (only with normtype='abs')."""
    nv = Smat.shape[0]
    nr = Smat.shape[1]
    if norm_cut is not None and not np.isscalar(norm_cut) and len(norm_cut) != nv:
        raise ValueError("norm_cut must be a scalar or a %d-long array" % nv)
    Index = np.zeros((nv,nr))
    for i in range(nv):
        Index[i,:] = CSPIndices_one_var(Proj[i,:], Smat, rvec, normtype, _row_cut(norm_cut, i))
    return Index

def CSPIndices_one_var(Proj_i, Smat, rvec, normtype, norm_cut=None):
    """Given the i-th row of the projector, computes 2Nr indexes of reactions to variable i.
    Proj_i must be a nv-long array. Returns a 2Nr-long array"""
    PS = np.matmul(Proj_i,Smat) 
    Index = np.multiply(PS,rvec)
    return _normalize_index(Index, normtype, norm_cut)


def CSP_amplitude_participation_indices(B, Smat, rvec, normtype, norm_cut=None):
    """Nv x 2Nr array containing participation index of reaction k to mode i"""
    API = CSPIndices(B, Smat, rvec, normtype, norm_cut)
    return API

def CSP_importance_indices(A, B, M, Smat, rvec, normtype, norm_cut=None):
    """Nv x 2Nr arrays containing fast/slow importance index of reaction k to variable i"""
    fastProj = np.matmul(np.transpose(A[0:M]),B[0:M])
    Ifast = CSPIndices(fastProj, Smat, rvec, normtype, norm_cut)
    slowProj = np.matmul(np.transpose(A[M:]),B[M:])
    Islow = CSPIndices(slowProj, Smat, rvec, normtype, norm_cut)      
    return [Ifast,Islow]
                
def CSP_pointers(A,B):
    """
    CSP pointers following Lam & Goussis.
    - pointers[m,k] = diag_k(Q_m)
    - Q_m = A_m B_m 
    - m is the mode index
    - k is the species index
    """ 
    nv = A.shape[0]
    pointers = np.array([[np.transpose(A)[spec,mode]*B[mode,spec] for spec in range(nv)] for mode in range(nv)])            
    return pointers

def classify_species(stateYT, rhs, pointers, M, trace):
    """
    CSP species classification following Lam & Goussis.
    - pointers[m,k] = diag_k(Q_m)
    - M = number of fast modes
    """

    n = len(stateYT)
    species_type = np.full(n, 'slow', dtype=object)

    # 1. Build Q_hat = sum of fast-mode Q_m
    Qhat = pointers[:M, :].sum(axis=0)   # shape (n_species,)

    # 2. Sort diagonal entries of Q_hat
    sorted_idx = np.argsort(Qhat)[::-1]

    # 3. First M species → radical (fast) species
    radicals = sorted_idx[:M]
    species_type[radicals] = 'fast'

    # 4. Temperature (last index) is always slow
    species_type[-1] = 'slow'

    # 5. Trace species identification
    if trace:
        ytol = 1e-20
        for i in range(n - 1):
            if stateYT[i] < ytol and rhs[i] <= 0.0:
                species_type[i] = 'trace'

    return species_type

        
def CSP_participation_to_one_timescale(i, nr, JacK, evals, A, B, normtype, norm_cut=None):
    """2Nr-long array of contributions of each reaction to the i-th eigenvalue
    (its real part for a complex-conjugate pair): lambda_i = sum_k b^i J_k a_i."""
    imPart = evals.imag!=0
    if(imPart[i] and imPart[i-1] and evals[i].real==evals[i-1].real): i = i-1  #if the second of a complex pair, shift back the index by 1
    Index = np.zeros(2*nr)
    if(imPart[i]):
        for k in range(2*nr):
            # complex pair stored as real basis [Re v, Im v]: both diagonal terms
            # b^i J a_i and b^(i+1) J a_(i+1) equal Re(lambda), so their mean
            # decomposes Re(lambda) over the reactions.
            Index[k] = 0.5 * ( np.matmul(np.matmul(B[i],JacK[k]),np.transpose(A)[:,i]) +
                np.matmul(np.matmul(B[i+1],JacK[k]),np.transpose(A)[:,i+1]))
    else:
        for k in range(2*nr):
            Index[k] = np.matmul(np.matmul(B[i],JacK[k]),np.transpose(A)[:,i])
    return _normalize_index(Index, normtype, norm_cut)
    
def CSP_timescale_participation_indices(nr, JacK, evals, A, B, normtype, norm_cut=None):
    """Nv x 2Nr array containing TPI of reaction k to mode i"""
    nv = A.shape[0]
    if norm_cut is not None and not np.isscalar(norm_cut) and len(norm_cut) != nv:
        raise ValueError("norm_cut must be a scalar or a %d-long array" % nv)
    TPI = np.zeros((nv,2*nr))
    for i in range(nv):
        TPI[i] = CSP_participation_to_one_timescale(i, nr, JacK, evals, A, B, normtype, _row_cut(norm_cut, i))
    return TPI    


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
    
        
    return rhs_ext, h, Smat_ext, rvec_ext 

