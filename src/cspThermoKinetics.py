#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Riccardo Malpica Galassi, Sapienza University, Roma, Italy
"""

import numpy as np
import pyjacob as pj
import cantera as ct


class CanteraThermoKinetics(ct.Solution):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  



    """ ~~~~~~~~~~~~ RHS ~~~~~~~~~~~~~
    """
    def rhs_const_p(self):
        """Computes chemical RHS [shape:(ns)]. Inert (last) term is not returned.
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
        """Retrieves chemical RHS from PyJac [shape(ns)]. Inert (last) term is not returned.
        Input must be an instance of the CSPCantera class"""
        
        #setup the state vector
        y = np.zeros(self.n_species)
        y[0] = self.T
        y[1:] = self.Y[:-1]
        
        #create ydot vector
        ydot = np.zeros_like(y)
        pj.py_dydt(0, self.P, y, ydot)
           
        return ydot
    
    """ ~~~~~~~~~~~~ Stoichiometric matrix ~~~~~~~~~~~~~
    """
    
    def generalized_Stoich_matrix(self):
        """N_s x 2*N_r matrix containing the S components in column major format, 
        such that S dot Rvec yields RHS"""
        nu_p = self.product_stoich_coeffs()
        nu_r = self.reactant_stoich_coeffs()
        rho = self.density
        numat = np.concatenate((nu_p-nu_r,nu_r-nu_p),axis=1)
        smat = np.vstack([numat[i] * self.molecular_weights[i] for i in range(self.n_species)])/rho
        #compute first row (temperature) of the matrix
        cp = self.cp_mass #[J/Kg K]
        hspec = self.standard_enthalpies_RT  #non-dimensional
        Hspec = ct.gas_constant * self.T * hspec #[J/Kmol]
        smatT = np.sum([- numat[i] * Hspec[i] for i in range(self.n_species)],axis=0)/(rho*cp)
        Smat = np.vstack((smatT,smat))
        return Smat[:-1,:]
        
    def R_vector(self):
        """ 2*Nr-long vector containing the rates of progress in [Kmol/m3/s]"""        
        rvec = np.concatenate((self.forward_rates_of_progress,self.reverse_rates_of_progress))
        return rvec
    
    """ ~~~~~~~~~~~~ JACOBIAN ~~~~~~~~~~~~~
    """
    def jac_pyJac(self):
        """Computes analytic Jacobian, using PyJac.
        Returns a N_s x N_s array [jac]. 
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
        Returns a N_s x N_s array [jac]. Input must be an instance of the CSPCantera class"""
        roundoff = np.finfo(float).eps
        sro = np.sqrt(roundoff)
        #setup the state vector
        T = self.T
        y = self.Y.copy()   #ns-long
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


    def jac_contribution_numeric(self):
        """Computes contributions of each reaction to numerical Jacobian.
        Given that g = Sr = Sum_k S_k r^k, it follows that      
        J(g) = Sum_k^(2nr) J_k, where J_k = Jac(S_k r^k)   
        S_k r^k is the product of the k-th column of the matrix S and the k-th 
        component of the vector r. 
        Returns a list of 2*Nr  (N_s x N_s) arrays [jacK]. Input must be an instance of the CSPCantera class"""
        roundoff = np.finfo(float).eps
        sro = np.sqrt(roundoff)
        ns = self.n_species
        nr = self.n_reactions
        #setup the state vector
        T = self.T
        y = self.Y   #ns-long
        Smat = self.generalized_Stoich_matrix()   # ns x 2nr
        rvec = self.R_vector()    # 2nr-long

        
        Smatp = np.zeros((ns,ns,2*nr))
        rvecp = np.zeros((ns,2*nr))       
        #evaluate Smat and Rvec in y+dy[i]
        for i in range(ns-1):
            dy = np.zeros(ns)
            dy[i] = max(sro*abs(y[i]),1e-8)
            self.set_unnormalized_mass_fractions(y+dy)
            Smatp[i+1] = self.generalized_Stoich_matrix()
            rvecp[i+1] = self.R_vector()
                    
        self.Y = y  #reset original Y
        dT = max(sro*abs(T),1e-3)
        self.TP = T+dT,self.P
        Smatp[0] = self.generalized_Stoich_matrix()
        rvecp[0] = self.R_vector()
 
        self.TP = T,self.P  #reset original T,P
        
        
        JacK = np.zeros((2*nr,ns,ns))
        #evaluate derivatives per each reaction
        for k in range(2*nr):
            jac2D = np.zeros((ns,ns))
            for i in range(ns-1):
                dy = np.zeros(ns)
                dy[i] = max(sro*abs(y[i]),1e-8)
                ydotp = Smatp[i+1,:,k]*rvecp[i+1,k]
                ydot  = Smat[:,k]*rvec[k]
                dydot = ydotp-ydot
                jac2D[:,i+1] = dydot/dy[i]
            
            ydotp = Smatp[0,:,k]*rvecp[0,k]
            ydot  = Smat[:,k]*rvec[k]
            dydot = ydotp-ydot
            dT = max(sro*abs(T),1e-3)
            jac2D[:,0] = dydot/dT
            JacK[k] = jac2D
        
        #to check for correctness, in main program:
        #jack = gas.jac_contribution_numeric()
        #jac=np.sum(jack,axis=0)
        #jacn = gas.jac_numeric()
        #np.allclose(jac,jacn,rtol=1e-8,atol=1e-12)
        
        return JacK
    
    
    
    """ ~~~~~~~~~~~~ OTHER JAC FORMULATIONS ~~~~~~~~~~~~~
    """

    def jacThermal(self):
        nv = self.nv
        R = ct.gas_constant
        hspec = self.standard_enthalpies_RT
        Hspec = hspec * R * self.T
        Wk = self.molecular_weights
        cp = self.cp_mass
        TJrow = Hspec[:-1] / ( Wk[:-1] * cp)
        TJcol = self.jac = self.jac_numeric()[1:nv,0]
        JacThermal = np.outer(TJcol,TJrow)
        return JacThermal
    
    def jacKinetic(self):
        nv = self.nv
        jacKinetic = self.jac = self.jac_numeric()[1:nv,1:nv] 
        return jacKinetic


    """ ~~~~~~~~~~~~ REAC NAMES ~~~~~~~~~~~~~
    """     
    def reaction_names(self):
        nr = self.n_reactions
        rnames = self.reaction_equations()
        reacnames = np.zeros(2*nr,dtype=object)
        reacnames[0:nr] = [s + ' (Fwd)' for s in rnames]
        reacnames[nr:2*nr] = [s + ' (Bwd)' for s in rnames]
        return reacnames