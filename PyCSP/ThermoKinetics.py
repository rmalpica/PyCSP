#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Riccardo Malpica Galassi, Sapienza University, Roma, Italy
"""
import numpy as np
import cantera as ct


class CanteraThermoKinetics(ct.Solution):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  

        self.constP = 0.0001
        self.constRho = 0.0001
        self._problemtype = 'unset'
        self._nv = self.n_species + 1
        self._source = []
        self._jacobian = []
        self._generalized_Stoich_matrix = []
        self._R_vector = []
        self._jacobian_diagonal = []


    """ ~~~~~~~~~~~~ PROPERTIES ~~~~~~~~~~~~~
    """
    @property
    def problemtype(self):
        return self._problemtype

    
    @property
    def constP(self):
        if(self.problemtype != 'const_p'):
            raise ValueError("Constant pressure is unset")
        else:
            return self._constP
          
    @constP.setter
    def constP(self,value):
        if value > 0:
            self._constP = value
            self._problemtype = 'const_p'
        else:
            raise ValueError("Pressure must be positive")

    @property
    def constRho(self):
        if(self.problemtype != 'const_v'):
            raise ValueError("Constant density is unset")
        else:
            return self._constRho
          
    @constRho.setter
    def constRho(self,value):
        if value > 0:
            self._constRho = value
            self._problemtype = 'const_v'
        else:
            raise ValueError("Density must be positive")
    
    @property
    def nv(self):
        return self._nv
    
    @nv.setter
    def nv(self,value):
        if self.n_species <= value <= self.n_species+1:
            self._nv = value
        else:
            raise ValueError("Number of variables must be Ns or Ns+1")
       
    @property
    def source(self):
        if (self.problemtype == 'const_p'):        
            return self.rhs_const_p()
        elif (self.problemtype == 'const_v'):
            return self.rhs_const_v()
        else:
            raise ValueError("Need to set either constP or constRho value")


    @property
    def jacobian(self):
        if (self.problemtype == 'const_p'):
            return self.jacobian_const_p()
        elif (self.problemtype == 'const_v'):
            return self.jacobian_const_v()
        else:
            raise ValueError("Need to set either constP or constRho value")

    @property                    
    def generalized_Stoich_matrix(self):
        if (self.problemtype == 'const_p'):
            return self.generalized_Stoich_matrix_const_p()
        elif (self.problemtype == 'const_v'):
            return self.generalized_Stoich_matrix_const_v()
        else:
            raise ValueError("Need to set either constP or constRho value")
              
    @property                    
    def R_vector(self):
        return self.Rates_vector()

    @property
    def jacobian_diagonal(self):
        if (self.problemtype == 'const_p'):
            return self.jacobian_diagonal_const_p()
        elif (self.problemtype == 'const_v'):
            return self.jacobian_diagonal_const_v()
        else:
            raise ValueError("Need to set either constP or constRho value")


    """ ~~~~~~~~~~~~ METHODS ~~~~~~~~~~~~~
    """

    
    """ ~~~~~~~~~~~~ state ~~~~~~~~~~~~~
    """
    
    def set_stateYT(self,y):
        if (self.problemtype == 'const_p'):
            return self.set_stateYT_const_p(y)
        elif (self.problemtype == 'const_v'):
            return self.set_stateYT_const_v(y)
        else:
            raise ValueError("Need to set either constP or constRho value")
    
    def stateYT(self):
        y = np.zeros(self.n_species+1)
        y[-1] = self.T
        y[0:-1] = self.Y
        return y

        
    def set_stateYT_const_p(self,y):
        self.Y = y[0:-1]
        self.TP = y[-1],self.constP

    def set_stateYT_const_v(self,y):
        self.Y = y[0:-1]
        self.TD = y[-1],self.constRho            
            
    """ ~~~~~~~~~~~~ rhs ~~~~~~~~~~~~~
    """
    def rhs_const_p(self):
        """Computes chemical RHS [shape:(ns+1)] for a constant pressure reactor. 
        """
        
        ns = self.n_species    
        ydot = np.zeros(ns+1)
        Wk = self.molecular_weights
        R = ct.gas_constant
        
        wdot = self.net_production_rates    #[kmol/m^3/s]
        orho = 1./self.density
        
        ydot[-1] = - R * self.T * np.dot(self.standard_enthalpies_RT, wdot) * orho / self.cp_mass
        ydot[0:-1] = wdot * Wk * orho
        return ydot
    
    
    def rhs_const_v(self):
        """Computes chemical RHS [shape:(ns+1)] for a constant volume reactor. 
        """
        
        ns = self.n_species    
        ydot = np.zeros(ns+1)
        Wk = self.molecular_weights
        R = ct.gas_constant
        
        wdot = self.net_production_rates    #[kmol/m^3/s]
        orho = 1./self.density
        cp = self.cp_mass
        cv = self.cv_mass
        wmix = self.mean_molecular_weight
        
        gamma = cp / cv
        
        ydot[-1] = - R * self.T * np.dot(self.standard_enthalpies_RT, wdot) * gamma * orho / cp + ( (gamma - 1.0) * self.T * wmix * np.sum(wdot) * orho )
        ydot[0:-1] = wdot * Wk * orho
        return ydot
    
    """ ~~~~~~~~~~~~ Stoichiometric matrix and Rates vector ~~~~~~~~~~~~~
    """
    
    def generalized_Stoich_matrix_const_p(self):
        """N_v x 2*N_r matrix containing the S components in column major format, 
        such that S dot Rvec yields RHS"""
        nu_p = self.product_stoich_coeffs
        nu_r = self.reactant_stoich_coeffs
        rho = self.density
        numat = np.concatenate((nu_p-nu_r,nu_r-nu_p),axis=1)
        smat = np.vstack([numat[i] * self.molecular_weights[i] for i in range(self.n_species)])/rho
        #compute last row (temperature) of the matrix
        cp = self.cp_mass #[J/Kg K]
        hspec = self.standard_enthalpies_RT  #non-dimensional
        Hspec = ct.gas_constant * self.T * hspec #[J/Kmol]
        smatT = np.sum([- numat[i] * Hspec[i] for i in range(self.n_species)],axis=0)/(rho*cp)
        Smat = np.vstack((smat,smatT))
        return Smat[:self.nv]
    

    def generalized_Stoich_matrix_const_v(self):
        """N_v x 2*N_r matrix containing the S components in column major format, 
        such that S dot Rvec yields RHS"""
        Wk = self.molecular_weights
        R = ct.gas_constant
        cp = self.cp_mass
        cv = self.cv_mass
        wmix = self.mean_molecular_weight
        gamma = cp / cv
        
        nu_p = self.product_stoich_coeffs
        nu_r = self.reactant_stoich_coeffs
        rho = self.density
        
        c1g = gamma/(rho*cp)
        c2g = (gamma-1.0)*self.T*wmix/rho
        
        numat = np.concatenate((nu_p-nu_r,nu_r-nu_p),axis=1)
        smat = np.vstack([numat[i] * Wk[i] for i in range(self.n_species)])/rho
        #compute last row (temperature) of the matrix
        cp = self.cp_mass #[J/Kg K]
        hspec = self.standard_enthalpies_RT  #non-dimensional
        Hspec = R * self.T * hspec #[J/Kmol]
        smatT = np.sum([numat[i] * (-Hspec[i] * c1g + c2g) for i in range(self.n_species)],axis=0)
        Smat = np.vstack((smat,smatT))
        return Smat[:self.nv]
    
    def Rates_vector(self):
        """ 2*Nr-long vector containing the rates of progress in [Kmol/m3/s]"""        
        rvec = np.concatenate((self.forward_rates_of_progress,self.reverse_rates_of_progress))
        return rvec
    
    """ ~~~~~~~~~~~~ jacobian ~~~~~~~~~~~~~
    All Jacobians are forward finite differences in the variables (Y_1..Y_Ns, T).
    Derivatives are taken holding the thermodynamic constraint of the problem
    fixed: pressure for const_p, density for const_v. Mass fractions are
    perturbed without renormalisation.
    """

    @staticmethod
    def _fd_steps(y, T):
        sro = np.sqrt(np.finfo(float).eps)
        dy = np.maximum(sro*np.abs(y), 1e-8)
        dT = max(sro*abs(T), 1e-3)
        return dy, dT

    def _set_constrained_state(self, y, T, P, rho, problemtype):
        """Sets (unnormalised) mass fractions y and temperature T, holding
        P (const_p) or rho (const_v). Note that set_unnormalized_mass_fractions
        alone holds the density, hence P must be reset explicitly."""
        self.set_unnormalized_mass_fractions(y)
        if problemtype == 'const_p':
            self.TP = T, P
        else:
            self.TD = T, rho

    def _numerical_jacobian(self, problemtype):
        rhs = self.rhs_const_p if problemtype == 'const_p' else self.rhs_const_v
        T = self.T
        P = self.P
        rho = self.density
        y = self.Y.copy()   #ns-long
        ydot = rhs()        #ns+1-long (Y1,...,Yn,T)
        dy, dT = self._fd_steps(y, T)

        jac2D = np.zeros((self.n_species+1, self.n_species+1))
        for i in range(self.n_species):
            yp = y.copy()
            yp[i] += dy[i]
            self._set_constrained_state(yp, T, P, rho, problemtype)
            jac2D[:,i] = (rhs()-ydot)/dy[i]

        self._set_constrained_state(y, T+dT, P, rho, problemtype)
        jac2D[:,-1] = (rhs()-ydot)/dT

        self._set_constrained_state(y, T, P, rho, problemtype)
        return jac2D

    def jacobian_const_p(self):
        """Computes numerical Jacobian at constant pressure.
        Returns a N_s+1 x N_s+1 array [jac]."""
        return self._numerical_jacobian('const_p')

    def jacobian_const_v(self):
        """Computes numerical Jacobian at constant density.
        Returns a N_s+1 x N_s+1 array [jac]."""
        return self._numerical_jacobian('const_v')


    def jac_contribution(self):
        """Computes contributions of each reaction to numerical Jacobian.
        Given that g = Sr = Sum_k S_k r^k, it follows that
        J(g) = Sum_k^(2nr) J_k, where J_k = Jac(S_k r^k)
        S_k r^k is the product of the k-th column of the matrix S and the k-th
        component of the vector r.
        Returns a (2*Nr, nv, nv) array [jacK], whose sum over the first axis
        equals the (full or kinetic) Jacobian."""
        problemtype = self.problemtype
        if problemtype not in ('const_p', 'const_v'):
            raise ValueError("Need to set either constP or constRho value")
        nv = self.nv
        ns = self.n_species
        nr = self.n_reactions
        T = self.T
        P = self.P
        rho = self.density
        y = self.Y.copy()   #ns-long
        dy, dT = self._fd_steps(y, T)
        Sr = self.generalized_Stoich_matrix * self.R_vector   # nv x 2nr, column k = S_k r^k

        JacK = np.zeros((2*nr,nv,nv))
        for i in range(ns):
            yp = y.copy()
            yp[i] += dy[i]
            self._set_constrained_state(yp, T, P, rho, problemtype)
            Srp = self.generalized_Stoich_matrix * self.R_vector
            JacK[:,:,i] = np.transpose((Srp - Sr)/dy[i])

        if nv == ns+1:
            self._set_constrained_state(y, T+dT, P, rho, problemtype)
            Srp = self.generalized_Stoich_matrix * self.R_vector
            JacK[:,:,-1] = np.transpose((Srp - Sr)/dT)

        self._set_constrained_state(y, T, P, rho, problemtype)
        return JacK


    def _numerical_jacobian_diagonal(self, problemtype):
        """Approximates the Jacobian diagonal with one simultaneous perturbation
        of all variables (cheap, used by the G-Scheme basis-reuse heuristic)."""
        rhs = self.rhs_const_p if problemtype == 'const_p' else self.rhs_const_v
        T = self.T
        P = self.P
        rho = self.density
        y = self.Y.copy()
        ydot = rhs()
        dy, dT = self._fd_steps(y, T)
        self._set_constrained_state(y+dy, T+dT, P, rho, problemtype)
        dydot = rhs()-ydot
        diagjac = np.zeros(self.n_species+1)
        diagjac[:-1] = dydot[:-1]/dy
        diagjac[-1] = dydot[-1]/dT
        self._set_constrained_state(y, T, P, rho, problemtype)
        return diagjac

    def jacobian_diagonal_const_p(self):
        """Computes (an approx. to) the diagonal of the numerical Jacobian at
        constant pressure. Returns a N_s+1 array [diagjac]."""
        return self._numerical_jacobian_diagonal('const_p')

    def jacobian_diagonal_const_v(self):
        """Computes (an approx. to) the diagonal of the numerical Jacobian at
        constant density. Returns a N_s+1 array [diagjac]."""
        return self._numerical_jacobian_diagonal('const_v')

    """ ~~~~~~~~~~~~ OTHER JAC FORMULATIONS ~~~~~~~~~~~~~
    """

    def jacThermal(self):
        """Correction that constrains the species Jacobian to the manifold of
        constant enthalpy (const_p) or constant internal energy (const_v):
        J_constrained = J_YY - jacThermal, with
        jacThermal[i,j] = dg_i/dT * e_j/c, where e_j = h_j, c = cp (const_p)
        or e_j = u_j, c = cv (const_v), mass-specific species properties."""
        ns = self.n_species
        RT = ct.gas_constant * self.T
        Wk = self.molecular_weights
        if self.problemtype == 'const_v':
            espec = self.standard_int_energies_RT * RT / Wk
            c = self.cv_mass
        else:
            espec = self.standard_enthalpies_RT * RT / Wk
            c = self.cp_mass
        TJrow = espec / c
        TJcol = self.jacobian[0:ns,-1]
        JacThermal = np.outer(TJcol,TJrow)
        return JacThermal
    
    def jacKinetic(self):
        ns = self.n_species
        jacKinetic = self.jacobian[0:ns,0:ns] 
        return jacKinetic


    def jacSIM(self,Jac,major_ind,Levec):
        nv = len(Jac)
        n = len(major_ind)
        m = nv - n
       
        tot=[*range(0, nv, 1)]
        minor_ind = list(set(tot) - set(major_ind)) + list(set(major_ind) - set(tot))
        mask = minor_ind + major_ind
        sortedJac = Jac[mask][:,mask]
        
        # Build Jacobian constrained to SIM 
    
        #First term: dg_i/du_j (n-by-n)    
        mat1 = sortedJac[m:nv,m:nv]
        
        #Second term: dg_i/dv_alpha (n-by-m)
        mat2 = sortedJac[m:nv,0:m]
   
        #Third term: (m-by-m) inverse of mat3 = b^i * dg/dx_j  where j spans the fast variables
        mat3inv=np.linalg.inv( np.matmul(Levec[0:m],Jac[:][:,mask][:,0:m]) )  
    
        #Fourth term: (m-by-(n-m)) matrix of b^i * dg/dx_k  where k spans the slow variables
        mat4 = np.matmul(Levec[0:m],Jac[:][:,mask][:,m:nv])
       
        # Constrained  (to SIM) jacobian: mat1 - mat2*mat3inv*mat4
        jacSIM = mat1 - np.matmul(mat2,np.matmul(mat3inv,mat4))
    
        return jacSIM
    
    
    """ ~~~~~~~~~~~~ REAC NAMES ~~~~~~~~~~~~~
    """     
    def reaction_names(self):
        nr = self.n_reactions
        rnames = self.reaction_equations()
        reacnames = np.zeros(2*nr,dtype=object)
        reacnames[0:nr] = ['(Rf-'+str(i+1)+') '+ s for s,i in zip(rnames,range(nr)) ]
        reacnames[nr:2*nr] = ['(Rb-'+str(i+1)+') '+ s for s,i in zip(rnames,range(nr))]
        return reacnames
