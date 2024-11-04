#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  11 15:41:22 2023

@author: Riccardo Malpica Galassi, Sapienza University, Roma, Italy
"""
import sys
import numpy as np
import PyCSP.Functions as cspF


class GScheme:
    def __init__(self, gas):   
        self._gas = gas
        self._y = []
        self._t = []
        self._Tc = []
        self._Hc = []
        self._dt = 1e+10
        self.csprtolT = 1e-2
        self.cspatolT = 1e-8
        self.csprtolH = 1e-4
        self.cspatolH = 1e-10
        self.jacobiantype = 'full'
        self._T = 0
        self._H = gas.nv - gas.n_elements 
        self.factor = 0.2
        self.reuseThr = 1.e-1
        self._iter = 0

        self.reuse = False
        self._diagJac = []
        self._normJac = 0.0
        self._skipBasis = 0
        self._updateBasis = 0
        self._basisStatus = 0
        
        self._lam = []
        self._A = []
        self._B = []
        self._f = []
        self._tau = []

    
    @property
    def csprtolT(self):
        return self._csprtolT
          
    @csprtolT.setter
    def csprtolT(self,value):
        self._csprtolT = value
    
    @property
    def cspatolT(self):
        return self._cspatolT
          
    @cspatolT.setter
    def cspatolT(self,value):
        self._cspatolT = value
             
    @property
    def csprtolH(self):
        return self._csprtolH
          
    @csprtolH.setter
    def csprtolH(self,value):
        self._csprtolH = value
    
    @property
    def cspatolH(self):
        return self._cspatolH
          
    @cspatolH.setter
    def cspatolH(self,value):
        self._cspatolH = value
 
    @property
    def jacobiantype(self):
        return self._jacobiantype
          
    @jacobiantype.setter
    def jacobiantype(self,value):
        if value == 'full':
            self._jacobiantype = value
        else:
            raise ValueError("Invalid jacobian type --> %s" %value)
    
    @property
    def Tc(self):
        return self._Tc  
        
    @property
    def Hc(self):
        return self._Hc   
    
    @property
    def y(self):
        return self._y

    @property
    def t(self):
        return self._t 
    
    @property
    def dt(self):
        return self._dt 

    @property
    def T(self):
        return self._T

    @property
    def H(self):
        return self._H 
    
    @property
    def factor(self):
        return self._factor
          
    @factor.setter
    def factor(self,value):
        self._factor = value

    @property
    def reuseThr(self):
        return self._reuseThr
          
    @reuseThr.setter
    def reuseThr(self,value):
        self._reuseThr = value
    
    @property
    def iter(self):
        return self._iter
    
    @property
    def reuse(self):
        return self._reuse
          
    @reuse.setter
    def reuse(self,value):
        self._reuse = value
 
    @property
    def normJac(self):
        return self._normJac 

    @normJac.setter
    def normJac(self,value):
        self._normJac = value

    @property
    def diagJac(self):
        return self._diagJac      
    
    @diagJac.setter
    def diagJac(self,value):
        self._diagJac = value

    @property
    def skipBasis(self):
        return self._skipBasis

    @skipBasis.setter
    def skipBasis(self,value):
        self._skipBasis = value
 
    @property
    def updateBasis(self):
        return self._updateBasis

    @updateBasis.setter
    def updateBasis(self,value):
        self._updateBasis = value
    
    @property
    def basisStatus(self):
        return self._basisStatus

    @basisStatus.setter
    def basisStatus(self,value):
        self._basisStatus = value

    @property
    def lam(self):
        return self._lam
        
    @property
    def A(self):
        return self._A
    
    @property
    def B(self):
        return self._B
    
    @property
    def f(self):
        return self._f
    
    @property
    def tau(self):
        return self._tau
    
    """ ### Methods ### """
            
    def CSPcore(self,y):
        self._gas.set_stateYT(y)
        if(self.reuse): self.normJac = self.calc_normJac(y)
        if(not self.reuse or self.iter < 3 or self.normJac > self.reuseThr or self.skipBasis > 250):
            self.skipBasis = 0
            self.basisStatus = 1
            self._lam,self._A,self._B,self._f = self._gas.get_kernel()
            self._tau = cspF.timescales(self.lam)
            self.updateBasis = self.updateBasis + 1
        else:
            self._f = np.matmul(self.B,self.rhs(y)) 
            self.basisStatus = 0
            self.skipBasis = self.skipBasis + 1
        
    
    
    def CSPexhaustedModes(self,y,lam,A,B,tau,rtol,atol):
        self._gas.set_stateYT(y)
        f = np.matmul(B,self._gas.source)
        M = cspF.findM(self._gas.n_elements,y,lam,A,tau,f,rtol,atol)
        return M
    
    def CSPslowModes(self,y,lam,A,B,dt,rtol,atol,Tail):
        self._gas.set_stateYT(y)
        f = np.matmul(B,self._gas.source)
        H = findH(self._gas.n_elements,y,lam,A,dt,f,rtol,atol,Tail)
        return H
    
    def CSPamplitudes(self,y,B):
        f = np.matmul(B,self.rhs(y))
        return f
    
    def rhs(self,y):
        self._gas.set_stateYT(y)
        dydt = np.zeros(self._gas.nv) 
        dydt = self._gas.source
        return dydt
        
    def set_integrator(self,**kwargs):
        for key, value in kwargs.items():
            if(key == 'cspRtolTail'): 
                self.csprtolT = value
            elif(key == 'cspAtolTail'):
                self.cspatolT = value
            elif(key == 'cspRtolHead'): 
                self.csprtolH = value
            elif(key == 'cspAtolHead'):
                self.cspatolH = value
            elif(key == 'factor'):
                self.factor = value
            elif(key == 'jacobiantype'):
                self.jacobiantype = value
            else:
                sys.exit('Unknown keyword in set_integrator')
                    
    def set_initial_value(self,y0,t0):
        self._y = y0
        self._t = t0
    
    def integrate(self):
        #calc CSP basis in yold
        self._gas.jacobiantype = self.jacobiantype
        yold = self.y
        self.CSPcore(yold)
        
        #apply tail correction and advance to ysol
        self._Tc = TCorr(self.A,self.f,self.lam,self.T)
        ysol = self.tail_correction()
        self._y = ysol
        
        #calc new T
        self._T = self.CSPexhaustedModes(ysol,self.lam,self.A,self.B,self.tau,self.csprtolT,self.cspatolT)
        
        #calc integration time
        self._dt = smart_timescale(self.tau,self.factor,self.T,self.lam[self.T],self.dt)
        
        #calc new H
        self._H = self.CSPslowModes(ysol,self.lam,self.A,self.B,self.dt,self.csprtolH,self.cspatolH,self.T) 
        
        #advance in time active dynamics to ya
        #print('integrating %d - %d = %d modes\n' %(self.H,self.T,len(ysol[self.T:self.H])))
        ya = self.RK4gsc(self.A,self.B)
        
        #calc Hc in ystar               
        self._y = ya
        f = self.CSPamplitudes(ya,self.B)
        self._Hc = HCorr(self.A,f,self.lam,self.H,self._gas.n_elements,self.dt)
        #apply head correction and advance to ystar
        ystar = self.head_correction()

        #calc Tc in ystar
        self._y = ystar
        f = self.CSPamplitudes(ystar,self.B)
        self._Tc = TCorr(self.A,f,self.lam,self.T)        
        #apply tail correction and advance to ynew
        ynew = self.tail_correction()
        
        self._y = ynew
        self._t = self.t + self.dt
        self._iter = self.iter + 1
    
                                
    def RK4gsc(self,A,B):
        def delcsi(csi):            
            y2 = self.y + np.matmul(np.transpose(A[self.T:self.H]),csi)
            dcsi = np.matmul(B[self.T:self.H],self.rhs(y2))
            return dcsi
        h = self.dt
        yn = self.y
        csi = np.zeros((self.H-self.T))
        csi2 = csi + 0.5*h*np.array(delcsi(csi))
        csi3 = csi + 0.5*h*np.array(delcsi(csi2))
        csi2 = csi2 + 2.0*csi3
        csi3 = csi + h*np.array(delcsi(csi3))
        csi2 = csi2 + csi3
        csi = (csi2 - csi + 0.5*h*np.array(delcsi(csi3))) / 3.0
        
        ynn = yn + np.matmul(np.transpose(A[self.T:self.H]),csi)
        return ynn   
    
    def tail_correction(self):
        ynew = self.y - self.Tc
        return ynew
        
    def head_correction(self):
        ynew = self.y + self.Hc
        return ynew
    
    def calc_normJac(self,y):
        diagOld = self.diagJac.copy()
        self.diagJac = self._gas.jacobian_diagonal
        normJac = 0.0
        if (self.iter > 3):
            dum = [(self.diagJac[i]-diagOld[i])/self.diagJac[i] if(np.abs(self.diagJac[i]) > 1e-20) else 0.0 for i in range(len(diagOld)-1)]
            normJac = np.linalg.norm(dum)
        #print('norm = %10.3e' % normJac)
        return normJac   
    

def TCorr(A,f,lam,T):
    ns = A.shape[0]
    Tc = np.zeros((ns))
    if(T > 0):
        #fLamVec = f[0:T]*(1.0/lam[0:T].real)   #linear approximation
        fLamVec = f[0:T]*(1 - np.exp((1.0/abs(lam[T].real))*lam[0:T].real) )/lam[0:T].real    #exponential decay approximation
        Tc = np.matmul(np.transpose(A[0:T]), fLamVec)
    return Tc
    
def HCorr(A,f,lam,H,nel,dt):
    ns = A.shape[0]
    Hc = np.zeros((ns))
    up = ns - nel
    if(H < up):
        fLamVec = dt*f[H:up]*(1 + 0.5 * dt * lam[H:up].real)    #first-order    
        Hc = np.matmul(np.transpose(A[H:up]), fLamVec)
    return Hc

def smart_timescale(tau,factor,T,lamT,dtold):
    if(T==0):
        dt = tau[T]*factor
    else:
        dt = tau[T-1] + factor*(tau[T]-tau[T-1])
    #if(lamT.real > 0):
    dt = min(dt,1.5*dtold)
    return dt

def findH(n_elements,stateYT,evals,Revec,dt,f,rtol,atol,Tail):
    nv = len(Revec)
    nEl = n_elements 
    #nconjpairs = sum(1 for x in self.eval.imag if x != 0)/2
    imPart = evals.imag!=0
    nModes = nv - nEl - 1   #removing conserved modes
    ewt = cspF.setEwt(stateYT,rtol,atol)    
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



        
       
    
    
    
    