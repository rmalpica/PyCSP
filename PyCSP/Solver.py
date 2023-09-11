#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 12:26:32 2021

@author: Riccardo Malpica Galassi, Sapienza University, Roma, Italy
"""
import sys
import numpy as np
import PyCSP.Functions as cspF


class CSPsolver:
    def __init__(self, gas):   
        self._gas = gas
        self._y = []
        self._t = []
        self._Qs = []
        self._Rc = []
        self._dt = 1e+10
        self.csprtol = 1e-2
        self.cspatol = 1e-8
        self.jacobiantype = 'full'
        self._M = 0
        self.factor = 0.2
    
    @property
    def csprtol(self):
        return self._csprtol
          
    @csprtol.setter
    def csprtol(self,value):
        self._csprtol = value
    
    @property
    def cspatol(self):
        return self._cspatol
          
    @cspatol.setter
    def cspatol(self,value):
        self._cspatol = value
             
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
    def Qs(self):
        return self._Qs      
    
    @property
    def Rc(self):
        return self._Rc   
    
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
    def M(self):
        return self._M 
    
    @property
    def factor(self):
        return self._factor
          
    @factor.setter
    def factor(self,value):
        self._factor = value
        
            
    def CSPcore(self,y):
        self._gas.set_stateYT(y)
        lam,A,B,f = self._gas.get_kernel()
        tau = cspF.timescales(lam)
        return [A,B,f,lam,tau]
    
    
    def CSPexhaustedModes(self,y,lam,A,B,tau,rtol,atol):
        self._gas.set_stateYT(y)
        f = np.matmul(B,self._gas.source)
        M = cspF.findM(self._gas.n_elements,y,lam,A,tau,f,rtol,atol)
        return M
    
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
            if(key == 'cspRtol'): 
                self.csprtol = value
            elif(key == 'cspAtol'):
                self.cspatol = value
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
        A,B,f,lam,tau = self.CSPcore(yold)
        #apply radical correction and advance to yman
        self._Rc = RCorr(A,f,lam,self.M)
        yman = self.radical_correction()
        self._y = yman
        #calc new M
        self._M = self.CSPexhaustedModes(yman,lam,A,B,tau,self.csprtol,self.cspatol)
        #calc Projection matrix with old basis and new M
        self._Qs = QsMatrix(A,B,self.M)
        #advance in time dydt = Qsg with RK4 to ystar
        self._dt = smart_timescale(tau,self.factor,self.M,lam[self.M],self.dt)
        ystar = self.RK4csp()
        #calc Rc in ystar
        self._y = ystar
        f = self.CSPamplitudes(ystar,B)
        self._Rc = RCorr(A,f,lam,self.M)
        #apply radical correction and advance to ynew
        ynew = self.radical_correction()
        self._y = ynew
        self._t = self.t + self.dt
    
                                
    def RK4csp(self):
        def f(y):
            Qsg = np.matmul(self.Qs, self.rhs(y))
            return Qsg
        yn = self.y
        h = self.dt
        k1 = np.array(f(yn))
        k2 = np.array(f(yn + h/2.0 * k1))
        k3 = np.array(f(yn + h/2.0 * k2))
        k4 = np.array(f(yn + h * k3))
        ynn = yn + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return ynn   
    
    def radical_correction(self):
        ynew = self.y - self.Rc
        return ynew
    
    
def QsMatrix(A,B,M):
    ns = A.shape[0]
    if(M > 0):
        QsMat = np.identity(ns) - sum( [np.outer(A[i],B[i]) for i in range(M)])
    else:
        QsMat = np.identity(ns)
    return QsMat

def RCorr(A,f,lam,M):
    ns = A.shape[0]
    Rc = np.zeros((ns))
    if(M > 0):
        #fLamVec = f[0:M]*(1.0/lam[0:M].real)   #linear approximation
        fLamVec = f[0:M]*(1 - np.exp((1.0/abs(lam[M].real))*lam[0:M].real) )/lam[0:M].real    #exponential decay approximation
        Rc = np.matmul(np.transpose(A[0:M]), fLamVec)
    return Rc

def smart_timescale(tau,factor,M,lamM,dtold):
    if(M==0):
        dt = tau[M]*factor
    else:
        dt = tau[M-1] + factor*(tau[M]-tau[M-1])
    #if(lamM.real > 0):
    dt = min(dt,1.5*dtold)
    return dt
        
       
    
    
    
    