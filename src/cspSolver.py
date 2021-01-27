#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 12:26:32 2021

@author: Riccardo Malpica Galassi, Sapienza University, Roma, Italy
"""
import sys
import numpy as np
import src.cspFunctions as cspF


class CSPsolver:
    def __init__(self, gas):   
        self.gas = gas
        self.P = gas.P
        self.y = []
        self.t = []
        self.Qs = []
        self.Rc = []
        self.dt = 1e+10
        self.csprtol = 1e-2
        self.cspatol = 1e-8
        self.jacobiantype = 'full'
        self.M = 0
        self.factor = 0.2
    
    def CSPcore(self,y):
        self.gas.set_stateYT(y)
        lam,A,B,f = self.gas.get_kernel(jacobiantype=self.jacobiantype)
        M = self.gas.calc_exhausted_modes(rtol=self.csprtol, atol=self.cspatol)
        tau = cspF.timescales(lam)
        return [A,B,f,lam,tau,M]
    
    def CSPcoreBasic(self,y):
        self.gas.set_stateYT(y)
        lam,A,B,f = self.gas.get_kernel(jacobiantype=self.jacobiantype)
        return [A,B,f,lam]
    
    def rhs(self,y):
        self.gas.set_stateYT(y)
        dydt = np.zeros(self.gas.nv) 
        dydt = self.gas.rhs.copy()
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
        self.y = y0
        self.t = t0
    
    def integrate(self):
        #calc CSP basis and projection matrix Qs in yold
        yold = self.y
        A,B,f,lam,tau,M = self.CSPcore(yold)
        self.M = M
        self.Qs = QsMatrix(A,B,M)
        #advance in time dydt = Qsg with RK4 to ystar
        self.dt = smart_timescale(tau,self.factor,self.M,lam[M],self.dt)
        ystar = self.RK4csp()
        #calc CSP basis and radical matrix Rc in ystar
        self.y = ystar
        A,B,f,lam = self.CSPcoreBasic(ystar)
        self.Rc = RCorr(A,f,lam,self.M)
        #apply radical correction and advance to ynew
        ynew = self.radical_correction()
        self.y = ynew
        self.t = self.t + self.dt
    
                                
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
        lamMat = np.diag(lam[0:M])
        Rc = np.matmul(np.matmul(np.transpose(A[0:M]), np.linalg.inv(lamMat.real)), f[0:M])
    return Rc

def smart_timescale(tau,factor,M,lamM,dtold):
    if(M==0):
        dt = tau[M]*factor
    else:
        dt = tau[M-1] + factor*(tau[M]-tau[M-1])
    if(lamM.real > 0):
        dt = min(dt,1.5*dtold)
    return dt
        
       
    
    
    
    