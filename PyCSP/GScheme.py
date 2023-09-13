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
        self._iter = 0
    
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
    def iter(self):
        return self._iter
        
            
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
        A,B,f,lam,tau = self.CSPcore(yold)
        
        #apply tail correction and advance to ysol
        self._Tc = TCorr(A,f,lam,self.T)
        ysol = self.tail_correction()
        self._y = ysol
        
        #calc new T
        self._T = self.CSPexhaustedModes(ysol,lam,A,B,tau,self.csprtolT,self.cspatolT)
        
        #calc integration time
        self._dt = smart_timescale(tau,self.factor,self.T,lam[self.T],self.dt)
        
        #calc new H
        self._H = self.CSPslowModes(ysol,lam,A,B,self.dt,self.csprtolH,self.cspatolH,self.T) 
        
        #advance in time active dynamics to ya
        print('integrating %d - %d = %d modes\n' %(self.H,self.T,len(ysol[self.T:self.H])))
        ya = self.RK4gsc(A,B)
        
        #calc Hc in ystar               
        self._y = ya
        f = self.CSPamplitudes(ya,B)
        self._Hc = HCorr(A,f,lam,self.H,self._gas.n_elements,self.dt)
        #apply head correction and advance to ystar
        ystar = self.head_correction()

        #calc Tc in ystar
        self._y = ystar
        f = self.CSPamplitudes(ystar,B)
        self._Tc = TCorr(A,f,lam,self.T)        
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
        fLamVec = dt*f[H:up]*(1 + 0.5 * dt * lam[H:up].real)    #first-order      #not sure if it's H or H+1
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
        #print("mode %d" %j)
        #print(ewt)
        Aj = Revec[j]                  #this mode right evec
        fj = f[j]                      #this mode amplitude
        lamj = evals[j].real           #this mode eigenvalue (real part)
        #print(evals)
        #print("evalj %e, dt %e, f %e" %(lamj,dt,fj))
        
        delw = delw + 0.5*dt*dt*Aj*fj*np.abs(lamj)    #contribution of j-th mode to all vars     
        #print(delw)       
        if np.any(np.abs(delw) > ewt):
            #print("inside if")
            if j==nModes:
                H = nModes  
            else:
                H = j+1 if (imPart[j] and imPart[j+1] and evals[j].real==evals[j+1].real) else j    #if j is the second of a pair, move fwd by 2                    
            return H

    #print("No modes are active")
    H = Tail   #if criterion is never verified, no modes are active.
    #print("-----------")
    return H

        
       
    
    
    
    