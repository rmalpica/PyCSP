#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 17:51:37 2020

@author: Riccardo Malpica Galassi, Sapienza University, Roma, Italy
"""

import numpy as np

def select_eval(evals,indexes):
    """    

    Parameters
    ----------
    evals : 2D numpy array containing a series of eigenvalues.
    indexes : 1D numpy array with integer indexes (e.g. M). Note that indexes start from zero!

    Returns
    -------
    eval : 1D numpy array with selected eigenvalues


    """
    eval = evals[range(evals.shape[0]),indexes]
    return eval



def reorder_evals(ev,times):
    """    

    Parameters
    ----------
    evals : 2D numpy array containing a series of eigenvalues.
    times : 1D numpy array containing the corresponding times
   
    Returns
    -------
    newev : 2D numpy array with continuous (along dim=0) eigenvalues


    """

    evals = ev.real.copy()
    evalsd = ev.real.copy()
    img = ev.imag.copy()
    nstep = evals.shape[0]
    nv = evals.shape[1]
    delta = np.zeros(nv)
    mask = np.zeros((nstep,nv), dtype=np.int8)
    mask[0] = np.arange(nv)
    mask[1] = np.arange(nv)

    
    for i in range(2,nstep):
        for l in range(nv):

            f0 = evalsd[i-2,mask[i-2]]
            f1 = evalsd[i-1,mask[i-1]]
            h1 = times[i-1]-times[i-2]
            h2 = times[i]-times[i-1]

            for j in range(nv):
                delta[j] = np.abs( 2* (h2*f0[l] - (h1+h2)*f1[l]  + h1*evals[i,j]) / ( h1*h2*(h1+h2))  )
            k = np.argmin(delta) 
            mask[i,l] = k
            evals[i,k] = 1.0e+30
    
    newev = np.zeros((nstep,nv),dtype=complex)
    for i in range(nstep):
            newev[i] = ev[i,mask[i]]
    return newev, mask