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



def reorder_evals(ev):
    """    

    Parameters
    ----------
    evals : 2D numpy array containing a series of eigenvalues.
   
    Returns
    -------
    newev : 2D numpy array with continuous (along dim=0) eigenvalues


    """
    evals = ev.real.copy()
    nstep = evals.shape[0]
    nv = evals.shape[1]
    delta = np.zeros(nv)
    mask = np.zeros((nstep,nv), dtype=np.int8)
    mask[0] = np.arange(nv)
    
    for l in range(nv):
        chosen = evals[0,l]
        prev = chosen
        prevv = prev
        for i in range(1,nstep):
            if(i==1):
                for j in range(nv):
                    delta[j] = np.abs(prev-evals[i,j])
            else:
                for j in range(nv):
                    delta[j] = np.abs(prevv - 2*prev + evals[i,j])
            k = np.argmin(delta)
            chosen = evals[i,k]
            prevv = prev
            prev = chosen
            mask[i,l] = k
            evals[i,k] = 1.0e+30
    newev = np.zeros((nstep,nv),dtype=complex)
    for i in range(nstep):
            newev[i] = ev[i,mask[i]]
    return newev, mask