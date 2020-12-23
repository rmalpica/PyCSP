#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 17:51:37 2020

@author: Riccardo Malpica Galassi, Sapienza University, Roma, Italy
"""

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