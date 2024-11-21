#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 17:51:37 2020

@author: Riccardo Malpica Galassi, Sapienza University, Roma, Italy
"""

import cantera as ct
import numpy as np
import PyCSP.Functions as csp


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


def integrate_batch_constP(gas,temperature,pressure,eqratio,FuComp,OxComp,tend, rtol=1e-9, atol=1e-15):
    """
    Integrates the constant-pressure batch reactor.

    Parameters:
        gas (ct.Solution): Cantera gas object.
        temperature (float): Initial temperature [K].
        pressure (float): Initial pressure [Pa].
        eqratio (float): Equivalence ratio.
        FuComp (str): Fuel composition.
        OxComp (str): Oxidizer composition.
        tend (float): End time for the simulation [s].
        rtol (float): Relative tolerance for the integrator.
        atol (float): Absolute tolerance for the integrator.
        
    Returns:
        ct.SolutionArray: Solution array containing states and additional data.
    """

    #set the gas state
    T = temperature
    P = pressure
    gas.TP = T, P
    gas.constP = P
    gas.set_equivalence_ratio(eqratio,FuComp,OxComp)
    
    
    #integrate ODE
    r = ct.IdealGasConstPressureReactor(gas)
    sim = ct.ReactorNet([r])
    sim.rtol = rtol
    sim.atol = atol
    states = ct.SolutionArray(gas, 1, extra={'t': [0.0], 'rhsT': [0.0]})
    
    sim.initial_time = 0.0
    while sim.time < tend:
        sim.step()
        states.append(r.thermo.state, t=sim.time, rhsT=gas.source[-1])
    return states


def integrate_batch_constV(gas,temperature,pressure,eqratio,FuComp,OxComp,tend, rtol=1e-9, atol=1e-15):
    """
    Integrates the constant-volume batch reactor.

    Parameters:
        gas (ct.Solution): Cantera gas object.
        temperature (float): Initial temperature [K].
        pressure (float): Initial pressure [Pa].
        eqratio (float): Equivalence ratio.
        FuComp (str): Fuel composition.
        OxComp (str): Oxidizer composition.
        tend (float): End time for the simulation [s].
        rtol (float): Relative tolerance for the integrator.
        atol (float): Absolute tolerance for the integrator.
        
    Returns:
        ct.SolutionArray: Solution array containing states and additional data.
    """
    #set the gas state
    T = temperature
    P = pressure
    gas.TP = T, P
    rho = gas.density
    gas.constRho = rho
    gas.set_equivalence_ratio(eqratio,FuComp,OxComp)
    
    
    #integrate ODE
    r = ct.IdealGasReactor(gas)
    sim = ct.ReactorNet([r])
    sim.rtol = rtol
    sim.atol = atol
    states = ct.SolutionArray(gas, 1, extra={'t': [0.0], 'rhsT': [0.0]})
    
    sim.initial_time = 0.0
    while sim.time < tend:
        sim.step()
        states.append(r.thermo.state, t=sim.time, rhsT=gas.source[-1])
    return states

def find_duplicate_reactions(R):
    """    

    Parameters
    ----------
    R : an instance of a Reaction object, e.g.:
    R = ct.Reaction.list_from_file("gri30.yaml", gas)
    R = ct.Reaction.listFromCti(open("path/to/gri30.cti").read())
    R = ct.Reaction.listFromXml(open("path/to/gri30.xml").read())

    Returns
    -------
    duplicates : a list of lists of duplicate reactions

    """
    duplicates = []
    for id,reac in zip(range(len(R)),R):
        if reac.duplicate:
            listdup = [idx for idx,r in zip(range(len(R)),R) if r.duplicate and reac.equation==r.equation]
            if listdup not in duplicates:
                duplicates.append(listdup)
    return duplicates            