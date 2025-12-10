#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 17:51:37 2020

@author: Riccardo Malpica Galassi, Sapienza University, Roma, Italy
"""

import cantera as ct
import numpy as np
import PyCSP.Functions as csp
import matplotlib.pyplot as plt
import matplotlib as mpl
import itertools
import os


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


class TaggedDatabase:
    """
    A simple database to store time series data during simulations.
    """
    def __init__(self):
        self._data = {}

    def store(self, **kwargs):
        """
        Store values for the given tags.
        Usage: db.store(time=0.1, temp=300, ...)
        """
        for key, value in kwargs.items():
            if key not in self._data:
                self._data[key] = []
            self._data[key].append(value)

    def to_arrays(self):
        """
        Convert all stored lists to numpy arrays.
        """
        for key in self._data:
            self._data[key] = np.array(self._data[key])
    
    def __getitem__(self, key):
        return np.array(self._data[key])
        
    def __getattr__(self, name):
        if name in self._data:
            return np.array(self._data[name])
        raise AttributeError(f"'TaggedDatabase' object has no attribute '{name}'")


def plot_index_over_time(time, data, temp, threshold, names, output_path, ylabel, title=None, xlim=None, xlabel='time [s]'):
    """
    Helper function to plot indices over time.
    data: 2D array (time, items)
    names: list of names for items
    threshold: relative threshold (fraction of max absolute value)
    """
    l_styles = ['-','--','-.',':']
    m_styles = ['s','.','o','^','*']
    colormap = mpl.cm.Dark2.colors   # Qualitative colormap
    
    # Check dimensions
    if data.ndim != 2:
        print(f"Warning: Data for {ylabel} has wrong dimensions {data.shape}. Skipping.")
        return

    # Calculate max absolute value for relative thresholding
    max_val = np.max(np.abs(data))
    if max_val == 0:
        return # Nothing to plot

    actual_threshold = threshold * max_val

    # Find indices of items that are significant
    significant_indices = np.unique(np.nonzero(np.abs(data) > actual_threshold)[1])
    
    if len(significant_indices) == 0:
        return # Nothing to plot

    # Sort time for plotting
    gridIdx = np.argsort(time)
    
    # Cycle through styles
    style_cycle = itertools.cycle(itertools.product(m_styles, l_styles, colormap))

    fig, ax1 = plt.subplots(figsize=(8,6))
    # Temperature (last column of state)
    ax1.plot(time, temp, 'grey', linestyle='-', label='Temperature')
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel('Temperature [K]', color='grey')
    ax1.tick_params(axis='y', labelcolor='grey')
    if xlim:
        ax1.set_xlim(xlim)
    ax2 = ax1.twinx()
    
    for idx in significant_indices:
        marker, linestyle, color = next(style_cycle)
        label_name = names[idx] if idx < len(names) else f"Index {idx}"
        ax2.plot(time[gridIdx], data[gridIdx, idx], color=color, linestyle=linestyle, marker=marker, markersize=2, label=label_name,markevery=int(0.02*len(time)))
    
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel(ylabel)
    if title:
        ax2.set_title(title)
    if xlim:
        ax2.set_xlim(xlim)
    ax2.grid(False)
    
    # Legend outside
    box = ax2.get_position()
    ax2.set_position([box.x0, box.y0, box.width * 0.6, box.height])
    labels = [line.get_label() for line in ax2.get_lines()]
    max_len = max(len(lbl) for lbl in labels)
    if max_len > 60: ax2.legend(
        loc='lower center',
        bbox_to_anchor=(0.5, 1.15)
    )
    else: ax2.legend(loc='center left', bbox_to_anchor=(1.15, 0.5))
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def save_CSP_plots(db, gas, output_folder='CSP_Plots', threshold=0.05, xlim=None, xlabel='time [s]', species_threshold=1e-3):
    """
    Generates and saves all requested CSP plots.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    reaction_names = gas.reaction_names()
    species_names = list(gas.species_names) + ['Temperature']
    mode_names = [f'Mode {i+1}' for i in range(gas.nv-gas.n_elements)]
    
    # 1. API (nvariables images)
    if hasattr(db, 'API'):
        folder = os.path.join(output_folder, 'API')
        os.makedirs(folder, exist_ok=True)
        print("Plotting API...")
        for i, mode_name in enumerate(mode_names):
            if i < db.API.shape[1]:
                plot_index_over_time(db.time, db.API[:, i, :], db.state[:, -1], threshold, reaction_names, 
                                     os.path.join(folder, f'API_{mode_name.replace(" ", "_")}.png'), 'Amplitude Participation Index', title=f'API for {mode_name}', xlim=xlim, xlabel=xlabel)

    # 2. Ifast (nvariables images)
    if hasattr(db, 'Ifast'):
        folder = os.path.join(output_folder, 'Ifast')
        os.makedirs(folder, exist_ok=True)
        print("Plotting Ifast...")
        for i, var_name in enumerate(species_names):
            if i < db.Ifast.shape[1]:
                plot_index_over_time(db.time, db.Ifast[:, i, :], db.state[:, -1], threshold, reaction_names, 
                                     os.path.join(folder, f'Ifast_{var_name}.png'), 'Ifast Index', title=f'Ifast for {var_name}', xlim=xlim, xlabel=xlabel)

    # 3. Islow (nvariables images)
    if hasattr(db, 'Islow'):
        folder = os.path.join(output_folder, 'Islow')
        os.makedirs(folder, exist_ok=True)
        print("Plotting Islow...")
        for i, var_name in enumerate(species_names):
            if i < db.Islow.shape[1]:
                plot_index_over_time(db.time, db.Islow[:, i, :], db.state[:, -1], threshold, reaction_names, 
                                     os.path.join(folder, f'Islow_{var_name}.png'), 'Islow Index', title=f'Islow for {var_name}', xlim=xlim, xlabel=xlabel)

    # 4. Pointers (nvariables images)
    if hasattr(db, 'Pointers'):
        folder = os.path.join(output_folder, 'Pointers')
        os.makedirs(folder, exist_ok=True)
        print("Plotting Pointers...")
        for i, var_name in enumerate(species_names):
            if i < db.Pointers.shape[2]:
                plot_index_over_time(db.time, db.Pointers[:, :gas.nv-gas.n_elements, i], db.state[:, -1], threshold, mode_names, 
                                     os.path.join(folder, f'Pointers_{var_name}.png'), 'Pointer', title=f'Pointers for {var_name}', xlim=xlim, xlabel=xlabel)

    # 5. TPI (nvariables images)
    if hasattr(db, 'TPI'):
        folder = os.path.join(output_folder, 'TPI')
        os.makedirs(folder, exist_ok=True)
        print("Plotting TPI...")
        for i, mode_name in enumerate(mode_names):
            if i < db.TPI.shape[1]:
                plot_index_over_time(db.time, db.TPI[:, i, :], db.state[:, -1], threshold, reaction_names, 
                                     os.path.join(folder, f'TPI_{mode_name.replace(" ", "_")}.png'), 'TPI Index', title=f'TPI for {mode_name}', xlim=xlim, xlabel=xlabel)

    # 6. TSRAPI (one image)
    if hasattr(db, 'tsrApi'):
        folder = os.path.join(output_folder, 'TSR')
        os.makedirs(folder, exist_ok=True)
        print("Plotting TSR API...")
        plot_index_over_time(db.time, db.tsrApi, db.state[:, -1], threshold, reaction_names, 
                             os.path.join(folder, 'TSR_API.png'), 'TSR API Index', title='TSR Amplitude Participation', xlim=xlim, xlabel=xlabel)

    # 7. TSRTPI (one image)
    if hasattr(db, 'tsrTpi'):
        folder = os.path.join(output_folder, 'TSR')
        os.makedirs(folder, exist_ok=True)
        print("Plotting TSR TPI...")
        plot_index_over_time(db.time, db.tsrTpi, db.state[:, -1], threshold, reaction_names, 
                             os.path.join(folder, 'TSR_TPI.png'), 'TSR TPI Index', title='TSR Timescale Participation', xlim=xlim, xlabel=xlabel)

    # 8. Eigenvalues
    if hasattr(db, 'evals') and hasattr(db, 'M'):
        print("Plotting Eigenvalues...")
        evalM = select_eval(db.evals, db.M)
        logevals = np.clip(np.log10(1.0+np.abs(db.evals)),0,100)*np.sign(db.evals.real)
        logevalM = np.clip(np.log10(1.0+np.abs(evalM.real)),0,100)*np.sign(evalM.real)
        
        fig, ax1 = plt.subplots(figsize=(8,5))
        # Temperature (last column of state)
        temp = db.state[:, -1]
        ax1.plot(db.time, temp, 'grey', linestyle='-', label='Temperature')
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel('Temperature [K]', color='grey')
        ax1.tick_params(axis='y', labelcolor='grey')
        if xlim:
            ax1.set_xlim(xlim)
        ax2 = ax1.twinx()
        for idx in range(db.evals.shape[1]):
            ax2.plot(db.time, logevals[:,idx], color='black', marker='.', markersize = 5, linestyle = 'None')
        ax2.plot(db.time, logevalM, color='orange', marker='.', markersize = 3, linestyle = 'None', label='lam(M+1)')
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel('log10(|evals|)')
        ax2.set_ylim([-10, 6])
        if xlim:
            ax2.set_xlim(xlim)
        ax2.grid(False)
        ax2.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'eigenvalues.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)

    # 9. Exhausted Modes
    if hasattr(db, 'M'):
        print("Plotting Exhausted Modes...")
        fig, ax1 = plt.subplots(figsize=(8,5))
        # Temperature (last column of state)
        temp = db.state[:, -1]
        ax1.plot(db.time, temp, 'grey', linestyle='-', label='Temperature')
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel('Temperature [K]', color='grey')
        ax1.tick_params(axis='y', labelcolor='grey')
        if xlim:
            ax1.set_xlim(xlim)
        ax2 = ax1.twinx()
        ax2.plot(db.time, db.M, color='orange',marker='s',markersize=2)
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel('# of exhausted modes (M)', color='orange')
        ax2.tick_params(axis='y', labelcolor='orange')
        ax2.set_ylim([0, gas.nv])
        if xlim:
            ax2.set_xlim(xlim)
        ax2.grid(False)
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'exhausted_modes.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)

    # 9bis. Slow Modes
    if hasattr(db, 'M'):
        print("Plotting Slow Modes...")
        fig, ax1 = plt.subplots(figsize=(8,5))
        # Temperature (last column of state)
        temp = db.state[:, -1]
        ax1.plot(db.time, temp, 'grey', linestyle='-', label='Temperature')
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel('Temperature [K]', color='grey')
        ax1.tick_params(axis='y', labelcolor='grey')
        if xlim:
            ax1.set_xlim(xlim)
        ax2 = ax1.twinx()
        ax2.plot(db.time, gas.nv-gas.n_elements-db.M, color='red',marker='s',markersize=2)
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel('SIM dimension', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.set_ylim([0, gas.nv])
        if xlim:
            ax2.set_xlim(xlim)
        ax2.grid(False)
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'slow_modes.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)

    # 10. Amplitudes
    if hasattr(db, 'fvec'):
        print("Plotting Amplitudes...")
        fig, ax1 = plt.subplots(figsize=(8,5))
        # Temperature (last column of state)
        temp = db.state[:, -1]
        ax1.plot(db.time, temp, 'grey', linestyle='-', label='Temperature')
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel('Temperature [K]', color='grey')
        ax1.tick_params(axis='y', labelcolor='grey')
        if xlim:
            ax1.set_xlim(xlim)
        ax2 = ax1.twinx()
        for idx in range(db.fvec.shape[1]):
            ax2.plot(db.time, np.log10(1e-10+db.fvec[:,idx]), label=f'Mode {idx+1}', marker='.', markersize = 2, linestyle = 'None')
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel('log10(Amplitude)')
        ax2.set_ylim([-8, 1+np.max(np.log10(1e-10+db.fvec))])
        if xlim:
            ax2.set_xlim(xlim)
        ax2.grid(False)
        if db.fvec.shape[1] < 10: ax2.legend(loc='center left', bbox_to_anchor=(1.15, 0.5)) # Legend might be too crowded if many modes
        plt.tight_layout() # Adjust layout to make room for legend
        plt.savefig(os.path.join(output_folder, 'amplitudes.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)

    # 11. State Evolution
    if hasattr(db, 'state'):
        print("Plotting State Evolution...")
        fig, ax1 = plt.subplots(figsize=(8,5))
        
        # Temperature (last column of state)
        temp = db.state[:, -1]
        ax1.plot(db.time, temp, 'r-', label='Temperature',marker='o',markersize=2,markevery=int(0.02*len(db.time)))
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel('Temperature [K]', color='r')
        ax1.tick_params(axis='y', labelcolor='r')
        if xlim:
            ax1.set_xlim(xlim)
            
        ax2 = ax1.twinx()
        
        # Species Mass Fractions
        # Filter for major species
        species_data = db.state[:, :-1]
        max_fractions = np.max(species_data, axis=0)
        major_indices = np.where(max_fractions > species_threshold)[0]
        
        l_styles = ['-','--','-.',':']
        m_styles = ['','.','o','^','*']
        colormap = mpl.cm.Dark2.colors
        style_cycle = itertools.cycle(itertools.product(m_styles, l_styles, colormap))
        
        for idx in major_indices:
            marker, linestyle, color = next(style_cycle)
            label_name = gas.species_names[idx]
            # Use log scale for species, clip to avoid log(0)
            ax2.plot(db.time, species_data[:, idx], color=color, linestyle=linestyle, marker=marker, label=label_name)
            
        ax2.set_ylabel('Mass Fraction', color='k')
        ax2.set_yscale('log')
        ax2.set_ylim(bottom=1e-6) # Set a reasonable lower bound for log scale
        ax2.tick_params(axis='y', labelcolor='k')
        
        # Combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='center left', bbox_to_anchor=(1.15, 0.5))
        
        plt.tight_layout() # Adjust layout to make room for legend
        plt.savefig(os.path.join(output_folder, 'state_evolution.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)