# -*- coding: utf-8 -*-
"""
@author: Riccardo Malpica Galassi, Sapienza University, Roma, Italy
"""
import cantera as ct
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import itertools
import PyCSP.Functions as csp


#create gas from original mechanism file hydrogen.cti
gas = csp.CanteraCSP('hydrogen.cti')

#set the gas state
T = 1000
P = ct.one_atm
#gas.TPX = T, P, "H2:2.0, O2:1, N2:3.76"
gas.TP = T, P
gas.set_equivalence_ratio(1.0, 'H2', 'O2:1, N2:3.76')

#push pressure
gas.constP = P

#edit CSP parameters
gas.jacobiantype='full'
gas.rtol=1.0e-3
gas.atol=1.0e-10


#integrate ODE
r = ct.IdealGasConstPressureReactor(gas)
sim = ct.ReactorNet([r])
time = 0.0
states = ct.SolutionArray(gas, extra=['t'])


API = []
TPI = []
Ifast = []
Islow = []
classify = []
TSRAPI = []
TSRTPI = []


sim.set_initial_time(0.0)
while sim.time < 0.001:
    sim.step()
    states.append(r.thermo.state, t=sim.time)
    print('%10.3e %10.3f %10.3f %14.6e' % (sim.time, r.T, r.thermo.P, r.thermo.u))
    api, tpi, ifast, islow, species_type = gas.calc_CSPindices(API=True,Impo=True,species_type=True,TPI=False)
    tsrapi = gas.calc_TSRindices(type='amplitude')
    tsrtpi = gas.calc_TSRindices(type='timescale')
    API.append(api)
    TPI.append(tpi)
    Ifast.append(ifast)
    Islow.append(islow)
    classify.append(species_type)
    TSRAPI.append(tsrapi)
    TSRTPI.append(tsrtpi)


API = np.array(API)
TPI = np.array(TPI)
Ifast = np.array(Ifast)
Islow = np.array(Islow)
classify = np.array(classify)
TSRAPI = np.array(TSRAPI)
TSRTPI = np.array(TSRTPI)


def plot(grid,cspindex,thr,gas,outname):
    
    print('plotting indices...')
    
    l_styles = ['-','--','-.',':']
    m_styles = ['','.','o','^','*']
    colormap = mpl.cm.Dark2.colors   # Qualitative colormap
    
    gridIdx = np.argsort(grid)
    reacIdx = np.unique(np.nonzero(np.abs(cspindex) > thr)[1])  #indexes of reactions with TSRapi > thr
    reac = cspindex[:,reacIdx][gridIdx]
    
    fig, ax = plt.subplots(figsize=(8,4))
    for idx,(marker,linestyle,color) in zip(range(len(reacIdx)),itertools.product(m_styles,l_styles, colormap)):
        plt.plot(grid[gridIdx], reac[:,idx], color=color, linestyle=linestyle,marker=marker,label=gas.reaction_names()[reacIdx[idx]])
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.);
 
    ax.set_xlabel('time [s]')
    ax.set_ylabel('Index')
    ax.grid(False)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.6, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show(block = False)
    plt.savefig(outname, dpi=800, transparent=False)
    
    
spec = -1
thr = 0.2
plot(states.t,Islow[:,spec,:],thr,gas,'IslowTemp.png')
plot(states.t,Ifast[:,spec,:],thr,gas,'IfastTemp.png')

spec = gas.species_names.index('H2O')
plot(states.t,Islow[:,spec,:],thr,gas,'IslowNO.png')
plot(states.t,Ifast[:,spec,:],thr,gas,'IfastNO.png')

plot(states.t,TSRAPI[:,:],thr,gas,'TSRAPI.png')
plot(states.t,TSRTPI[:,:],thr,gas,'TSRTPI.png')
