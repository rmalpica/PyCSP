# -*- coding: utf-8 -*-
"""
@author: Riccardo Malpica Galassi, Sapienza University, Roma, Italy
"""
import cantera as ct
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr
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
    tsr, tsrapi = gas.calc_TSRindices(type='amplitude')
    tsr, tsrtpi = gas.calc_TSRindices(type='timescale')
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


c_list = [np.random.choice(list(clr.CSS4_COLORS.values())) for i in range(10*gas.n_reactions)] 


thr = 0.1
TSRreacIdx = np.unique(np.nonzero(np.abs(TSRAPI) > thr)[1])  #indexes of reactions with TSRapi > thr
TSRreac = TSRAPI[:,TSRreacIdx]
print('plotting TSR-API indices...')
fig, ax = plt.subplots(figsize=(8,4))
for idx in range(len(TSRreacIdx)):
    #color = next(ax._get_lines.prop_cycler)['color']
    ax.plot(TSRreac[:,idx], c=c_list[TSRreacIdx[idx]] , label=gas.reaction_names()[TSRreacIdx[idx]], linestyle='-')
ax.set_xlabel('counter')
ax.set_ylabel('TSR API')
ax.grid(False)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.6, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show(block = False)
plt.savefig('TSR-API.png', dpi=800, transparent=False)

thr = 0.1
TSRreacIdx = np.unique(np.nonzero(np.abs(TSRTPI) > thr)[1])  #indexes of reactions with TSRapi > thr
TSRreac = TSRTPI[:,TSRreacIdx]
print('plotting TSR-TPI indices...')
fig, ax = plt.subplots(figsize=(8,4))
for idx in range(len(TSRreacIdx)):
    #color = next(ax._get_lines.prop_cycler)['color']
    ax.plot(TSRreac[:,idx], c=c_list[TSRreacIdx[idx]] ,label=gas.reaction_names()[TSRreacIdx[idx]], linestyle='-')
ax.set_xlabel('counter')
ax.set_ylabel('TSR TPI')
ax.grid(False)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.6, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show(block = False)
plt.savefig('TSR-TPI.png', dpi=800, transparent=False)