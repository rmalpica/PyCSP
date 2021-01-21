# -*- coding: utf-8 -*-
"""
@author: Riccardo Malpica Galassi, Sapienza University, Roma, Italy
"""
import cantera as ct
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0,'../..')
import src.cspFunctions as csp
import src.utils as utils

#create gas from original mechanism file hydrogen.cti
gas = csp.CanteraCSP('hydrogen.cti')
#reorder the gas to match pyJac (N2 in last place)
n2_ind = gas.species_index('N2')
specs = gas.species()[:]
gas = csp.CanteraCSP(thermo='IdealGas', kinetics='GasKinetics',
        species=specs[:n2_ind] + specs[n2_ind + 1:] + [specs[n2_ind]],
        reactions=gas.reactions())

#set the gas state
T = 1000
P = ct.one_atm
#gas.TPX = T, P, "H2:2.0, O2:1, N2:3.76"
gas.TP = T, P
gas.set_equivalence_ratio(1.0, 'H2', 'O2:1, N2:3.76')
gas.jacobiantype='numeric'
gas.rtol=1.0e-2
gas.atol=1.0e-8


#equilibrium
#gas.equilibrate('HP')

#integrate ODE
r = ct.IdealGasConstPressureReactor(gas)
sim = ct.ReactorNet([r])
time = 0.0
states = ct.SolutionArray(gas, extra=['t'])


API = []
Ifast = []
Islow = []
classify = []
TSRAPI = []


sim.set_initial_time(0.0)
while sim.time < 0.001:
    sim.step()
    states.append(r.thermo.state, t=sim.time)
    print('%10.3e %10.3f %10.3f %14.6e' % (sim.time, r.T, r.thermo.P, r.thermo.u))
    api, ifast, islow, species_type = gas.calc_CSPindices(API=True,Impo=True,species_type=True)
    tsrapi = gas.calc_TSRindices()
    API.append(api)
    Ifast.append(ifast)
    Islow.append(islow)
    classify.append(species_type)
    TSRAPI.append(tsrapi)


API = np.array(API)
Ifast = np.array(Ifast)
Islow = np.array(Islow)
classify = np.array(classify)
TSRAPI = np.array(TSRAPI)



thr = 0.1
TSRreacIdx = np.unique(np.nonzero(np.abs(TSRAPI) > thr)[1])  #indexes of reactions with TSRapi > thr
TSRreac = TSRAPI[:,TSRreacIdx]
print('plotting TSR indices...')
fig, ax = plt.subplots(figsize=(8,4))
for idx in range(len(TSRreacIdx)):
    #color = next(ax._get_lines.prop_cycler)['color']
    ax.plot(TSRreac[:,idx], label=gas.reaction_names()[TSRreacIdx[idx]])
ax.set_xlabel('counter')
ax.set_ylabel('TSR API')
ax.grid(False)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.6, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()