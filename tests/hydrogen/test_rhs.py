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

#equilibrium
#gas.equilibrate('HP')

#integrate ODE
r = ct.IdealGasConstPressureReactor(gas)
sim = ct.ReactorNet([r])
time = 0.0
states = ct.SolutionArray(gas, extra=['t'])

RHS = []
splitRHS = []
varnames = np.array(['Temperature']+gas.species_names)[:-1]
sim.set_initial_time(0.0)
while sim.time < 1.5e-3:
    sim.step()
    states.append(r.thermo.state, t=sim.time)
    print('%10.3e %10.3f %10.3f %14.6e' % (sim.time, r.T, r.thermo.P, r.thermo.u))
    rhs = gas.rhs_const_p_pyJac()
    checkrhs = np.isclose(rhs, gas.rhs_const_p(), rtol=1e-6, atol=1e-6, equal_nan=False)
    if(np.any(checkrhs == False)):
        idx = np.array([*range(len(rhs))]) 
        print('Mismatch between analytical and numerical RHS')
        print(varnames[~checkrhs],rhs[~checkrhs],gas.rhs_const_p()[~checkrhs])
    Smat = gas.generalized_Stoich_matrix()
    rvec = gas.R_vector()
    splitrhs = np.dot(Smat,rvec)
    checksplitrhs = np.isclose(gas.rhs_const_p(), splitrhs, rtol=1e-6, atol=0, equal_nan=False)
    if(np.any(checkrhs == False)):
        idx = np.array([*range(len(rhs))]) 
        print('Mismatch between numerical RHS and S.r')
        print(varnames[~checkrhs],gas.rhs_const_p()[~checkrhs],splitrhs[~checkrhs])
    RHS.append(rhs)
    splitRHS.append(splitrhs)


RHS = np.array(RHS)
splitRHS = np.array(splitRHS)

#plot solution
print('plotting ODE solution...')
plt.clf()
plt.subplot(2, 2, 1)
plt.plot(states.t, states.T)
plt.xlabel('Time (s)')
plt.ylabel('Temperature (K)')
plt.xlim(0., 0.002)
plt.subplot(2, 2, 2)
plt.plot(states.t, states.X[:,gas.species_index('OH')])
plt.xlabel('Time (s)')
plt.ylabel('OH Mole Fraction')
plt.xlim(0., 0.002)
plt.subplot(2, 2, 3)
plt.plot(states.t, states.X[:,gas.species_index('H')])
plt.xlabel('Time (s)')
plt.ylabel('H Mole Fraction')
plt.xlim(0., 0.002)
plt.subplot(2, 2, 4)
plt.plot(states.t, states.X[:,gas.species_index('H2')])
plt.xlabel('Time (s)')
plt.ylabel('H2 Mole Fraction')
plt.xlim(0., 0.002)
plt.tight_layout()
plt.show()

#plot RHS(T)

print('plotting RHS...')
fig, ax = plt.subplots(figsize=(6,4))
ax.plot(states.t, RHS[:,0], color='black', label='rhs pyJac')
ax.plot(states.t, splitRHS[:,0], color='red', linestyle='--',label='S.r')
ax.set_xlabel('time (s)')
ax.set_ylabel('rhs[T]')
ax.set_xlim([0., 0.001])
ax.grid(False)
ax.legend()
plt.show()
#plt.savefig('figures/prediction_combined.png', dpi=500, transparent=False)