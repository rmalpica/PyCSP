# -*- coding: utf-8 -*-
"""
@author: Riccardo Malpica Galassi, Sapienza University, Roma, Italy
"""
import cantera as ct
import numpy as np
import matplotlib.pyplot as plt
import sys
import PyCSP.Functions as csp
import PyCSP.utils as utils

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


#integrate ODE
r = ct.IdealGasConstPressureReactor(gas)
sim = ct.ReactorNet([r])
time = 0.0
states = ct.SolutionArray(gas, extra=['t'])


evals = []
Revec = []
Levec = []
fvec = []
Mr2a8 = []
Mr3a9 = []
Mr4a1 = []

sim.set_initial_time(0.0)
while sim.time < 0.001:
    sim.step()
    states.append(r.thermo.state, t=sim.time)
    print('%10.3e %10.3f %10.3f %14.6e' % (sim.time, r.T, r.thermo.P, r.thermo.u))
    lam,R,L,f = gas.get_kernel(jacobiantype='full')
    NofDM28 = gas.calc_exhausted_modes(rtol=1.0e-2,atol=1.0e-8)
    NofDM39 = gas.calc_exhausted_modes(rtol=1.0e-3,atol=1.0e-9)
    NofDM41 = gas.calc_exhausted_modes(rtol=1.0e-4,atol=1.0e-10)
    evals.append(lam)
    Revec.append(R)
    Levec.append(L)
    fvec.append(f)
    Mr2a8.append(NofDM28)
    Mr3a9.append(NofDM39)
    Mr4a1.append(NofDM41)

evals = np.array(evals)
Revec = np.array(Revec)
Levec = np.array(Levec)
fvec = np.array(fvec)
Mr2a8 = np.array(Mr2a8)
Mr3a9 = np.array(Mr3a9)
Mr4a1 = np.array(Mr4a1)


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

#plot eigenvalues and lambda_M+1
evalMr2a8 = utils.select_eval(evals,Mr2a8)
evalMr3a9 = utils.select_eval(evals,Mr3a9)
evalMr4a1 = utils.select_eval(evals,Mr4a1)
logevals = np.clip(np.log10(1.0+np.abs(evals)),0,100)*np.sign(evals.real)
logevalMr2a8 = np.clip(np.log10(1.0+np.abs(evalMr2a8.real)),0,100)*np.sign(evalMr2a8.real)
logevalMr3a9 = np.clip(np.log10(1.0+np.abs(evalMr3a9.real)),0,100)*np.sign(evalMr3a9.real)
logevalMr4a1 = np.clip(np.log10(1.0+np.abs(evalMr4a1.real)),0,100)*np.sign(evalMr4a1.real)
print('plotting eigenvalues...')
fig, ax = plt.subplots(figsize=(6,4))
for idx in range(evals.shape[1]):
    #color = next(ax._get_lines.prop_cycler)['color']
    ax.plot(states.t, logevals[:,idx], color='black', marker='.', markersize = 5,linestyle = 'None')
ax.plot(states.t, logevalMr2a8, color='orange', marker='.', markersize = 4,linestyle = 'None', label='lam(M+1) rtol e-2; atol e-8')
ax.plot(states.t, logevalMr3a9, color='blue', marker='.', markersize = 3,linestyle = 'None', label='lam(M+1) rtol e-3; atol e-9')
ax.plot(states.t, logevalMr4a1, color='red', marker='.', markersize = 2,linestyle = 'None', label='lam(M+1)  rtol e-4; atol e-10')
ax.set_xlabel('time (s)')
ax.set_ylabel('evals')
ax.set_ylim([-9, 6])
ax.set_xlim([0., 0.001])
ax.grid(False)
ax.legend()
plt.show()
plt.savefig('eigenvalues.png', dpi=500, transparent=False)


#plot exhausted modes
print('plotting exhausted modes...')
fig, ax = plt.subplots(figsize=(6,4))
#ax.plot(states.t, M, color='black')
ax.plot(Mr2a8, color='orange', label='rtol e-2; atol e-8')
ax.plot(Mr3a9, color='red', label='rtol e-3; atol e-9')
ax.plot(Mr4a1, color='blue', label='rtol e-4; atol e-10')
#ax.set_xlabel('time (s)')
ax.set_xlabel('# timestep')
ax.set_ylabel('M')
ax.set_ylim([0,10])
#ax.set_xlim([0., 0.001])
ax.grid(False)
ax.legend()
plt.show()

#plot amplitudes
print('plotting mode amplitudes...')
fig, ax = plt.subplots(figsize=(6,4))
for idx in range(fvec.shape[1]):
    #color = next(ax._get_lines.prop_cycler)['color']
    ax.plot(states.t, np.log10(1e-10+fvec[:,idx]), label='Mode %d' %(idx+1), marker='.', markersize = 2,linestyle = 'None')
ax.set_xlabel('time (s)')
ax.set_ylabel('Amplitude')
ax.set_ylim([-8, 10])
ax.set_xlim([0., 0.001])
ax.grid(False)
ax.legend()
plt.show()
#plt.savefig('figures/prediction_combined.png', dpi=500, transparent=False)