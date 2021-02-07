# -*- coding: utf-8 -*-
"""
@author: Riccardo Malpica Galassi, Sapienza University, Roma, Italy
"""
import cantera as ct
import numpy as np
import matplotlib.pyplot as plt
import sys
import PyCSP.Functions as csp
import PyCSP.Simplify as simp
import PyCSP.utils as utils

#-------CREATE DATASET---------
#create gas from original mechanism file hydrogen.cti
gas = csp.CanteraCSP('gri30.cti')

#set the gas state
T = 1000
P = ct.one_atm
#gas.TPX = T, P, "H2:2.0, O2:1, N2:3.76"
gas.TP = T, P
gas.set_equivalence_ratio(1.0, 'CH4', 'O2:1, N2:3.76')


#integrate ODE and dump test data
r = ct.IdealGasConstPressureReactor(gas)
sim = ct.ReactorNet([r])
time = 0.0
states = ct.SolutionArray(gas, extra=['t'])

sim.set_initial_time(0.0)
while sim.time < 1000:
    sim.step()
    states.append(r.thermo.state, t=sim.time)
    #print('%10.3e %10.3f %10.3f %14.6e' % (sim.time, r.T, r.thermo.P, r.thermo.u))

dataset = np.concatenate((states.Y,states.T[:,np.newaxis],states.P[:,np.newaxis]),axis=1)
print('Dataset created, starting processing...')

dataset = dataset[::64]

#-------IMPORT DATA---------
#read data from file


simplifier = simp.CSPsimplify(gas,dataset)

simplifier.targetset = {'CH4','O2','HCO','H2O','CO2'}
simplifier.scaled = True

simplifier.dataset_info()

simplifier.process_dataset()

sys.exit()

tol = 0.1
species, reactions = simplifier.simplify_mechanism(tol)

sys.exit()



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
evalM = utils.select_eval(evals,M)
logevals = np.clip(np.log10(1.0+np.abs(evals)),0,100)*np.sign(evals.real)
logevalM = np.clip(np.log10(1.0+np.abs(evalM)),0,100)*np.sign(evalM.real)
print('plotting eigenvalues...')
fig, ax = plt.subplots(figsize=(6,4))
for idx in range(evals.shape[1]):
    #color = next(ax._get_lines.prop_cycler)['color']
    ax.plot(states.t, logevals[:,idx], color='black', marker='.', markersize = 5,linestyle = 'None')
ax.plot(states.t, logevalM, color='orange', marker='.', markersize = 4,linestyle = 'None', label='lam(M+1) rtol e-2; atol e-8')
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
ax.plot(M, color='orange', label='rtol e-2; atol e-8')
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