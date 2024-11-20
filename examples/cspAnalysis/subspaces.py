# -*- coding: utf-8 -*-
"""
@author: Riccardo Malpica Galassi, Sapienza University, Roma, Italy
"""
import cantera as ct
import numpy as np
import matplotlib.pyplot as plt
import PyCSP.Functions as csp
import PyCSP.utils as utils

#create gas from original mechanism file hydrogen.cti
gas = csp.CanteraCSP('gri30.yaml')

#set the gas state
T = 1000
P = ct.one_atm
gas.TP = T, P
gas.set_equivalence_ratio(1.0, 'CH4', 'O2:1, N2:3.76')

#push pressure
gas.constP = P

#integrate ODE
r = ct.IdealGasConstPressureReactor(gas)
sim = ct.ReactorNet([r])
sim.rtol=1.0e-12
sim.atol=1.0e-14
time = 0.0
states = ct.SolutionArray(gas, extra=['t'])


evals = []
Revec = []
Levec = []
fvec = []
Tail = []
Head = []

sim.initial_time = 0.0
while sim.time < 10:
    sim.step()
    states.append(r.thermo.state, t=sim.time)
    print('%10.3e %10.3f %10.3f %14.6e' % (sim.time, r.T, r.thermo.P, r.thermo.u))
    lam,R,L,f = gas.get_kernel()
    M, H = gas.calc_subspaces(rtolTail=1.0e-3,atolTail=1.0e-9,rtolHead=1.0e-3,atolHead=1.0e-9)
    evals.append(lam)
    Revec.append(R)
    Levec.append(L)
    fvec.append(f)
    Tail.append(M)
    Head.append(H)

evals = np.array(evals)
Revec = np.array(Revec)
Levec = np.array(Levec)
fvec = np.array(fvec)
Tail = np.array(Tail)
Head = np.array(Head)



#plot solution
print('plotting ODE solution...')
plt.clf()
plt.subplot(2, 2, 1)
plt.plot(states.t, states.T)
plt.xlabel('Time (s)')
plt.ylabel('Temperature (K)')
plt.xlim(1., 1.15)
plt.subplot(2, 2, 2)
plt.plot(states.t, states.X[:,gas.species_index('OH')])
plt.xlabel('Time (s)')
plt.ylabel('OH Mole Fraction')
plt.xlim(1., 1.15)
plt.subplot(2, 2, 3)
plt.plot(states.t, states.X[:,gas.species_index('H')])
plt.xlabel('Time (s)')
plt.ylabel('H Mole Fraction')
plt.xlim(1., 1.15)
plt.subplot(2, 2, 4)
plt.plot(states.t, states.X[:,gas.species_index('CH4')])
plt.xlabel('Time (s)')
plt.ylabel('CH4 Mole Fraction')
plt.xlim(1., 1.15)
plt.tight_layout()
plt.show(block = False)
plt.savefig('traj.png', dpi=500, transparent=False)

#plot eigenvalues and lambda_M+1
evalM = utils.select_eval(evals,Tail)
evalH = utils.select_eval(evals,Head-1)
logevals = np.clip(np.log10(1.0+np.abs(evals)),0,100)*np.sign(evals.real)
logevalM = np.clip(np.log10(1.0+np.abs(evalM.real)),0,100)*np.sign(evalM.real)
logevalH = np.clip(np.log10(1.0+np.abs(evalH.real)),0,100)*np.sign(evalH.real)
print('plotting eigenvalues...')
fig, ax = plt.subplots(figsize=(6,4))
for idx in range(evals.shape[1]):
    #color = next(ax._get_lines.prop_cycler)['color']
    ax.plot(states.t, logevals[:,idx], color='black', marker='.', markersize = 5,linestyle = 'None')
ax.plot(states.t, logevalM, color='orange', marker='.', markersize = 4,linestyle = 'None', label='lam(M+1)')
ax.plot(states.t, logevalH, color='blue', marker='.', markersize = 3,linestyle = 'None', label='lam(H)')
ax.set_xlabel('time (s)')
ax.set_ylabel('evals')
ax.set_ylim([-9, 6])
ax.set_xlim([1, 1.15])
ax.grid(False)
ax.legend()
plt.show(block = False)
plt.savefig('eigenvalues_subspaces.png', dpi=500, transparent=False)

from matplotlib.patches import Rectangle
fig, ax = plt.subplots(figsize=(6,4))
tailplot=ax.plot(states.t, Tail, color='orange', label='M+1')[0] 
headplot=ax.plot(states.t, Head, color='blue', label='H')[0]
ax.fill_between(states.t, Tail, Head, color='deepskyblue', alpha=0.5)
shade=Rectangle((0, 0), 1, 1, fc="deepskyblue", alpha=0.5, label='active modes')
ax.set_xlabel('time (s)')
ax.set_ylabel('modes')
ax.set_xlim([1, 1.15])
ax.set_ylim([0, 54])
plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
plt.legend([tailplot, headplot, shade],['tail','head','active'],fontsize=7)
#plt.show()
fig.savefig('modes.png', dpi=800, transparent=False)

fig, ax = plt.subplots(figsize=(6,4))
ax.plot(states.t, Head - Tail,color='deepskyblue' ) 
ax.set_xlabel('time (s)')
ax.set_ylabel('dim(Active)')
ax.set_xlim([1, 1.15])
ax.set_ylim([0, 54])
plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
#plt.show()
fig.savefig('dim_Active.png', dpi=800, transparent=False)