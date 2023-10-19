# -*- coding: utf-8 -*-
"""
@author: Riccardo Malpica Galassi, Sapienza University, Roma, Italy
"""
import cantera as ct
import numpy as np
import matplotlib.pyplot as plt
import PyCSP.Functions as cspF
import PyCSP.GScheme as gsc
import time

#create gas from original mechanism file hydrogen.cti
gas = cspF.CanteraCSP('gri30.yaml')

T = 1000
P = ct.one_atm
fu = 'CH4' 
ox = 'O2:1, N2:3.76'
phi = 1.0 

#set the gas state
gas.TP = T, P
gas.set_equivalence_ratio(phi, fu, ox)
gas.constP = P

#set initial condition
y0 = np.hstack((gas.Y,gas.T))
t0 = 0.0

t_end = 2

#integrate ODE with G-Scheme 
solver = gsc.GScheme(gas)
solver.set_integrator(cspRtolTail=1e-3,cspAtolTail=1e-9,factor=0.25,jacobiantype='full')
solver.set_initial_value(y0,t0)

states = ct.SolutionArray(gas, 1, extra={'t': [0.0], 'Tail': 0, 'Head':0, 'dt': [0.0]})

starttimegsc = time.time()
while solver.t < t_end:
    solver.integrate()
    states.append(gas.state, t=solver.t, Tail=solver.T, Head=solver.H, dt=solver.dt)
    #print('%10.3e %10.3f %10.3e %10.3e %10.3e %2i' % (solver.t, solver.y[-1], solver.dt, gas.P, gas.density, solver.M))
endtimegsc = time.time()
nupB = gas.nUpdates

#reset the gas state
gas.TP = T, P
gas.set_equivalence_ratio(phi, fu, ox)

#integrate ODE with G-Scheme with basis reuse
solver = gsc.GScheme(gas)
solver.set_integrator(cspRtolTail=1e-3,cspAtolTail=1e-9,factor=0.25,jacobiantype='full')
solver.set_initial_value(y0,t0)
solver.reuse = True
solver.reuseThr = 2e-1

statesRU = ct.SolutionArray(gas, 1, extra={'t': [0.0], 'Tail': 0, 'Head':0, 'dt': [0.0], 'bs':0, 'norm': [0.0]})

starttimegscreu = time.time()
while solver.t < t_end:
    solver.integrate()
    statesRU.append(gas.state, t=solver.t, Tail=solver.T, Head=solver.H, dt=solver.dt, bs=solver.basisStatus, norm=solver.normJac)
    #print('%10.3e %10.3f %10.3e %10.3e' % (solver.t, solver.y[-1], solver.dt, solver.normJac))
endtimegscreu = time.time()
nupBRU = solver.updateBasis



#reset the gas state
gas.TP = T, P
gas.set_equivalence_ratio(phi, fu, ox)

#integrate ODE with CVODE
r = ct.IdealGasConstPressureReactor(gas)
sim = ct.ReactorNet([r])

statesCV = ct.SolutionArray(gas, extra=['t'])
#sim.set_initial_time(t0)

starttimeCV = time.time()
while sim.time < t_end:
    sim.step()
    statesCV.append(r.thermo.state, t=sim.time)
endtimeCV = time.time()

elapsedgsc = endtimegsc - starttimegsc
print('G-Scheme elapsed time [s]: %10.3e, # of steps taken: %5i, # of kernel evaluations: %5i' % (elapsedgsc,states.t.shape[0],nupB))

elapsedgscreu = endtimegscreu - starttimegscreu
print('G-Scheme w/ reuse elapsed time [s]: %10.3e, # of steps taken: %5i, # of kernel evaluations: %5i' % (elapsedgscreu,statesRU.t.shape[0],nupBRU))

elapsedCV = endtimeCV - starttimeCV
print('CVODE elapsed time [s]:      %10.3e, # of steps taken: %5i' % (elapsedCV,statesCV.t.shape[0]))

#plot solution
t_start_plot = 1
t_end_plot = 1.15
print('plotting ODE solution...')
plt.clf()
plt.subplot(2, 3, 1)
plt.plot(statesCV.t, statesCV.T, color = 'blue', label='cvode')
plt.plot(states.t, states.T, color = 'orange', label='gsc')
plt.plot(statesRU.t, statesRU.T, color = 'green', label='gsc-ru')
plt.xlabel('Time (s)')
plt.ylabel('Temperature (K)')
plt.legend(loc="upper left")
plt.xlim(t_start_plot, t_end_plot)

plt.subplot(2, 3, 2)
plt.plot(statesCV.t, statesCV.X[:,gas.species_index('OH')], color = 'blue', label='cvode')
plt.plot(states.t, states.X[:,gas.species_index('OH')], color = 'orange', label='gsc')
plt.plot(statesRU.t, statesRU.X[:,gas.species_index('OH')], color = 'green', label='gsc-ru')
plt.xlabel('Time (s)')
plt.ylabel('OH Mole Fraction')
plt.legend(loc="upper left")
plt.xlim(t_start_plot, t_end_plot)

plt.subplot(2, 3, 3)
plt.plot(statesRU.t, statesRU.norm)
calc = statesRU.norm[statesRU.bs == 1] 
reuse = statesRU.norm[statesRU.bs == 0]  
plt.plot(statesRU.t[statesRU.bs == 1], calc, 'ko', markersize=2, label='compute')
plt.plot(statesRU.t[statesRU.bs == 0], reuse, '+r', markersize=2, label='reuse')
plt.xlabel('Time (s)')
plt.ylabel('norm Jac')
plt.legend(loc="upper left")
plt.xlim(t_start_plot, t_end_plot)

plt.subplot(2, 3, 4)
plt.plot(states.t, states.Tail, statesRU.t, statesRU.Tail, states.t, states.Head, statesRU.t, statesRU.Head)
plt.fill_between(states.t, states.Tail, states.Head, color='grey', alpha=0.2)
plt.fill_between(statesRU.t, statesRU.Tail, statesRU.Head, color='orange', alpha=0.2)
plt.xlabel('Time (s)')
plt.ylabel('Modes')
plt.xlim(t_start_plot, t_end_plot)

plt.subplot(2, 3, 5)
plt.plot(states.t, states.Head - states.Tail, color='orange', label = 'gsc')
plt.plot(statesRU.t, statesRU.Head - statesRU.Tail, color='green', label='gsc-ru')
plt.xlabel('Time (s)')
plt.ylabel('active modes')
plt.legend(loc="upper left")
plt.xlim(t_start_plot, t_end_plot)

plt.subplot(2, 3, 6)
plt.plot(states.t, states.dt,color='orange', label = 'gsc')
plt.plot(statesRU.t, statesRU.dt, color='green', label='gsc-ru')
plt.xlabel('Time (s)')
plt.ylabel('dt')
plt.yscale('log')
plt.legend(loc="upper left")
plt.xlim(t_start_plot, t_end_plot)

plt.tight_layout()
plt.show()
