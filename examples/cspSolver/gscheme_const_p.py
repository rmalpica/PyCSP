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

#set the gas state
T = 1000
P = ct.one_atm
#gas.TPX = T, P, "H2:2.0, O2:1, N2:3.76"
gas.TP = T, P
gas.set_equivalence_ratio(1.0, 'CH4', 'O2:1, N2:3.76')
gas.constP = P

# GScheme settings
rtolTail = 1e-3 
atolTail = 1e-9
rtolHead = 1e-4
atolHead = 1e-10
gamma = 0.25
reuse = False
reusetol = 2e-1
# Initial conditions
t_end = 2.0

#set initial condition
y0 = np.hstack((gas.Y,gas.T))
t0 = 0.0

#integrate ODE with G-Scheme
solver = gsc.GScheme(gas)
solver.set_integrator(cspRtolTail=rtolTail,cspAtolTail=atolTail,cspRtolHead=rtolHead,cspAtolHead=atolHead,factor=gamma,jacobiantype='full')
solver.set_initial_value(y0,t0)

states = ct.SolutionArray(gas, 1, extra={'t': [0.0], 'Tail': 0, 'Head':0, 'dt': [0.0]})

while solver.t < t_end:
    solver.integrate()
    states.append(gas.state, t=solver.t, Tail=solver.T, Head=solver.H, dt=solver.dt)
    #print('%10.3e %10.3f %10.3e %10.3e %10.3e %2i' % (solver.t, solver.y[-1], solver.dt, gas.P, gas.density, solver.M))

print('\n G-Scheme profiling:')
solver.profiling()

#reset the gas state
T = 1000
P = ct.one_atm
#gas.TPX = T, P, "H2:2.0, O2:1, N2:3.76"
gas.TP = T, P
gas.set_equivalence_ratio(1.0, 'CH4', 'O2:1, N2:3.76')

#integrate ODE with CVODE
r = ct.IdealGasConstPressureReactor(gas)
sim = ct.ReactorNet([r])

statesCV = ct.SolutionArray(gas, extra=['t'])
#sim.set_initial_time(t0)

while sim.time < t_end:
    sim.step()
    statesCV.append(r.thermo.state, t=sim.time)
dt_cvode=statesCV.t[1:]-statesCV.t[0:-1]

#plot solution
t_start_plot = 1
t_end_plot = 1.15
print('plotting ODE solution...')

# Enlarge the figure
plt.figure(figsize=(12, 8))

# Subplot 1: Temperature
plt.subplot(2, 3, 1)
plt.plot(statesCV.t, statesCV.T, label='CVODE')
plt.plot(states.t, states.T, label='G-Scheme')
plt.xlabel('Time (s)')
plt.ylabel('Temperature (K)')
plt.xlim(t_start_plot, t_end_plot)
plt.legend()

# Subplot 2: OH Mole Fraction
plt.subplot(2, 3, 2)
plt.plot(statesCV.t, statesCV.X[:, gas.species_index('OH')], label='CVODE')
plt.plot(states.t, states.X[:, gas.species_index('OH')], label='G-Scheme')
plt.xlabel('Time (s)')
plt.ylabel('OH Mole Fraction')
plt.xlim(t_start_plot, t_end_plot)
plt.legend()

# Subplot 3: H Mole Fraction
plt.subplot(2, 3, 3)
plt.plot(statesCV.t, statesCV.X[:, gas.species_index('H')], label='CVODE')
plt.plot(states.t, states.X[:, gas.species_index('H')], label='G-Scheme')
plt.xlabel('Time (s)')
plt.ylabel('H Mole Fraction')
plt.xlim(t_start_plot, t_end_plot)
plt.legend()

# Subplot 4: Modes
plt.subplot(2, 3, 4)
plt.plot(states.t, states.Tail, label='Tail', color='red')
plt.plot(states.t, states.Head, label='Head', color='green')
plt.fill_between(states.t, states.Tail, states.Head, color='grey', alpha=0.2, label='Active Modes')
plt.xlabel('Time (s)')
plt.ylabel('Modes')
plt.xlim(t_start_plot, t_end_plot)
plt.legend()

# Subplot 5: Active Modes
plt.subplot(2, 3, 5)
plt.plot(states.t, states.Head - states.Tail, color='orange', label='Active Modes')
plt.xlabel('Time (s)')
plt.ylabel('Active Modes')
plt.xlim(t_start_plot, t_end_plot)
plt.legend()

# Subplot 6: dt
plt.subplot(2, 3, 6)
plt.plot(statesCV.t[0:-1], dt_cvode, label='CVODE Time Step')
plt.plot(states.t, states.dt, label='G-Scheme Time Step')
plt.xlabel('Time (s)')
plt.ylabel('dt')
plt.yscale('log')
plt.xlim(t_start_plot, t_end_plot)
plt.legend()

# Adjust layout and show the plot
plt.tight_layout()
plt.show()