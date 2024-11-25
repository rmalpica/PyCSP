# -*- coding: utf-8 -*-
"""
@author: Riccardo Malpica Galassi, Sapienza University, Roma, Italy
"""
import cantera as ct
import numpy as np
import matplotlib.pyplot as plt
import PyCSP.Functions as cspF
import PyCSP.Solver as cspS
import time

#create gas from mechanism file
gas = cspF.CanteraCSP('gri30.yaml')

#user-set initial condition
T = 1000
P = ct.one_atm
fuel = 'CH4'
oxid = 'O2:1, N2:3.76'
eqratio = 1.0

#user-set CSP-solver settings
rtol=1e-3
atol=1e-9
gamma = 0.25
t_end = 2.0

#set the gas state
gas.TP = T, P
gas.set_equivalence_ratio(eqratio, fuel, oxid)   
rho = gas.density
gas.constRho = rho

#initial condition
y0 = np.hstack((gas.Y,gas.T))
t0 = 0.0

#integrate ODE with CSP solver
solver = cspS.CSPsolver(gas)
solver.set_integrator(cspRtol=rtol,cspAtol=atol,factor=gamma,jacobiantype='full')
solver.set_initial_value(y0,t0)

states = ct.SolutionArray(gas, 1, extra={'t': [0.0], 'M': 0, 'dt': [0.0]})

starttimecsp = time.time()
while solver.t < t_end:
    solver.integrate()
    states.append(gas.state, t=solver.t, M=solver.M, dt=solver.dt)
    #print('%10.3e %10.3f %10.3e %10.3e %10.3e %2i' % (solver.t, solver.y[-1], solver.dt, gas.P, gas.density, solver.M))
endtimecsp = time.time()

print('\n CSP-solver profiling:')
solver.profiling()

#integrate ODE with CVODE

#reset the gas state
gas.TP = T, P
gas.set_equivalence_ratio(eqratio, fuel, oxid)

#integrate ODE with CVODE
r = ct.IdealGasReactor(gas)
sim = ct.ReactorNet([r])

statesCV = ct.SolutionArray(gas, extra=['t'])
#sim.set_initial_time(t0)

starttimeCV = time.time()
while sim.time < t_end:
    sim.step()
    statesCV.append(r.thermo.state, t=sim.time)
endtimeCV = time.time()
dt_cvode=statesCV.t[1:]-statesCV.t[0:-1]

elapsedcsp = endtimecsp - starttimecsp
print('CSP solver elapsed time [s]: %10.3e, # of steps taken: %5i, # of kernel evaluations: %5i' % (elapsedcsp,states.t.shape[0],gas.nUpdates))

elapsedCV = endtimeCV - starttimeCV
print('CVODE elapsed time [s]:      %10.3e, # of steps taken: %5i' % (elapsedCV,statesCV.t.shape[0]))

#plot solution
t_start_plot = 1
t_end_plot = 1.15
print('plotting ODE solution...')

# Enlarge the figure
plt.figure(figsize=(12, 8))

# Subplot 1: Temperature
plt.subplot(2, 3, 1)
plt.plot(statesCV.t, statesCV.T, label='CVODE')
plt.plot(states.t, states.T, label='CSP-solver')
plt.xlabel('Time (s)')
plt.ylabel('Temperature (K)')
plt.xlim(t_start_plot, t_end_plot)
plt.legend()

# Subplot 2: OH Mole Fraction
plt.subplot(2, 3, 2)
plt.plot(statesCV.t, statesCV.X[:, gas.species_index('OH')], label='CVODE')
plt.plot(states.t, states.X[:, gas.species_index('OH')], label='CSP-solver')
plt.xlabel('Time (s)')
plt.ylabel('OH Mole Fraction')
plt.xlim(t_start_plot, t_end_plot)
plt.legend()

# Subplot 3: H Mole Fraction
plt.subplot(2, 3, 3)
plt.plot(statesCV.t, statesCV.X[:, gas.species_index('H')], label='CVODE')
plt.plot(states.t, states.X[:, gas.species_index('H')], label='CSP-solver')
plt.xlabel('Time (s)')
plt.ylabel('H Mole Fraction')
plt.xlim(t_start_plot, t_end_plot)
plt.legend()

# Subplot 4: Modes
plt.subplot(2, 3, 4)
plt.plot(states.t, states.M-1, label='Fast-subspace dim', color='red')
plt.xlabel('Time (s)')
plt.ylabel('M')
plt.xlim(t_start_plot, t_end_plot)
plt.legend()

# Subplot 5: Pressure
plt.subplot(2, 3, 5)
plt.plot(statesCV.t, statesCV.P, label='CVODE')
plt.plot(states.t, states.P, label='CSP-solver')
plt.xlabel('Time (s)')
plt.ylabel('Pressure (Pa)')
plt.xlim(t_start_plot, t_end_plot)
plt.legend()

# Subplot 6: dt
plt.subplot(2, 3, 6)
plt.plot(statesCV.t[0:-1], dt_cvode, label='CVODE Time Step')
plt.plot(states.t, states.dt, label='CSP-solver Time Step')
plt.xlabel('Time (s)')
plt.ylabel('dt')
plt.yscale('log')
plt.xlim(t_start_plot, t_end_plot)
plt.legend()

# Adjust layout and show the plot
plt.tight_layout()
plt.show()