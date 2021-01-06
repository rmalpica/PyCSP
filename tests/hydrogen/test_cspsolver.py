# -*- coding: utf-8 -*-
"""
@author: Riccardo Malpica Galassi, Sapienza University, Roma, Italy
"""
import cantera as ct
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0,'../..')
import src.cspFunctions as cspF
import src.cspSolver as cspS
import time

#create gas from original mechanism file hydrogen.cti
gas = cspF.CanteraCSP('hydrogen.cti')
#reorder the gas to match pyJac (N2 in last place)
n2_ind = gas.species_index('N2')
specs = gas.species()[:]
gas = cspF.CanteraCSP(thermo='IdealGas', kinetics='GasKinetics',
        species=specs[:n2_ind] + specs[n2_ind + 1:] + [specs[n2_ind]],
        reactions=gas.reactions())

#set the gas state
T = 1000
P = ct.one_atm
#gas.TPX = T, P, "H2:2.0, O2:1, N2:3.76"
gas.TP = T, P
gas.set_equivalence_ratio(1.0, 'H2', 'O2:1, N2:3.76')

y0 = np.hstack((gas.T, gas.Y))
t0 = 0.0

t_end = 1e-2

#integrate ODE with CSP solver
solver = cspS.CSPsolver(gas)
solver.set_integrator(cspRtol=1e-3,cspAtol=1e-9,factor=0.25,jacobiantype='numeric')
solver.set_initial_value(y0,t0)

states = ct.SolutionArray(gas, 1, extra={'t': [0.0], 'M': 0})

starttimecsp = time.time()
while solver.t < t_end:
    solver.integrate()
    states.append(gas.state, t=solver.t, M=solver.M)
    #print('%10.3e %10.3f %10.3e %2i' % (solver.t, solver.y[0], solver.dt, solver.M))
endtimecsp = time.time()

#reset the gas state
T = 1000
P = ct.one_atm
#gas.TPX = T, P, "H2:2.0, O2:1, N2:3.76"
gas.TP = T, P
gas.set_equivalence_ratio(1.0, 'H2', 'O2:1, N2:3.76')

#integrate ODE with CVODE
r = ct.IdealGasConstPressureReactor(gas)
sim = ct.ReactorNet([r])

statesCV = ct.SolutionArray(gas, extra=['t'])
sim.set_initial_time(t0)

starttimeCV = time.time()
while sim.time < t_end:
    sim.step()
    statesCV.append(r.thermo.state, t=sim.time)
endtimeCV = time.time()

elapsedcsp = endtimecsp - starttimecsp
print('CSP solver elapsed time [s]: %10.3e, # of steps taken: %5i' % (elapsedcsp,states.t.shape[0]))

elapsedCV = endtimeCV - starttimeCV
print('CVODE elapsed time [s]:      %10.3e, # of steps taken: %5i' % (elapsedCV,statesCV.t.shape[0]))

#plot solution
print('plotting ODE solution...')
plt.clf()
plt.subplot(2, 2, 1)
plt.plot(states.t, states.T,statesCV.t, statesCV.T)
plt.xlabel('Time (s)')
plt.ylabel('Temperature (K)')
plt.xlim(0., 0.002)
plt.subplot(2, 2, 2)
plt.plot(states.t, states.X[:,gas.species_index('OH')],statesCV.t, statesCV.X[:,gas.species_index('OH')])
plt.xlabel('Time (s)')
plt.ylabel('OH Mole Fraction')
plt.xlim(0., 0.002)
plt.subplot(2, 2, 3)
plt.plot(states.t, states.X[:,gas.species_index('H')],statesCV.t, statesCV.X[:,gas.species_index('H')])
plt.xlabel('Time (s)')
plt.ylabel('H Mole Fraction')
plt.xlim(0., 0.002)
plt.subplot(2, 2, 4)
plt.plot(states.t, states.M)
plt.xlabel('Time (s)')
plt.ylabel('Exhausted modes')
plt.xlim(0., 0.002)
plt.tight_layout()
plt.show()

