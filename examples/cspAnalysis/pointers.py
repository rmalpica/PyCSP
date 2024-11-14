# -*- coding: utf-8 -*-
"""
@author: Riccardo Malpica Galassi, Sapienza University, Roma, Italy
"""
import cantera as ct
import numpy as np
import matplotlib.pyplot as plt
import PyCSP.Functions as csp
import PyCSP.utils as utils

species_to_be_pointed = 'NO'
threshold_on_pointer_magnitude = 0.05  #picks only modes whose pointer is larger than this value

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
pointed_modes = []
pointer_values = []

sim.set_initial_time(0.0)
while sim.time < 10:
    sim.step()
    states.append(r.thermo.state, t=sim.time)
    print('%10.3e %10.3f %10.3f %14.6e' % (sim.time, r.T, r.thermo.P, r.thermo.u))
    lam,R,L,f = gas.get_kernel()
    pointers = csp.CSP_pointers(R,L)
    pointed = [i for i, pointer in enumerate(pointers) if np.abs(pointer[gas.species_index(species_to_be_pointed)]) > threshold_on_pointer_magnitude ]
    ptrs_values = pointers[pointed,gas.species_index(species_to_be_pointed)]
    evals.append(lam)
    pointed_modes.append(pointed)
    pointer_values.append(ptrs_values)

evals = np.array(evals)
pointed_modes = np.array(pointed_modes, dtype=object)
pointer_values = np.array(pointer_values, dtype=object)
pointed_evals = [evals[i, idx_list] for i, idx_list in enumerate(pointed_modes)]

# Apply the log transformation to each array in pointed_evals
log_pointed_evals = [np.clip(np.log10(1.0 + np.abs(values.real)), 0, 100) * np.sign(values.real) for values in pointed_evals]

# Convert pointer_values to ensure it is a flat, numeric array
pointer_values = np.array([val[0] if isinstance(val, (list, np.ndarray)) else val for val in pointer_values], dtype=float)

# Flattening 
flat_log_pointed_evals = np.concatenate(log_pointed_evals)
flat_time_values = np.repeat(states.t, [len(values) for values in log_pointed_evals])
flat_color_values = np.repeat(pointer_values, [len(values) for values in log_pointed_evals])

# Plot with color shading based on `flat_color_values`
plt.figure(figsize=(10, 6))
sc = plt.scatter(flat_time_values, flat_log_pointed_evals, c=np.abs(flat_color_values), cmap="viridis", alpha=0.6)

plt.xlim(1, 1.2)
plt.xlabel("Time")
plt.ylabel("Log Evals")
plt.title("Evals pointing %s" %species_to_be_pointed)

# Add a colorbar
plt.colorbar(sc, label="Pointer Magnitude")
plt.show()