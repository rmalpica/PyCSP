# -*- coding: utf-8 -*-
"""
@author: Riccardo Malpica Galassi, Sapienza University, Roma, Italy
"""
import cantera as ct
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import sys
import PyCSP.Functions as csp
import PyCSP.Simplify as simp


#-------CREATE DATASET---------
#create gas from original mechanism file hydrogen.cti
dtl_mech = csp.CanteraCSP('gri30.cti')

#set the gas state
T = 1000
P = ct.one_atm
#gas.TPX = T, P, "H2:2.0, O2:1, N2:3.76"
dtl_mech.TP = T, P
dtl_mech.set_equivalence_ratio(1.0, 'CH4', 'O2:1, N2:3.76')


#integrate ODE and dump test data
r = ct.IdealGasConstPressureReactor(dtl_mech)
sim = ct.ReactorNet([r])
time = 0.0
states = ct.SolutionArray(dtl_mech, extra=['t'])

sim.set_initial_time(0.0)
while sim.time < 1000:
    sim.step()
    states.append(r.thermo.state, t=sim.time)
    #print('%10.3e %10.3f %10.3f %14.6e' % (sim.time, r.T, r.thermo.P, r.thermo.u))

dataset = np.concatenate((states.Y,states.T[:,np.newaxis],states.P[:,np.newaxis]),axis=1)
print('Dataset created, start processing...')

dataset = dataset[::64]

#-------IMPORT DATA---------
#read data from file


simplifier = simp.CSPsimplify(dtl_mech,dataset)

simplifier.targetset = {'CH4','O2','N2','HCO','H2O','CO2'}
simplifier.scaled = True
simplifier.dataset_info()

#process dataset
simplifier.process_dataset()

print('Done processing')

print('Start pruning...may take a while')
#loop over thresholds
thr = np.arange(0.01, 0.99, 0.01)
simp_mech = []
prev_species = dtl_mech.species_names
for i in range(len(thr)):
    species, reactions = simplifier.simplify_mechanism(thr[i])
    simp = ct.Solution(thermo='IdealGas', kinetics='GasKinetics', species=species, reactions=reactions)
    if simp.species_names != prev_species: 
        simp_mech.append(simp)
        prev_species = simp.species_names

nmech = len(simp_mech)
print('%i mechanisms found' % (nmech))

all_states_simp = []
for i in range(nmech):
    # Re-run the ignition problem with the simplified mechanisms
    simp_mech[i].TP = T,P
    simp_mech[i].set_equivalence_ratio(1.0, 'CH4', 'O2:1, N2:3.76')
    r = ct.IdealGasConstPressureReactor(simp_mech[i])
    sim = ct.ReactorNet([r])
    states_simp = ct.SolutionArray(simp_mech[i], extra=['t'])
    
    sim.set_initial_time(0.0)
    while sim.time < 1000:
        sim.step()
        states_simp.append(r.thermo.state, t=sim.time)
    all_states_simp.append(states_simp)
        


C = [np.random.choice(list(clr.CSS4_COLORS.values())) for i in range(nmech)] 
plt.plot(states.t, states.T, lw=2, color='b' ,
         label='Detailed')
for i in range(nmech):
    plt.plot(all_states_simp[i].t, all_states_simp[i].T, lw=2, color = C[i],
         label='M#{0}, Ns={1}, Nr={2}'.format(i, simp_mech[i].n_species, simp_mech[i].n_reactions))
plt.xlabel('Time (s)')
plt.ylabel('Temperature (K)')
plt.legend(loc='lower right')
plt.xlim(0, 2)
plt.tight_layout()


