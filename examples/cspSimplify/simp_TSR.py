# -*- coding: utf-8 -*-
"""
@author: Riccardo Malpica Galassi, Sapienza University, Roma, Italy
"""
import cantera as ct
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import PyCSP.Functions as csp
import PyCSP.Simplify as simp
from PyCSP.soln2ck import write

#-------CREATE DATASET---------
#create gas from original mechanism file hydrogen.cti
dtl_mech = csp.CanteraCSP('gri30.cti')

#set the gas state
T = 1000
P = ct.one_atm
dtl_mech.TP = T, P
dtl_mech.constP = P
dtl_mech.set_equivalence_ratio(1.0, 'CH4', 'O2:1, N2:3.76')


#integrate ODE with detailed mech
r = ct.IdealGasConstPressureReactor(dtl_mech)
sim = ct.ReactorNet([r])
time = 0.0
states = ct.SolutionArray(dtl_mech, 1, extra={'t': [0.0], 'rhsT': [0.0]})

sim.set_initial_time(0.0)
while sim.time < 1000:
    sim.step()
    states.append(r.thermo.state, t=sim.time, rhsT=dtl_mech.source[-1])
    #print('%10.3e %10.3f %10.3f %14.6e' % (sim.time, r.T, r.thermo.P, r.thermo.u))

dataset = np.concatenate((states.Y,states.T[:,np.newaxis],states.P[:,np.newaxis]),axis=1)
print('Dataset created, start processing...')

#coarsen dataset (to speed up)
dataset = dataset[::64]

#init simplifier
simplifier = simp.CSPsimplify(dtl_mech,dataset)

#simplifier settings
simplifier.TSRtargetset = True
simplifier.TSRtol = 0.5
simplifier.targetset = {'N2'}
simplifier.problemtype = 'constP'
simplifier.scaled = False
simplifier.csprtol = 1.0e-2
simplifier.cspatol = 1.0e-8

simplifier.dataset_info()

#process dataset
simplifier.process_dataset()

print('Done processing')

print('Start pruning...may take a while')
#loop over thresholds
thr = np.arange(0.01, 1, 0.01)

simp_mech = []
prev_species = dtl_mech.species_names
for i in range(len(thr)):
    species, reactions = simplifier.simplify_mechanism(thr[i])
    simp = csp.CanteraCSP(thermo='IdealGas', kinetics='GasKinetics', species=species, reactions=reactions)
    if simp.species_names != prev_species:  #append only if different from previous one
        simp_mech.append(simp)
        write(simp, output_filename='skeletal_N'+str(simp.n_species)+'.inp', skip_thermo=True, skip_transport=True)
        prev_species = simp.species_names

nmech = len(simp_mech)
print('%i mechanisms found' % (nmech))

  
all_states_simp = []
for i in range(nmech):
    # Re-run the ignition problem with the simplified mechanisms
    simp_mech[i].TP = T,P
    simp_mech[i].constP = P
    simp_mech[i].set_equivalence_ratio(1.0, 'CH4', 'O2:1, N2:3.76')
    r = ct.IdealGasConstPressureReactor(simp_mech[i])
    sim = ct.ReactorNet([r])
    states_simp = ct.SolutionArray(simp_mech[i], 1, extra={'t': [0.0], 'rhsT': [0.0]})
    
    sim.set_initial_time(0.0)
    while sim.time < 1000:
        sim.step()
        states_simp.append(r.thermo.state, t=sim.time, rhsT=simp_mech[i].source[-1])
    all_states_simp.append(states_simp)
        


C = [np.random.choice(list(clr.CSS4_COLORS.values())) for i in range(nmech)] 
fig, ax = plt.subplots(figsize=(6,4))
ax.plot(states.t, states.T, lw=3, color='b', linestyle='--', label='Detailed')
for i in range(nmech):
    ax.plot(all_states_simp[i].t, all_states_simp[i].T, lw=2, color = C[i],
         label='M#{0}, Ns={1}, Nr={2}'.format(i, simp_mech[i].n_species, simp_mech[i].n_reactions))
ax.set_xlabel('Time (s)')
ax.set_ylabel('Temperature (K)')
ax.legend(loc='lower right')
ax.set_xlim([0, 2])
plt.savefig('ignition.png', dpi=800, transparent=False)

dtl_idt = states.t[states.rhsT.argmax()]
simp_idt = [all_states_simp[i].t[all_states_simp[i].rhsT.argmax()] for i in range(nmech)]
err_idt = abs((simp_idt-dtl_idt)/dtl_idt)*100

fig, ax = plt.subplots(figsize=(6,4))
ax.plot([simp_mech[i].n_species for i in range(nmech)], err_idt, color='red', marker='.')
ax.set_xlabel('# of species')
ax.set_ylabel('ignition delay time relative Error [%]')
ax.set_yscale('log')
ax.set_ylim([1e-3, 1e2])
plt.savefig('ign_delay_error.png', dpi=800, transparent=False)


