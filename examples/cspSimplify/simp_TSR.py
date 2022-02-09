# -*- coding: utf-8 -*-
"""
@author: Riccardo Malpica Galassi, Sapienza University, Roma, Italy
"""
import os
import cantera as ct
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import PyCSP.Functions as csp
import PyCSP.Simplify as simp
import PyCSP.utils as utils
from PyCSP.soln2cti import write


FuComp = 'CH4'
OxComp = 'O2:1, N2:3.76'

#-------CREATE DATASET---------
#create gas from original mechanism file hydrogen.cti
dtl_mech = csp.CanteraCSP('gri30.cti')

T_list = [1000]
phi_list = [1.0]
P_list = [ct.one_atm]


states_db = []
for eqratio in phi_list:
    for temperature in T_list:
        for pressure in P_list:
            # Re-run the ignition problem with the detailed mechanisms
            states = utils.integrate_batch_constP(dtl_mech,temperature,pressure,eqratio,FuComp,OxComp,1000)            
            states_db.append(states)

allT = np.concatenate([states_db[i].T for i in range(len(states_db))]) 
allY = np.concatenate([states_db[i].Y for i in range(len(states_db))]) 
allP = np.concatenate([states_db[i].P for i in range(len(states_db))])          

dataset = np.concatenate((allY,allT[:,np.newaxis],allP[:,np.newaxis]),axis=1)
print('Dataset created, start processing...')

#coarsen dataset (to speed up)
dataset = dataset[::64]
print('Number of states in dataset: %i' %len(dataset))

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
prev_species = sorted(dtl_mech.species_names)
for i in range(len(thr)):
    species, reactions = simplifier.simplify_mechanism(thr[i])
    simp = csp.CanteraCSP(thermo='IdealGas', kinetics='GasKinetics', species=species, reactions=reactions)
    if sorted(simp.species_names) != prev_species:  #append only if different from previous one
        simp_mech.append(simp)
        #write(simp, output_filename='skeletal_N'+str(simp.n_species)+'_R'+str(simp.n_reactions)+'.inp', skip_thermo=True, skip_transport=True)
        write(simp, output_filename='skeletal_N'+str(simp.n_species)+'_R'+str(simp.n_reactions)+'.inp')
        prev_species = sorted(simp.species_names)

nmech = len(simp_mech)
print('%i mechanisms found' % (nmech))


print('Assessing skeletal mechanisms performance over a range of conditions...')

#import DTL
dtl_mech = csp.CanteraCSP('gri30.cti')

#import Skeletal(s)
read_mech = sorted([f for f in os.listdir('./') if f.startswith('skeletal')])
simp_mech = []
for i in range(len(read_mech)):
    simp_mech.append(csp.CanteraCSP(read_mech[i]))
print('%i mechanisms found' % len(simp_mech))

T_list = np.arange(1000,1800,100)
phi_list = [0.5, 0.7, 1.0, 1.2, 1.5]

nTemp = len(T_list)
nPhi = len(phi_list)

all_states_dtl = []
all_idt_dtl = []
all_equi_dtl = []
for eqratio in phi_list:
    for temperature in T_list:
            # Re-run the ignition problem with the detailed mechanisms
            states = utils.integrate_batch_constP(dtl_mech,temperature,P,eqratio,FuComp,OxComp,1000)            
            idt = states.t[states.rhsT.argmax()]
            equi = states.T[-1]
            all_states_dtl.append(states)
            all_idt_dtl.append(idt)
            all_equi_dtl.append(equi)

  
all_states_simp = []
all_idt_simp = []
all_equi_simp = []
for i in range(nmech):   
    for eqratio in phi_list:    
        for temperature in T_list:
            # Re-run the ignition problem with the simplified mechanisms
            states = utils.integrate_batch_constP(simp_mech[i],temperature,P,eqratio,FuComp,OxComp,1000)
            idt = states.t[states.rhsT.argmax()]
            equi = states.T[-1]
            all_states_simp.append(states)
            all_idt_simp.append(idt)
            all_equi_simp.append(equi)
        

#compute ignition delay errors
all_idt_err = []
for i in range(nmech):
    for j in range(nPhi):
        for k in range(nTemp):            
            err_idt = abs((all_idt_simp[i*nPhi*nTemp +j*nTemp +k]-all_idt_dtl[j*nTemp+k])/all_idt_dtl[j*nTemp+k])*100
            all_idt_err.append(err_idt)

all_idt_err = np.array(all_idt_err).reshape(nmech,nPhi,nTemp)


#compute equilibrium errors
all_equi_err = []
for i in range(nmech):
    for j in range(nPhi):
        for k in range(nTemp):            
            err_equi = abs((all_equi_simp[i*nPhi*nTemp +j*nTemp +k]-all_equi_dtl[j*nTemp+k])/all_equi_dtl[j*nTemp+k])*100
            all_equi_err.append(err_equi)

all_equi_err = np.array(all_equi_err).reshape(nmech,nPhi,nTemp)

plt.ioff()
for i in range(nmech):
    fig, ax = plt.subplots(figsize=(6,4))
    for j in range(nPhi):
        label='phi = '+str(phi_list[j])
        ax.plot(1000/T_list, all_idt_err[i,j], marker='.', label=label)
    title='# of species: '+str(simp_mech[i].n_species)
    ax.set_title(title)
    ax.set_xlabel('1000/T [1/K]')
    ax.set_ylabel('ignition delay time relative Error [%]')
    ax.set_yscale('log')
    ax.set_ylim([1e-3, 1e2])
    ax.legend()
    plt.savefig('ign_delay_error_sk'+str(simp_mech[i].n_species)+'_'+str(simp_mech[i].n_reactions)+'.png', dpi=800, transparent=False)
    
        
for i in range(nmech):
    fig, ax = plt.subplots(figsize=(6,4))
    for j in range(nPhi):
        label='phi = '+str(phi_list[j])
        ax.plot(1000/T_list, all_equi_err[i,j], marker='.', label=label)
    title='# of species: '+str(simp_mech[i].n_species)
    ax.set_title(title)
    ax.set_xlabel('1000/T [1/K]')
    ax.set_ylabel('Equilibrium Temp relative Error [%]')
    ax.set_yscale('log')
    ax.set_ylim([1e-6, 1e2])
    ax.legend()
    plt.savefig('equilibrium_error_sk'+str(simp_mech[i].n_species)+'_'+str(simp_mech[i].n_reactions)+'.png', dpi=800, transparent=False)
    


# =============================================================================
# C = [np.random.choice(list(clr.CSS4_COLORS.values())) for i in range(nmech)] 
# fig, ax = plt.subplots(figsize=(6,4))
# ax.plot(states.t, states.T, lw=2, color='b', label='Detailed')
# for i in range(nmech):
#     ax.plot(all_states_simp[i].t, all_states_simp[i].T, lw=2, color = C[i],
#          label='M#{0}, Ns={1}, Nr={2}'.format(i, simp_mech[i].n_species, simp_mech[i].n_reactions))
# ax.set_xlabel('Time (s)')
# ax.set_ylabel('Temperature (K)')
# ax.legend(loc='lower right')
# ax.set_xlim([0, 2])
# plt.savefig('ignition.png', dpi=800, transparent=False)
# =============================================================================