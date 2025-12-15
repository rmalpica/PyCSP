# -*- coding: utf-8 -*-
"""
@author: Riccardo Malpica Galassi, Sapienza University, Roma, Italy
"""
import numpy as np
import PyCSP.Functions as csp
import PyCSP.utils as utils
import cantera as ct



#create gas from original mechanism file hydrogen.yaml
gas = csp.CanteraCSP('hydrogen.yaml')

##### GENERATE DATA #####

#set the gas state
T = 1000
P = ct.one_atm
#gas.TPX = T, P, "H2:2.0, O2:1, N2:3.76"
gas.TP = T, P
gas.set_equivalence_ratio(1.0, 'H2', 'O2:1, N2:3.76')


#integrate ODE
r = ct.IdealGasConstPressureReactor(gas)
sim = ct.ReactorNet([r])
sim.rtol=1.0e-12
sim.atol=1.0e-14
time = 0.0
states = ct.SolutionArray(gas, extra=['t'])

sim.initial_time = 0.0
while sim.time < 10:
    sim.step()
    states.append(r.thermo.state, t=sim.time)
    print('%10.3e %10.3f %10.3f %14.6e' % (sim.time, r.T, r.thermo.P, r.thermo.u))

print(f'Done! Number of states {len(states.t)}')

##### RUN CSP ANALYSIS #####
print("Now Running CSP")

#Initialize database
db = utils.TaggedDatabase()

for step in range(0,len(states.t)):
    print(f'Analyzing state #{step}')
    Press = states.P[step]
    Y = states.Y[step]
    T = states.T[step]
    stateYT = np.append(Y,T)
    #push pressure
    gas.constP = P
    #set jacobiantype
    gas.jacobiantype = 'full'
    #set CSP tolerances
    gas.rtol = 1.0e-2
    gas.atol = 1.0e-8
    #push state
    gas.set_stateYT(stateYT)
    
    # Get kernel data
    lam, R, L, f = gas.get_kernel()
    # Detect frozen modes
    frozen = csp.detectFrozenModes(gas.nv-gas.n_elements, stateYT, R, L, gas.generalized_Stoich_matrix, gas.R_vector, 1e-4, 1e-12)
    # Detect exhusted modes
    M = gas.calc_exhausted_modes()
    # Get indexes
    api, tpi, ifast, islow, species_type = gas.calc_CSPindices(API=True,Impo=True,species_type=True,TPI=False)
        
    # Store everything in the database using keyword arguments
    db.store(
        time=states.t[step],
        state=stateYT,
        evals=lam,
        Revec=R,
        Levec=L,
        fvec=f,
        M=M,
        frozen=frozen,
        tsr = gas.calc_TSR(),
        API = api,
        #TPI = tpi,  #TPI can be expensive if number of species is large
        Ifast = ifast,
        Islow = islow,
        species_type = species_type,   
        Pointers = csp.CSP_pointers(R,L),
        tsrApi = gas.calc_TSRindices(type='amplitude'),
        tsrTpi = gas.calc_TSRindices(type='timescale')
    )

# Convert all stored lists to numpy arrays automatically
db.to_arrays()

print('DONE !')


print('Generating plots...')
utils.save_CSP_plots(
    db, 
    gas, 
    output_folder='CSP_Plots', 
    threshold=0.2, # Show indexes with value > threshold (percentage)
    species_threshold=1e-4,  # Show species with mass fraction > 1e-4
    xlim = (0, 0.001), # Can be used to restrict plotting window
    xlabel='time [s]' 
)
print('Plots saved to CSP_Plots/')
