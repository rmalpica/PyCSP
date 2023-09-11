# -*- coding: utf-8 -*-
"""
@author: Riccardo Malpica Galassi, Sapienza University, Roma, Italy
"""
import cantera as ct
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import PyCSP.Functions as csp
import PyCSP.ThermoKinetics as ctt


#create gas from original mechanism file hydrogen.cti
gas = csp.CanteraCSP('hydrogen.yaml')

#set the gas state
T = 1000
P = ct.one_atm
#gas.TPX = T, P, "H2:2.0, O2:1, N2:3.76"
gas.TP = T, P
gas.set_equivalence_ratio(1.0, 'H2', 'O2:1, N2:3.76')

#push pressure
gas.constP = P

#edit CSP parameters
gas.jacobiantype='full'
gas.rtol=1.0e-3
gas.atol=1.0e-10

#very important for SIM-constrained Jac
gas.classify_traces=False

#integrate ODE
r = ct.IdealGasConstPressureReactor(gas)
sim = ct.ReactorNet([r])
time = 0.0
states = ct.SolutionArray(gas, extra=['t'])


evals = []
evalsSIM = []


sim.set_initial_time(0.0)
while sim.time < 0.001:
    sim.step()
    states.append(r.thermo.state, t=sim.time)
    print('%10.3e %10.3f %10.3f %14.6e' % (sim.time, r.T, r.thermo.P, r.thermo.u))
    lam,R,L,f = gas.get_kernel()
    api, tpi, ifast, islow, species_type = gas.calc_CSPindices(API=False,Impo=False,species_type=True,TPI=False)
    major = [i for i, x in enumerate(species_type) if x == "slow"]
    
    jacSIM = gas.jacSIM(gas.jac,major,L)
    lamSIM = np.zeros(len(lam))
    lSIM,RSIM,LSIM = csp.eigsys(jacSIM)
    lamSIM[:len(lSIM)]=lSIM
    
    evals.append(lam)
    evalsSIM.append(lamSIM)



evals = np.array(evals)
evalsSIM = np.array(evalsSIM)

#plot eigenvalues of constrained model
#evalM = utils.select_eval(evalsFull,MFull)
logevals = np.clip(np.log10(1.0+np.abs(evals)),0,100)*np.sign(evals.real)
#logevalM = np.clip(np.log10(1.0+np.abs(evalM)),0,100)*np.sign(evalM.real)
logevalsC = np.clip(np.log10(1.0+np.abs(evalsSIM)),0,100)*np.sign(evalsSIM.real)
print('plotting eigenvalues of full and constrained model...')
fig, ax = plt.subplots(figsize=(6,4))
for idx in range(evals.shape[1]):
    ax.plot(states.t, logevals[:,idx], color='black', marker='.', markersize = 5,linestyle = 'None')
for idx in range(evalsSIM.shape[1]):
    ax.plot(states.t, logevalsC[:,idx], color='red', marker='.', markersize = 2,linestyle = 'None')
#ax.plot(time, logevalM, color='orange', marker='.', markersize = 3,linestyle = 'None', label='lam(M+1) rtol e-2; atol e-8')
ax.set_xlabel('time (s)')
ax.set_ylabel('evals')
ax.set_ylim([-9, 6])
ax.set_xlim([0., max(states.t)])
ax.grid(False)
ax.legend()
plt.savefig('jacobianSIM.png',dpi=800)

