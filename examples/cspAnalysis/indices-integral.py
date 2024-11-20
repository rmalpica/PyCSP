# -*- coding: utf-8 -*-
"""
@author: Riccardo Malpica Galassi, Sapienza University, Roma, Italy
"""
import cantera as ct
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import PyCSP.Functions as csp


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

gas.index_norm='none'  #needed for integral formulation


#integrate ODE
r = ct.IdealGasConstPressureReactor(gas)
sim = ct.ReactorNet([r])
time = 0.0
states = ct.SolutionArray(gas, extra=['t'])


API = []
TPI = []
Ifast = []
Islow = []
classify = []
TSRAPI = []
TSRTPI = []


sim.initial_time = 0.0
while sim.time < 0.001:
    sim.step()
    states.append(r.thermo.state, t=sim.time)
    print('%10.3e %10.3f %10.3f %14.6e' % (sim.time, r.T, r.thermo.P, r.thermo.u))
    api, tpi, ifast, islow, species_type = gas.calc_CSPindices(API=True,Impo=True,species_type=True,TPI=False)
    tsrapi = gas.calc_TSRindices(type='amplitude')
    tsrtpi = gas.calc_TSRindices(type='timescale')
    API.append(api)
    TPI.append(tpi)
    Ifast.append(ifast)
    Islow.append(islow)
    classify.append(species_type)
    TSRAPI.append(tsrapi)
    TSRTPI.append(tsrtpi)


API = np.array(API)
TPI = np.array(TPI)
Ifast = np.array(Ifast)
Islow = np.array(Islow)
classify = np.array(classify)
TSRAPI = np.array(TSRAPI)
TSRTPI = np.array(TSRTPI)


grid = states.t

#Islow
integral = np.reshape([np.trapz(np.abs(Islow[:,i,j]), x=grid) for i in range(gas.nv) for j in range(2*gas.n_reactions)],(gas.nv,2*gas.n_reactions)) 
norm = np.sum(np.abs(integral),axis=1)
IslowIntegralnorm = np.transpose(np.divide(np.transpose(integral), norm, out=np.zeros_like(np.transpose(integral)), where=norm!=0))

#Ifast
integral = np.reshape([np.trapz(np.abs(Ifast[:,i,j]), x=grid) for i in range(gas.nv) for j in range(2*gas.n_reactions)],(gas.nv,2*gas.n_reactions)) 
norm = np.sum(np.abs(integral),axis=1)
IfastIntegralnorm = np.transpose(np.divide(np.transpose(integral), norm, out=np.zeros_like(np.transpose(integral)), where=norm!=0))

#TSRAPI
integral = np.array([np.trapz(np.abs(TSRAPI[:,j]), x=grid) for j in range(2*gas.n_reactions)])
norm = np.sum(np.abs(integral))
TSRAPIIntegralnorm = np.transpose(np.divide(np.transpose(integral), norm, out=np.zeros_like(np.transpose(integral)), where=norm!=0))


spec = -1
thr = 0.05
indextype = IslowIntegralnorm
indexname='Islow integral'
idx = np.argsort(-indextype,axis=1)[spec]
values=indextype[spec][idx][indextype[spec][idx] >= thr]
names=gas.reaction_names()[idx][indextype[spec][idx] >= thr]
y_pos = np.arange(len(values))
labels=[str(names[i]) for i in range(len(names))]
varnames = np.array(gas.species_names+['Temperature'])

plt.rcdefaults()
fig, ax = plt.subplots(figsize = (8, 5))
ax.barh(y_pos, values, align='center')
ax.set_yticks(y_pos)
ax.set_yticklabels(labels)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel(indexname)
ax.set_title(varnames[spec])
plt.subplots_adjust(left=0.4)
plt.show()

plt.savefig('IslowTint.png', dpi=800, transparent=False)
