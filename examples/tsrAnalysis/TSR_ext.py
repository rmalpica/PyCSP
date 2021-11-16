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
import PyCSP.utils as utils

#create gas from original mechanism file hydrogen.cti
gas = csp.CanteraCSP('chaos12.cti')

#-------IMPORT DATA---------
#read data from file
state = np.loadtxt('flamelet_state.dat')
time = state[:,0]
counter = state[:,1]
zeta = state[:,2]
Pressure = state[:,3]
Temp = state[:,4]
Y =  state[:,5:]


rhsD = np.loadtxt('flamelet_rhsDiff.dat')
diffTemp = rhsD[:,3]
diffY =  rhsD[:,4:]

#set jacobiantype
gas.jacobiantype = 'full'

evals = []
M = []
tsr = []
Mext = []
tsrext = []


print('Analyzing %i data points, may take a while...' %len(state))

for step in range(time.shape[0]):
#for step in range(begin,end):
    gas.constP = Pressure[step]
    stateYT = np.append(Y[step],Temp[step])
    rhsdiffYT = np.append(diffY[step],diffTemp[step])
    gas.set_stateYT(stateYT)
    lam,R,L,f = gas.get_kernel()
    omegatau, NofDM = gas.calc_TSR(getM=True)
    omegatauext, NofDMext = gas.calc_extended_TSR(getMext=True,diff=rhsdiffYT)

    evals.append(lam)
    M.append(NofDM)
    tsr.append(omegatau)
    Mext.append(NofDMext)
    tsrext.append(omegatauext)

evals = np.array(evals)
Mext = np.array(Mext)
tsrext = np.array(tsrext)
M = np.array(M)
tsr = np.array(tsr)


chosenslice = np.isclose(time,4.059e-3,atol=1e-8)

beginslice=np.where(chosenslice)[0][0]
endslice=np.where(chosenslice)[0][-1]+1

TSRAPI = []

for step in range(beginslice,endslice):
    gas.constP = Pressure[step]
    stateYT = np.append(Y[step],Temp[step])
    rhsdiffYT = np.append(diffY[step],diffTemp[step])
    gas.set_stateYT(stateYT)
    gas.update_kernel()
    omegatauext, api = gas.calc_extended_TSRindices(diff=rhsdiffYT)

    TSRAPI.append(api)

TSRAPI = np.array(TSRAPI)

#plot eigenvalues and lambda_M+1
evalM = utils.select_eval(evals,M)
logevals = np.clip(np.log10(1.0+np.abs(evals.real)),0,100)*np.sign(evals.real)
logTSR = np.clip(np.log10(1.0+np.abs(tsr)),0,100)*np.sign(tsr)
logTSRext = np.clip(np.log10(1.0+np.abs(tsrext)),0,100)*np.sign(tsrext)
print('plotting eigenvalues...')
fig, ax = plt.subplots(figsize=(6,4))
for idx in range(evals.shape[1]):
    ax.plot(zeta[beginslice:endslice], logevals[beginslice:endslice,idx], color='gray', marker='.', markersize = 3,linestyle = 'None')
ax.plot(zeta[beginslice:endslice], logTSR[beginslice:endslice], color='red', marker='.', markersize = 6,linestyle = '-', label='TSR')
ax.plot(zeta[beginslice:endslice], logTSRext[beginslice:endslice], color='green', marker='.', markersize = 5,linestyle = '-', label='TSRext')

ax.set_xlabel('mixture fraction z')
ax.set_ylabel('evals')
ax.set_ylim([-9, 6])
ax.set_xlim([0.16, 0.24])
ax.grid(False)
ax.legend()
plt.show(block = False)
plt.savefig('TSR.png', dpi=500, transparent=False)


#plot TSR-API
procnames = np.concatenate((gas.reaction_names(),np.char.add("conv-",gas.species_names+["Temperature"]),np.char.add("diff-",gas.species_names+["Temperature"])))
nr=len(gas.reaction_names())
nv=len(gas.species_names)+1
thr = 0.15
TSRreacIdx = np.unique(np.nonzero(np.abs(TSRAPI) > thr)[1])  #indexes of processes with TSRapi > thr
TSRreac = TSRAPI[:,TSRreacIdx]

source_cmap=plt.cm.get_cmap('tab20')
c_list=[]
for color in np.arange(0,1.05,0.05):
    c_list.append( source_cmap(color) )
c_list.reverse()   
c_list = c_list[1::4]+c_list[::4]+c_list[2::4]+c_list[3::4]

print('plotting TSR-API indices...')
fig, ax1 = plt.subplots(figsize=(9,4))
ax2 = ax1.twinx()
for idx in range(len(TSRreacIdx)):
    #color = next(ax._get_lines.prop_cycler)['color']
    ax1.plot(zeta[beginslice:endslice], TSRreac[:,idx], c=c_list[idx] , label=procnames[TSRreacIdx[idx]], linestyle='-' if TSRreacIdx[idx] < nr else '--' if nr < TSRreacIdx[idx] < nr+nv else '-.')
ax2.plot(zeta[beginslice:endslice],Temp[beginslice:endslice], label="Temperature", c='black')
ax1.set_xlabel('mixture fraction Z [-]')
ax1.set_ylabel('TSR API')
ax2.set_ylabel('Temperature [K]')
ax1.set_xlim([0.16, 0.24])
ax2.set_ylim([900, 1700])
ax1.grid(False)
box = ax.get_position()
ax1.set_position([box.x0, box.y0, box.width * 1.2, box.height])
ax1.legend(loc='center right', bbox_to_anchor=(2.5, 0.5))
plt.show(block = False)
plt.savefig('TSR-API.png', dpi=800, transparent=False)

#contour plot
t = list(set(time))
z = list(set(zeta))
t.sort()
z.sort()
tsr2d = logTSR.reshape((len(t),len(z)))
tsrext2d = logTSRext.reshape((len(t),len(z)))
temp2d = Temp.reshape((len(t),len(z)))
x1, y1 = np.meshgrid(z, t)


fig, ax = plt.subplots(figsize=(8,6))
im=plt.contourf(x1, y1, tsr2d, 20, cmap='RdYlBu_r', vmin = -6, vmax = 6)
plt.xlim(0, 0.5)
plt.ylim(0.0039, 0.00415)
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
plt.xlabel("mixture fraction Z [-]")
plt.ylabel("time [s]")
plt.title("TSR")
cb = plt.colorbar(im)
cb.set_ticks((-6,-4,-2,0,2,4,6))
plt.show(block = False)
plt.savefig('TSRcontour.png', dpi=500, transparent=False)

fig, ax = plt.subplots(figsize=(8,6))
im=plt.contourf(x1, y1, tsrext2d, 20, cmap='RdYlBu_r', vmin = -6, vmax = 6)
plt.xlim(0, 0.5)
plt.ylim(0.0039, 0.00415)
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
plt.xlabel("mixture fraction Z [-]")
plt.ylabel("time [s]")
plt.title("ext-TSR")
cb = plt.colorbar(im)
cb.set_ticks((-6,-4,-2,0,2,4,6))
plt.show(block = False)
plt.savefig('TSRextcontour.png', dpi=500, transparent=False)

fig, ax = plt.subplots(figsize=(8,6))
img3 = ax.contour(x1, y1, temp2d, cmap='Greys', levels=[1000,1100,1200,1300,1400,1500,1600,1700,1800],linewidths=0.75, alpha=0.6)
img1 = plt.contourf(x1, y1, tsrext2d, 20, cmap='Reds', vmin = 2, vmax = 6, levels=[4,5,6])
img2 = plt.contourf(x1, y1, tsr2d, 20, cmap='Greens', vmin = 4, vmax = 6, levels=[4,5,6])
#ax.clabel(img3, img3.levels, inline=True, fontsize=10)
plt.xlim(0, 0.5)
plt.ylim(0.0039, 0.00415)
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
plt.xlabel("mixture fraction Z [-]")
plt.ylabel("time [s]")
plt.show(block = False)
plt.savefig('fronts.png', dpi=500, transparent=False)