#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 12:26:32 2021

@author: Riccardo Malpica Galassi, Sapienza University, Roma, Italy
"""
import numpy as np


class CSPsimplify:
    def __init__(self, gas, dataset):   
        self._gas = gas
        self._nr = self._gas.n_reactions
        self._ns = self._gas.n_species
        self._nv = self._gas.n_species+1
        self.csprtol = 1e-2
        self.cspatol = 1e-8
        self.scaled = True
        self.problemtype = 'constP'
        self.dataset = dataset
        self.targetset = {}
        self.ImpoFast = []
        self.ImpoSlow = []
        self.speciestype = []
 
    @property 
    def targetset(self):
        return self._targetset
    
    @targetset.setter
    def targetset(self,items):
        self._targetset = items

    @property 
    def nv(self):
        return self._nv
    
    @property 
    def nr(self):
        return self._nr
    
    @property 
    def ns(self):
        return self._ns
    
    @property
    def csprtol(self):
        return self._csprtol
          
    @csprtol.setter
    def csprtol(self,value):
        self._csprtol = value
        self._gas.csprtol = value
    
    @property
    def cspatol(self):
        return self._cspatol
          
    @cspatol.setter
    def cspatol(self,value):
        self._cspatol = value
        self._gas.cspatol = value
             
    @property
    def problemtype(self):
        return self._problemtype
    
    @problemtype.setter
    def problemtype(self,value):
        if value == 'constP' or value == 'constRho':
            self._problemtype = value
        else:
            raise ValueError("Invalid problem type --> %s" %value)


    
    def dataset_info(self):
        import time
        lenData = self.dataset.shape[0] 
        randY =  np.random.dirichlet(np.ones(self.ns),size=1)
        self._gas.TP = 1000,101325.0
        self._gas.Y = randY
        self._gas.constP = 101325.0
        start = time.time()
        api, tpi, ifast, islow, species_type = self._gas.calc_CSPindices(API=False,Impo=True,species_type=True,TPI=False)
        end = time.time()
        time = end-start
        totaltime = time*lenData
        print('Number of species:                %i' % self.ns)
        print('Number of reactions:              %i' % self.nr)
        print('Dataset length:                   %i' % lenData)
        print('Estimated data processing time:  %10.3e [s]' % totaltime)
        
   
    def process_dataset(self):
        #calc CSP basis and importance indexes
        self._gas.jacobiantype = 'full'
        lenData = self.dataset.shape[0] 
        self.ImpoFast = np.zeros((lenData,self.nv,2*self.nr), dtype=float)
        self.ImpoSlow = np.zeros((lenData,self.nv,2*self.nr), dtype=float)
        self.speciestype = np.zeros((lenData,self.nv), dtype=object)
        for idx in range(lenData):
            self._gas.Y = self.dataset[idx,:-2]
            self._gas.TP = self.dataset[idx,-2],self.dataset[idx,-1]
            if (self.problemtype == 'constP'):
                self._gas.constP = self.dataset[idx,-1]
            else:
                rho = self._gas.density
                self._gas.constRho = rho
            api, tpi, ifast, islow, species_type = self._gas.calc_CSPindices(API=False,Impo=True,species_type=True,TPI=False)
            self.ImpoFast[idx] = np.abs(ifast)
            self.ImpoSlow[idx] = np.abs(islow)
            self.speciestype[idx] = species_type

        
    def simplify_mechanism(self, threshold):
        if len(self.ImpoFast) == 0:
            raise ValueError("Need to process dataset first")
        if len(self.targetset) == 0:
            raise ValueError("Need to define targetset")
        lenData = self.dataset.shape[0] 
        all_active_species = [] 
        all_active_reacs = np.zeros((lenData,2*self.nr),dtype=int)
        for idx in range(lenData): #loop over dataset points
            active_species = self.targetset.copy()
            trace,fast,slow = self.get_species_sets(self.speciestype[idx])
            iter = 0            
            while True:
                
                previous_active = active_species.copy()
                #update species relevant to active species
                #print('iteration %i' % iter)
                active_reactions = np.zeros(2*self.nr)
                for i,specname in zip(range(self.ns),self._gas.species_names):               
                    newreactions = np.zeros(2*self.nr)
                    if specname in active_species and specname in slow:
                        newreactions = find_active_reactions(i,self.ImpoSlow[idx],threshold,self.scaled)
                    elif specname in active_species and specname in fast:
                        newreactions = find_active_reactions(i,self.ImpoFast[idx],threshold,self.scaled)
                    active_reactions[newreactions == 1] = 1  
                    
                #update species relevant to temperature
                newreactions = find_active_reactions(self.ns,self.ImpoSlow[idx],threshold,self.scaled)  
                active_reactions[newreactions == 1] = 1  
                
                newspecies = self.find_species_in_reactions(active_reactions)
                active_species.update(newspecies)            
                
                #remove trace species
                active_species.difference_update(trace)
                
                #print('after cleaning traces')
                #print(active_species)
                                
                iter = iter + 1
                if active_species == previous_active:
                    break
            all_active_species.append(active_species)  #list containing active species in each datapoint
            all_active_reacs[idx] = active_reactions  #list containing active reactions in each datapoint
            
      
        #union of active reactions
        reactions = self.unite_active_reactions(all_active_reacs)
        species = set().union(*all_active_species)
        
        
        #recovery: grab all the reactions containing active species
        reactions = self.find_reactions_given_species(species)
        species = [self._gas.species(name) for name in species]  #convert to species object

        print('@threshold: %1.3f, Simplified mechanism contains %i species and %i reactions' % (threshold, len(species), len(reactions)))
        
        return species, reactions



    def find_species_in_reactions(self, active_reactions):
        species = {}
        for k in range(self.nr):
            if active_reactions[k] == 1:
                species.update(self._gas.reaction(k).reactants)
                species.update(self._gas.reaction(k).products)        
            if active_reactions[self.nr+k] == 1:
                species.update(self._gas.reaction(k).reactants)
                species.update(self._gas.reaction(k).products)  
        return species
    
    
    def get_species_sets(self,speciestype):
        trace = set()
        fast = set()
        slow = set()
        for k,item in zip(range(self.ns),speciestype):            
            if item == 'trace': 
                trace.add(self._gas.species_name(k))
            elif item == 'slow': 
                slow.add(self._gas.species_name(k))
            elif item == 'fast': 
                fast.add(self._gas.species_name(k))
        return trace,fast,slow            

    def unite_active_reactions(self, reacs):
        reactions = []
        for k in range(self.nr):
            if np.any(reacs[:, k] == 1) or np.any(reacs[:, k+self.nr] == 1):
                reactions.append(self._gas.reaction(k)) 
        return reactions
            
    
    def find_reactions_given_species(self,species):
        reactions = []
        for k in range(self.nr):
            if set(self._gas.reaction(k).reactants.keys()).issubset(species) and set(self._gas.reaction(k).products.keys()).issubset(species):
                reactions.append(self._gas.reaction(k)) 
        return reactions
            
                
    
def find_active_reactions(ivar,impo,thr,scaled):
    if scaled:
        Imax = np.max(np.abs(impo[ivar]))
    else:
        Imax = 1.0   
    active_reactions = np.zeros(impo.shape[1])
    if Imax > 0: #protects against all-zeros ImpoIndex
        active_reactions[impo[ivar] >= thr*Imax] = 1
 
    return active_reactions


            