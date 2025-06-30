# PyCSP
![Screenshot](logo.png)
A collection of tools based on Computational Singular Perturbation for the analysis of chemically reacting systems. 
Requires cantera >= 3.0, numpy, matplotlib. (for older versions of Cantera, i.e. Cantera>=2.5, download release v1.2.1)

Installation in a new environment called "pycsp" with Anaconda (suggested, oterwise skip to #3):
From PyCSP folder
1) conda create --name pycsp anaconda --file requirements.txt --channel default --channel anaconda --channel cantera
2) conda activate pycsp
3) pip install PyCSP-lib 


Testing:
1) enter the folder tests/
2) run the command "python test_kernel.py"

Several examples are available in the Examples folder to test the functionalities related to:
- exhausted modes (M) 
- tangential stretching rate (TSR)
- CSP and TSR indices (importance indices, amplitude and timescale participatio indices, TSR amplitude/timescale participation indices)

Cantera chem-input files are required (.cti or .yaml, depending on the installed Cantera version). Cantera offers a utility to convert chemkin-format files into cantera-format: https://cantera.org/tutorials/ck2cti-tutorial.html

FluidFoam (https://github.com/fluiddyn/fluidfoam) is suggested as a direct interface with openFOAM data.

# Warning
Datasets in the tsrAnalysis example folder are quite heavy. Due to limited github bandwidth, it may happen that "flamelet_state.dat" and "flamelet_rhsDiff.dat" are not correctly checked out. In that case, please write me an e-mail (riccardo.malpicagalassi [at] uniroma1.it). I will send you the files.

# Documentation
Can be found in the /documentation folder

# How to cite?
This code has an associated publication. In addition to mentioning this GitHub repository (see below), I would be grateful if you could cite the publication: 
- Malpica Galassi, R., PyCSP: a Python package for the analysis and simplification of chemically reacting systems based on Computational Singular Perturbation, Computer Physics Communications, 2022, https://doi.org/10.1016/j.cpc.2022.108364.
(https://www.sciencedirect.com/science/article/pii/S0010465522000832)

BibTex citation of this publication:
```bibtex
@article{MALPICAGALASSI2022108364,
title = {PyCSP: a Python package for the analysis and simplification of chemically reacting systems based on Computational Singular Perturbation},
journal = {Computer Physics Communications},
pages = {108364},
year = {2022},
issn = {0010-4655},
doi = {https://doi.org/10.1016/j.cpc.2022.108364},
url = {https://www.sciencedirect.com/science/article/pii/S0010465522000832},
author = {Riccardo {Malpica Galassi}}
}

```

