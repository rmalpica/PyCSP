# PyCSP
![Screenshot](logo.png)
A collection of tools based on Computational Singular Perturbation for the analysis of chemically reacting systems. 
Requires cantera >= 2.4.0, numpy, matplotlib.
git-lfs is needed to correctly download the example datasets.

Installation in a new environment called "pycsp" with Anaconda (suggested, oterwise skip to #3):
From PyCSP folder
1) conda create --name pycsp anaconda --file requirements.txt --channel default --channel anaconda --channel cantera
2) conda activate pycsp
3) pip install $PATH_TO_PyCSP_MAIN_FOLDER (e.g. pip install /Users/rmalpica/PyCSP)


Testing:
1) enter the folder tests/
2) run the command "python test_kernel.py"

Several examples are available in the Examples folder to test the functionalities related to:
- exhausted modes (M) 
- tangential stretching rate (TSR)
- CSP and TSR indices (importance indices, amplitude and timescale participatio indices, TSR amplitude/timescale participation indices)

Cantera chem-input files are required. Cantera offers a utility to convert chemkin-format files into cantera-format: https://cantera.org/tutorials/ck2cti-tutorial.html

FluidFoam (https://github.com/fluiddyn/fluidfoam) is suggested as a direct interface with openFOAM data.

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

