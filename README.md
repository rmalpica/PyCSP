# PyCSP
![Screenshot](logo.png)
A collection of tools based on Computational Singular Perturbation for the analysis of chemically reacting systems. 
Requires Python <=3.7.0, cantera=2.4.0, numpy, matplotlib

Installation in a new environment called "py36" with Anaconda (suggested, oterwise skip to #3):
From PyCSP folder
1) conda create --name py36 python=3.6 anaconda --file requirements.txt --channel default --channel anaconda --channel cantera
2) conda activate py36
3) pip install $PATH_TO_PyCSP_MAIN_FOLDER (e.g. pip install /Users/rmalpica/PyCSP)


Testing:
1) enter the folder tests/
2) run the command "python test_kernel.py"

Several examples are available in the Examples folder to test the functionalities related to:
- exhausted modes (M) 
- tangential stretching rate (TSR)
- CSP and TSR indices (importance indices, amplitude and timescale participatio indices, TSR amplitude/timescale participation indices)

Cantera chem-input files are required. Cantera offers a utility to convert chemkin-format files into cantera-format: https://cantera.org/tutorials/ck2cti-tutorial.html

# Documentation
Can be found in the /documentation folder

# How to cite?
This code still doesn't have an associated publication. In addition to mentioning this GitHub repository (see below), I would be grateful if you could cite the following articles, which contain results produced with PyCSP.
- Malpica Galassi, R., Ciottoli, P. P., Sarathy, S. M., Im, H. G., Paolucci, S., & Valorani, M. (2018). Automated chemical kinetic mechanism simplification with minimal user expertise. Combustion and Flame, 197, 439–448. https://doi.org/10.1016/j.combustflame.2018.08.007
- AlRamadan, A. S., Malpica Galassi, R., Ciottoli, P. P., Valorani, M., & Sarathy, S. M. (2020). Multi-stage heat release in lean combustion: Insights from coupled tangential stretching rate (TSR) and computational singular perturbation (CSP) analysis. Combustion and Flame, 219, 242–257. https://doi.org/10.1016/j.combustflame.2020.05.026

BibTex citation of this repository:
```bibtex
@misc{malpicagalassi2021pycsp,
    author = {Riccardo Malpica Galassi},
    title = {PyCSP - A collection of tools based on Computational Singular Perturbation for the analysis of chemically reacting systems.},
    year = {2021},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/rmalpica/PyCSP/}},
}
```
