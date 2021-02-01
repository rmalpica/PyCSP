# PyCSP
A collection of tools based on Computational Singular Perturbation for the analysis of chemically reacting systems. 
Requires Python <=3.7.0, cantera=2.4.0, matplotlib

Installation in a new environment called "py36" (from PyCSP folder):
1) conda create --name py36 python=3.6 anaconda --file requirements.txt --channel default --channel anaconda --channel cantera
2) conda activate py36
3) pip install $PATH_TO_PyCSP_folder


Testing:
1) enter the folder tests/
2) run the command "python test_kernel.py"

Several examples are available in the Examples folder to test the functionalities related to:
- exhausted modes (M) 
- tangential stretching rate (TSR)
- CSP and TSR indices (importance indices, amplitude and timescale participatio indices, TSR amplitude/timescale participation indices)

Cantera chem-input files are required. Cantera offers a utility to convert chemkin-format files into cantera-format: https://cantera.org/tutorials/ck2cti-tutorial.html
