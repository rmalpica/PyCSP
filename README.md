# PyCSP
A collection of tools based on Computational Singular Perturbation for the analysis of chemically reacting systems. 
Requires Python 3.6, cantera, matplotlib

Installation in a new environment called "py36":
conda create --name py36 --file requirements.txt --channel default --channel anaconda --channel slackha --channel cantera

Testing:
1) enter the folder tests/hydrogen
2) run the command "python test_kernel.py"

Several other test scripts are available to test the functionalities related to:
- exhausted modes (M) 
- tangential stretching rate (TSR)
- CSP and TSR indices (importance indices, amplitude and timescale participatio indices, TSR amplitude/timescale participation indices)
