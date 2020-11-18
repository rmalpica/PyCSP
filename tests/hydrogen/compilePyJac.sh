#!/bin/bash

#before running this, execute "conda activate py36"

home="$(pwd)"

python -m pyjac --lang c --input chem.inp -t therm.dat

python -m pyjac.pywrap --source_dir $home/out --lang c
