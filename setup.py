#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="PyCSP", 
    version="0.0.3",
    author="Riccardo Malpica Galassi",
    author_email="riccardo.malpicagalassi@uniroma1.it",
    description="A collection of tools for the Computational Singular Perturbation analysis of chemically reacting flows",
    url="https://github.com/rmalpica/PyCSP",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.0.*,<3.7',
    install_requires=[
          "Cantera==2.4.0","numpy","matplotlib",
      ],
)