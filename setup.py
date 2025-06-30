#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="PyCSP-lib", 
    version="1.4.0",
    author="Riccardo Malpica Galassi",
    author_email="riccardo.malpicagalassi@uniroma1.it",
    description="A collection of tools for the Computational Singular Perturbation analysis of chemically reacting flows",
    url="https://github.com/rmalpica/PyCSP",
    packages=['PyCSP'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3',
    install_requires=[
          "Cantera>=3.0","numpy","matplotlib","pandas",
      ],
)
