# Introduction

<div class="pycsp-title" markdown>
**PyCSP `v1.5.0`**

![PyCSP logo](assets/img/logo.png)

<hr>
<span class="subtitle">Software documentation</span>
<hr>
</div>

PyCSP is a collection of libraries for the analysis, model reduction and ODE integration of reacting systems using the computational singular perturbation (CSP) method and its extensions, including the Tangential Stretching Rate (TSR).

It provides tools for the local characterization of the chemical dynamics, built on the eigen-decomposition of the Jacobian of the chemical source term:

- the CSP kernel: eigenvalues, right/left basis vectors, mode amplitudes and timescales;
- exhausted (fast) modes, and the fast/active/slow subspace partitioning;
- the Tangential Stretching Rate, and its extension to reaction-diffusion-convection systems;
- amplitude and timescale participation indices, fast/slow importance indices, CSP pointers and species classification;
- skeletal mechanism generation (CSP simplification);
- two CSP-based ODE integrators (CSP solver and G-Scheme).

The theory behind these tools is summarized in three pages: [CSP and TSR analysis](theory/analysis.md), [mechanism simplification](theory/simplification.md) and [CSP solver and G-Scheme](theory/solvers.md).

## Copyright and license

Copyright &copy; 2025 Riccardo Malpica Galassi.

PyCSP is released under the [Prosperity Public License 3.0.0](https://prosperitylicense.com/versions/3.0.0). You may use, copy, modify, and share this software for **noncommercial purposes only**. For commercial use, a commercial license must be obtained from the copyright holder. See the `LICENSE` file in the repository.

!!! note
    The documentation of v1.4 and earlier releases stated the MIT license.

## Contacts

Riccardo Malpica Galassi: `riccardo.malpicagalassi [at] uniroma1.it`, `riccardo.malpica [at] gmail.com`

## How to cite

This code has an associated publication. In addition to mentioning this GitHub repository, I would be grateful if you could cite the publication:

Malpica Galassi, R., PyCSP: a Python package for the analysis and simplification of chemically reacting systems based on Computational Singular Perturbation, *Computer Physics Communications*, 2022, [https://doi.org/10.1016/j.cpc.2022.108364](https://doi.org/10.1016/j.cpc.2022.108364).

BibTeX citation of this publication:

```bibtex
@article{MALPICAGALASSI2022108364,
title = {PyCSP: a Python package for the analysis and simplification of chemically
reacting systems based on Computational Singular Perturbation},
journal = {Computer Physics Communications},
pages = {108364},
year = {2022},
issn = {0010-4655},
doi = {https://doi.org/10.1016/j.cpc.2022.108364},
url = {https://www.sciencedirect.com/science/article/pii/S0010465522000832},
author = {Riccardo {Malpica Galassi}}
}
```

## Version history

### Changes with respect to previous version (v1.4)

Bug fixes that change numerical results:

- **Constant-pressure Jacobian.** The species columns were evaluated at constant *density* instead of constant pressure: perturbing a mass fraction changed the pressure. All Jacobians are now taken holding the thermodynamic constraint of the problem fixed (pressure for `const_p`, density for `const_v`). This affects every constant-pressure CSP/TSR quantity. In the H$_2$/air ignition test case, the Jacobian changes by up to about 1% (relative to its largest entries) and the TSR by up to 5% near ignition, while the number of exhausted modes is unchanged.
- **Per-reaction Jacobian contributions (`jac_contribution`).** The same constant-pressure correction; for constant-volume problems the temperature column is now taken at constant density. The contributions now sum to the Jacobian in both formulations, which affects the timescale participation indices (TPI) and TSR-TPI.
- **Timescale participation indices of complex-conjugate modes.** The indices of a complex pair summed to zero; they now decompose $\mathrm{Re}(\lambda)$.
- **Constrained Jacobian (`jacobiantype='constrained'`) for constant-volume problems.** It now uses the internal energies and $c_v$ (constant internal-energy constraint) instead of enthalpies and $c_p$.
- **TSR.** Modes with zero eigenvalue are now excluded consistently with the exhausted-modes calculation.
- `utils.integrate_batch_constV` now sets `constRho` to the density of the actual mixture.

Other bug fixes:

- `calc_subspaces` (and `findH`) no longer crash with the `kinetic` and `constrained` Jacobian types.
- The thermochemical state of the object is now always restored after Jacobian evaluations (it was left perturbed with the `kinetic`/`constrained` types).
- `jacobian_diagonal` for constant-volume problems used the constant-pressure source term.
- TPI are not available for `jacobiantype='constrained'` (the thermal correction is not split by reaction): a `NotImplementedError` is raised instead of returning inconsistent values.
- `utils.save_CSP_plots`: fixed crashes with short time series, and the rate-of-progress and RHS plots.
- `CSPsimplify.get_tsrtarget` no longer fails for `TSRtol = 1`.
- `CSPsolver`/`GScheme.set_integrator` raise a `ValueError` on unknown keywords instead of exiting the interpreter; `profiling()` no longer divides by zero.

New features and changes:

- Optional `norm_cut` argument for the CSP indices (`calc_CSPindices`, `calc_TSRindices`, `CSPsimplify.norm_cut`): rows whose sum of absolute contributions is below the cut are set to zero instead of being normalized. Contributed by students.
- `utils.shared_reactor`, for creating Cantera reactors that share the `CanteraCSP` object across Cantera versions.
- A pytest test-suite with correctness tests and golden-master regression tests, run by continuous integration.
- Removed `soln2cti.py` and `soln2ck.py`, which only worked with Cantera 2.x. Use Cantera's `Solution.write_yaml()` and `Solution.write_chemkin()` instead.
- Removed unused legacy functions (`findM_original`, `setEwt_original`, `modeContribution_original`, `CSPtiming`).
- Documentation moved from PDF to this website.

### Changes with respect to previous version (v1.3)

- Full compatibility with Cantera 3.0
- new CSP-analysis examples
- minor bug-fixes

### Changes with respect to previous version (v1.2)

- Added G-Scheme solver
- minor bug-fixes

### Changes with respect to previous version (v1.1)

- minor bug-fixes

### Changes with respect to previous version (v1.0)

- Added extended TSR analysis
- minor bug-fixes

## Folder structure

```text
PyCSP
├── PyCSP
│   ├── __init__.py
│   ├── ThermoKinetics.py
│   ├── Functions.py
│   ├── Solver.py
│   ├── GScheme.py
│   ├── Simplify.py
│   ├── utils.py
│   └── profiling_tools.py
├── tests
│   ├── conftest.py
│   ├── test_correctness.py
│   ├── test_golden.py
│   ├── test_kernel.py
│   ├── test_norm_cut.py
│   ├── test_rhs.py
│   ├── test_rhs_constV.py
│   ├── test_utils_and_solvers.py
│   └── golden
│       ├── make_golden.py
│       └── golden_h2.npz
├── examples
│   ├── cspAnalysis
│   │   ├── exhaustedModes.py
│   │   ├── full_CSP_analysis.py
│   │   ├── indices.py
│   │   ├── indices-integral.py
│   │   ├── JacobianSIM.py
│   │   ├── pointers.py
│   │   └── subspaces.py
│   ├── dataImportExport
│   │   └── export_import.py
│   ├── cspSolver
│   │   ├── cspsolver_const_p.py
│   │   ├── cspsolver_const_v.py
│   │   ├── gscheme_const_p.py
│   │   └── gscheme_const_p_reuse.py
│   ├── cspSimplify
│   │   ├── simp.py
│   │   └── simp_TSR.py
│   └── tsrAnalysis
│       ├── TSR.py
│       └── TSR_ext.py
└── docs        (this documentation)
```
