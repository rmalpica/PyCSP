# Classes, attributes and methods

PyCSP is organized in the following modules:

| Module | Content |
|---|---|
| [`ThermoKinetics.py`](thermokinetics.md) | class `CanteraThermoKinetics`: source term, generalized stoichiometric matrix, rates of progress and Jacobians of homogeneous reactors at constant pressure or constant volume. |
| [`Functions.py`](functions.md) | class `CanteraCSP`: CSP kernel, exhausted modes, subspaces, TSR, CSP and TSR indices. |
| [`Solver.py`](solver.md) | class `CSPsolver`: CSP-based ODE integrator. |
| [`GScheme.py`](gscheme.md) | class `GScheme`: G-Scheme ODE integrator. |
| [`Simplify.py`](simplify.md) | class `CSPsimplify`: skeletal mechanism generation. |
| [`utils.py`](utils.md) | reactors, data storage and plotting utilities. |

The class hierarchy is `cantera.Solution` → `CanteraThermoKinetics` → `CanteraCSP`, so that a `CanteraCSP` object is also a Cantera `Solution`, and all of Cantera's attributes and methods are available.

An automatically generated listing of the docstrings is available in the [API reference](../reference.md).
