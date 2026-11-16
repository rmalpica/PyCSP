# `Solver.py`

## class `CSPsolver`

This class is an ODE system integrator based on the CSP method. The class takes as input an instance of the [`CanteraCSP`](functions.md) class, with the problem type (`constP` or `constRho`) already set.

At each step the solver applies the radical correction for the $M$ exhausted modes and integrates the projected system $\dot{\mathbf{y}} = \mathbf{Q}_s\,\mathbf{g}$ with an explicit RK4 scheme. The algorithm is described in the [CSP solver and G-Scheme](../theory/solvers.md#csp-solver) theory page.

```python
import PyCSP.Functions as cspF
import PyCSP.Solver as cspS
gas = cspF.CanteraCSP('hydrogen.yaml')
...
solver = cspS.CSPsolver(gas)
```

### Instance attributes

`jacobiantype`
:   Type (*string*).
    Set/get the Jacobian type. Available choices are: `'full'`. Defaults to `'full'`.

`csprtol`
:   Type (*float*).
    Set/get the CSP relative tolerance for the exhausted modes calculation. Defaults to `1.0e-2`.

`cspatol`
:   Type (*float*).
    Set/get the CSP absolute tolerance for the exhausted modes calculation. Defaults to `1.0e-8`.

`factor`
:   Type (*float*).
    Set/get the timestep safety factor. Defaults to `0.2`.

`M`
:   Type (*int, read-only*).
    Get the current number of exhausted modes.

`y`
:   Type (*array of float, read-only*).
    Get the current system state in the form $[\,Y_1, \dots, Y_{N_s}, T\,]$.

`t`
:   Type (*float, read-only*).
    Get the current system time.

`dt`
:   Type (*float, read-only*).
    Get the current system timestep size.

`Qs`
:   Type ([N$_s$+1 $\times$ N$_s$+1] *array of float, read-only*).
    Get the current projection matrix.

`Rc`
:   Type ([N$_s$+1] *array of float, read-only*).
    Get the current radical correction vector.

### Class methods

`set_integrator()`
:   Set the integrator parameters. Does not return any output. Unknown keywords raise a `ValueError`.

    Optional keyword arguments:

    - `cspRtol =` *float*. Relative tolerance for the exhausted modes calculation.
    - `cspAtol =` *float*. Absolute tolerance for the exhausted modes calculation.
    - `factor =` *float*. Timestep safety factor.
    - `jacobiantype =` *string*. Jacobian type.

`set_initial_value(y0, t0)`
:   Set the integrator initial condition. Does not return any output.

    Mandatory positional arguments:

    - `y0` ([N$_s$+1] *array of float*). Initial state in the form $[\,Y_1, \dots, Y_{N_s}, T\,]$.
    - `t0` (*float*). Initial time.

`integrate()`
:   Advance in time by one timestep. The timestep is calculated according to the $M$-th and $M+1$-th timescales, with a safety factor of `factor`.

`profiling()`
:   Prints detailed CPU time consumption and statistics.
