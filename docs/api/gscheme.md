# `GScheme.py`

## class `GScheme`

This class is an ODE system integrator based on the G-Scheme [[6]](../references.md#gscheme2009). The class takes as input an instance of the [`CanteraCSP`](functions.md) class, with the problem type (`constP` or `constRho`) already set.

The G-Scheme partitions the modes into fast (Tail), active and slow (Head) subspaces. Only the active dynamics are integrated, with an explicit RK4 scheme, while the fast and slow subspaces are accounted for by the tail and head corrections. The CSP basis can be reused for several steps (`reuse = True`). The algorithm is described in the [CSP solver and G-Scheme](../theory/solvers.md#g-scheme) theory page.

```python
import PyCSP.Functions as cspF
import PyCSP.GScheme as gsc
gas = cspF.CanteraCSP('gri30.yaml')
...
solver = gsc.GScheme(gas)
```

### Instance attributes

`jacobiantype`
:   Type (*string*).
    Set/get the Jacobian type. Available choices are: `'full'`. Defaults to `'full'`.

`csprtolT`
:   Type (*float*).
    Set/get the CSP relative tolerance for the exhausted modes (Tail) calculation. Defaults to `1.0e-2`.

`cspatolT`
:   Type (*float*).
    Set/get the CSP absolute tolerance for the exhausted modes (Tail) calculation. Defaults to `1.0e-8`.

`csprtolH`
:   Type (*float*).
    Set/get the CSP relative tolerance for the slow modes (Head) calculation. Defaults to `1.0e-4`.

`cspatolH`
:   Type (*float*).
    Set/get the CSP absolute tolerance for the slow modes (Head) calculation. Defaults to `1.0e-10`.

`factor`
:   Type (*float*).
    Set/get the timestep safety factor. Defaults to `0.2`.

`T`
:   Type (*int, read-only*).
    Get the current number of exhausted modes (Tail index).

`H`
:   Type (*int, read-only*).
    Get the current number of exhausted plus active modes (Head index). The number of slow modes is $N_v - N_{el} - H$.

`y`
:   Type (*array of float, read-only*).
    Get the current system state in the form $[\,Y_1, \dots, Y_{N_s}, T\,]$.

`t`
:   Type (*float, read-only*).
    Get the current system time.

`dt`
:   Type (*float, read-only*).
    Get the current system timestep size.

`reuse`
:   Type (*boolean*).
    Activate/deactivate the CSP basis reuse. Defaults to `False`.

`reuseThr`
:   Type (*float*).
    Set/get the threshold for the CSP basis reuse: the basis is recomputed when the norm of the relative change of the Jacobian diagonal exceeds `reuseThr`, or after 250 consecutive reuses. Defaults to `1.0e-1`.

`iter`
:   Type (*int, read-only*).
    Get the number of steps taken.

`lam`, `A`, `B`, `f`, `tau`
:   Type (*arrays, read-only*).
    Get the eigenvalues, right and left basis vectors, amplitudes and timescales currently used by the integrator.

`Tc`, `Hc`
:   Type ([N$_v$] *array of float, read-only*).
    Get the current tail and head corrections.

`normJac`, `basisStatus`, `skipBasis`, `updateBasis`
:   Type (*float*, *int*).
    Diagnostic quantities of the basis-reuse algorithm: norm of the relative change of the Jacobian diagonal, 1 if the basis was updated in the last step (0 if reused), number of consecutive reuses, total number of basis updates.

### Class methods

`set_integrator()`
:   Set the integrator parameters. Does not return any output. Unknown keywords raise a `ValueError`.

    Optional keyword arguments:

    - `cspRtolTail =` *float*. Relative tolerance for the exhausted modes calculation.
    - `cspAtolTail =` *float*. Absolute tolerance for the exhausted modes calculation.
    - `cspRtolHead =` *float*. Relative tolerance for the slow modes calculation.
    - `cspAtolHead =` *float*. Absolute tolerance for the slow modes calculation.
    - `factor =` *float*. Timestep safety factor.
    - `jacobiantype =` *string*. Jacobian type.

`set_initial_value(y0, t0)`
:   Set the integrator initial condition. Does not return any output.

    Mandatory positional arguments:

    - `y0` ([N$_s$+1] *array of float*). Initial state in the form $[\,Y_1, \dots, Y_{N_s}, T\,]$.
    - `t0` (*float*). Initial time.

`integrate()`
:   Advance in time by one timestep. The timestep is calculated according to the $T$-th and $T+1$-th timescales, with a safety factor of `factor`.

`profiling()`
:   Prints detailed CPU time consumption and statistics.
