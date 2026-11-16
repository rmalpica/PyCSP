# `Functions.py`

## class `CanteraCSP`

This class inherits attributes and methods from the [`CanteraThermoKinetics`](thermokinetics.md) class, which is in turn a derived class of Cantera's `Solution` class (multilevel inheritance), and extends their capabilities by adding computational singular perturbation (CSP)-related class methods. Hence, the constructor for this class is equivalent to the constructor of Cantera's `Solution` class. The proper way to instantiate `CanteraCSP` objects is by using species and reactions defined in an input file, e.g. `hydrogen.yaml`:

```python
import PyCSP.Functions as csp
gas = csp.CanteraCSP('hydrogen.yaml')
```

The definitions of all the CSP quantities are given in the [Theory](../theory/analysis.md) page.

!!! note "Kernel caching"
    The CSP kernel is recomputed lazily: the `calc_*` methods and `get_kernel()` recompute it only if an attribute of the object has been set since the last calculation (e.g. `TPY`, `Y`, `constP`, `set_stateYT(...)`). State changes made internally by Cantera, e.g. by a `ReactorNet` advancing a reactor that shares the object, are not detected. In that case, set the state explicitly (e.g. `gas.TPY = gas.TPY`) or call `update_kernel()` before querying the kernel.

### Instance attributes

`jacobiantype`
:   Type (*string*).
    Set/get the Jacobian type. Available choices are: `'full'`, `'kinetic'`, `'constrained'` (see [Theory](../theory/analysis.md#jacobian)). Defaults to `'full'`. Note that this choice affects the dimension of the system ($N_v = N_s+1$ for `full`, $N_v = N_s$ otherwise), and in turn the dimension of the CSP-related attributes (eigenvalues, eigenvectors, timescales, amplitudes).

`rtol`
:   Type (*float*).
    Set/get the CSP relative tolerance for the exhausted modes calculation. Defaults to `1.0e-2`.

`atol`
:   Type (*float*).
    Set/get the CSP absolute tolerance for the exhausted modes calculation. Defaults to `1.0e-8`.

`rhs`
:   Type ([N$_v$] *array of float, read-only*).
    Get the chemical source term used for the CSP calculations.

`jac`
:   Type ([N$_v$ $\times$ N$_v$] *array of float, read-only*).
    Get the Jacobian of the chemical source term used for the CSP calculations.

`evals`
:   Type ([N$_v$] *array of complex, read-only*).
    Get the Jacobian eigenvalues, sorted by decreasing modulus. The last $N_{el}$ (number of elements) eigenvalues are set to zero.

`Revec`
:   Type ([N$_v$ $\times$ N$_v$] *array of float, read-only*).
    Get the matrix of the Jacobian right eigenvectors (CSP basis vectors $\mathbf{a}_i$). The eigenvectors are assembled as row vectors, hence the i-th eigenvector can be accessed using `Revec[i]`. Note that the properly defined right eigenvectors matrix is the transpose of `Revec`. Complex-conjugate pairs are stored as real and imaginary parts.

`Levec`
:   Type ([N$_v$ $\times$ N$_v$] *array of float, read-only*).
    Get the matrix of the Jacobian left eigenvectors (CSP co-basis vectors $\mathbf{b}^i$), obtained by inversion of the right eigenvectors matrix. The eigenvectors are assembled as row vectors, hence the i-th eigenvector can be accessed using `Levec[i]`.

`f`
:   Type ([N$_v$] *array of positive float, read-only*).
    Get the CSP mode amplitudes.

`tau`
:   Type ([N$_v$] *array of float, read-only*).
    Get the modes timescales $\tau_i = 1/|\lambda_i|$ ($10^{20}$ for zero eigenvalues).

`index_norm`
:   Type (*string*).
    Available choices are: `'abs'`, `'none'`. Defaults to `'abs'`. Set the way Importance and Participation indices are normalized. When `'abs'`, indices are normalized with their 1-norm (sum of absolute values). This means that their absolute values sum to 1. When `'none'`, indices are not normalized (to be used for integral indices only).

`classify_traces`
:   Type (*boolean*).
    When `True` (default), the species classification also identifies trace species.

`nUpdates`
:   Type (*int, read-only*).
    Get the number of kernel updates, starting from the instantiation of the object.

### Class methods

`update_kernel()`
:   Updates the CSP kernel (`evals, Revec, Levec, f, tau`), based on the current thermochemical state, employing the Jacobian formulation specified by setting either `constP` or `constRho`, and the Jacobian type specified with `jacobiantype`. Does not return any output.

`get_kernel()`
:   Returns the content of the `[evals, Revec, Levec, f]` attributes. If any attributes of the instance are changed with respect to the previous kernel calculation, the kernel is recomputed and returned.

`calc_exhausted_modes()`
:   Computes and returns the number of exhausted modes $M$ (*int*).

    Optional keyword arguments:

    - `rtol =` *float*. Relative tolerance. If not specified, the instance attribute is employed.
    - `atol =` *float*. Absolute tolerance. If not specified, the instance attribute is employed.

`calc_subspaces()`
:   Computes and returns the indices $M$ and $H$ (also known as Tail and Head in the G-Scheme nomenclature [[6]](../references.md#gscheme2009)) corresponding to the subspaces partitioning into fast/active/slow modes `[0:M]`, `[M:H]`, `[H:N]` (*int*, *int*).

    Optional keyword arguments:

    - `rtolTail =` *float*. Relative tolerance for M. If not specified, the instance attribute is employed.
    - `atolTail =` *float*. Absolute tolerance for M. If not specified, the instance attribute is employed.
    - `rtolHead =` *float*. Relative tolerance for H. If not specified, the instance attribute is employed.
    - `atolHead =` *float*. Absolute tolerance for H. If not specified, the instance attribute is employed.

`calc_TSR()`
:   Computes the number of exhausted modes and the Tangential Stretching Rate. Returns the Tangential Stretching Rate (*float*), or `[TSR, M]` if `getM=True`.

    Optional keyword arguments:

    - `rtol =` *float*. Relative tolerance for the exhausted modes calculation. If not specified, the instance attribute is employed.
    - `atol =` *float*. Absolute tolerance for the exhausted modes calculation. If not specified, the instance attribute is employed.
    - `getM =` *boolean*. If `True`, the number of exhausted modes is returned as well.

`calc_CSPindices()`
:   Computes the number of exhausted modes and returns the requested CSP indices: amplitude participation indices (API), timescale participation indices (TPI), fast and slow importance indices, and the species classification (fast/slow/trace), as the list `[API, TPI, Ifast, Islow, species_type]` ([N$_v \times$ 2N$_r$] *array of float* for each index, [N$_v$] *array of strings* for the classification). The quantities that are not requested are returned as `None`. If `getM=True`, $M$ is appended to the list.

    Optional keyword arguments:

    - `API =` *boolean*. If `True`, computes the amplitude participation indices. Defaults to `False`.
    - `TPI =` *boolean*. If `True`, computes the timescale participation indices. Defaults to `False`. Not available for `jacobiantype='constrained'`.
    - `Impo =` *boolean*. If `True`, computes the fast and slow importance indices. Defaults to `False`.
    - `species_type =` *boolean*. If `True`, computes the species classification. Defaults to `False`.
    - `rtol =` *float*. Relative tolerance for the exhausted modes calculation.
    - `atol =` *float*. Absolute tolerance for the exhausted modes calculation.
    - `getM =` *boolean*. If `True`, the number of exhausted modes is returned as well.
    - `norm_cut =` *float* or [N$_v$] *array of float*. With `index_norm='abs'`, the rows whose sum of absolute contributions is smaller than `norm_cut` (a scalar, or one value per row) are set to zero instead of being normalized. The rows are modes for API and TPI, and variables for the importance indices. The units are those of the contributions: the rate of the mode amplitude or of the variable, and 1/s for the TPI. Defaults to `None` (no cut). *New in v1.5.*

`calc_TSRindices()`
:   Computes and returns the Tangential Stretching Rate amplitude (TSR-API) or timescale participation indices (TSR-TPI) ([2N$_r$] *array of float*). If `getTSR=True`, returns the tuple `(TSR, indices)`.

    Optional keyword arguments:

    - `type =` *string*. If `'amplitude'`, the TSR-API are returned. If `'timescale'`, the TSR-TPI are returned. Defaults to `'amplitude'`. The `'timescale'` type is not available for `jacobiantype='constrained'`.
    - `rtol =` *float*. Relative tolerance for the exhausted modes calculation.
    - `atol =` *float*. Absolute tolerance for the exhausted modes calculation.
    - `getTSR =` *boolean*. If `True`, the TSR is returned as well.
    - `norm_cut =` *float* or [N$_v$] *array of float*. Cut on the per-mode API or TPI (see `calc_CSPindices`). *New in v1.5.*

`calc_extended_TSR()`
:   Computes the number of extended exhausted modes ($M_{ext}$) and the extended Tangential Stretching Rate, for systems with transport. Returns the extended TSR (*float*), or `[TSR_ext, Mext]` if `getMext=True`.

    Optional keyword arguments (at least one of `diff` and `conv` should be given):

    - `diff =` [N$_v$] *array of float*. Diffusion right hand side. Array containing the diffusion term of the mass fractions of all the species and of the temperature: $[\,D_1, \dots, D_{N_s}, D_T\,]$. The unit of measure must be [1/s] and [K/s]. The order of the $D_i$ values must comply with the species ordering set at the class instantiation.
    - `conv =` [N$_v$] *array of float*. Convection right hand side $[\,C_1, \dots, C_{N_s}, C_T\,]$, with the same units and ordering.
    - `rtol =` *float*. Relative tolerance for the exhausted modes calculation. If not specified, the instance attribute is employed.
    - `atol =` *float*. Absolute tolerance for the exhausted modes calculation. If not specified, the instance attribute is employed.
    - `getMext =` *boolean*. If `True`, the extended number of exhausted modes is returned as well.

`calc_extended_TSRindices()`
:   Computes and returns the amplitude participation indices to the extended Tangential Stretching Rate ([2N$_r$+N$_v$+N$_v$] *array of float*): the 2N$_r$ reactions followed by the N$_v$ convection and the N$_v$ diffusion terms. If `getTSRext=True`, returns the tuple `(TSR_ext, indices)`.

    Optional keyword arguments:

    - `diff =`, `conv =` as in `calc_extended_TSR`.
    - `rtol =` *float*. Relative tolerance for the exhausted modes calculation.
    - `atol =` *float*. Absolute tolerance for the exhausted modes calculation.
    - `getTSRext =` *boolean*. If `True`, the extended TSR is returned as well.

`kernel()`, `kernel_kinetic_only()`, `kernel_constrained_jac()`
:   Compute and return `[evals, Revec, Levec, f]` for the `full`, `kinetic` and `constrained` Jacobian types. They are called by `update_kernel()`.

`is_changed()`
:   Returns `True` if an attribute of the object has been set since the last kernel calculation.

## Module functions

The methods of `CanteraCSP` are built on module-level functions, which can also be used directly, e.g. on data computed elsewhere.

`eigsys(jac)`
:   Returns `[evals, Revec, Levec]` of a Jacobian, sorted by decreasing eigenvalue modulus, with complex-conjugate pairs stored as real and imaginary parts.

`evec_pos_ampl(Revec, Levec, f)`
:   Changes the sign of the basis vectors so that all the amplitudes are non-negative.

`timescales(evals)`
:   Returns the timescales $1/|\lambda_i|$.

`findM(n_elements, stateYT, evals, Revec, tau, f, rtol, atol)`
:   Returns the number of exhausted modes.

`findH(n_elements, stateYT, evals, Revec, dt, f, rtol, atol, Tail)`
:   Returns the Head index $H$, given the Tail index $M$ and the time horizon `dt`.

`setEwt(y, rtol, atol)`
:   Returns the error weights $\mathrm{rtol}\,|y| + \mathrm{atol}$.

`findTSR(n_elements, rhs, evals, Revec, f, M)`
:   Returns `[TSR, weights]`.

`TSRindices(weights, evals)`
:   Returns the participation indices of the modes to the TSR.

`TSR_participation_indices(TSRidx, CSPidx, normtype)`
:   Returns the participation indices of the processes to the TSR, given the mode-to-TSR indices and the per-mode CSP indices (API or TPI).

`CSP_amplitude_participation_indices(B, Smat, rvec, normtype, norm_cut=None)`
:   Returns the API [N$_v \times$ 2N$_r$].

`CSP_timescale_participation_indices(nr, JacK, evals, A, B, normtype, norm_cut=None)`
:   Returns the TPI [N$_v \times$ 2N$_r$], given the per-reaction Jacobians `JacK` (see `jac_contribution()`).

`CSP_importance_indices(A, B, M, Smat, rvec, normtype, norm_cut=None)`
:   Returns `[Ifast, Islow]`.

`CSPIndices(Proj, Smat, rvec, normtype, norm_cut=None)`
:   Returns the generic indices of the projections $\mathbf{P}\,\mathcal{S}_k\, r^k$, normalized row by row.

`CSP_pointers(A, B)`
:   Returns the CSP pointers, `pointers[m, j]` $= (\mathbf{a}_m \mathbf{b}^m)_{jj}$.

`classify_species(stateYT, rhs, pointers, M, trace)`
:   Returns the classification (`'fast'`, `'slow'`, `'trace'`) of the variables.

`detectFrozenModes(M, stateYT, Revec, Levec, Smat, rvec, rtol, atol)`
:   Returns a boolean mask (one entry per variable index) marking the modes among the first `M` whose projected reaction rates $\sum_k |\mathbf{a}_m \mathbf{b}^m \mathcal{S}_k r^k|$ are all below the error weights $\mathrm{rtol}\,|y| + \mathrm{atol}$ (frozen modes). Used by the [full CSP analysis example](../examples/full-analysis.md).

`add_transport(rhs, Levec, Smat, rvec, rhs_convYT, rhs_diffYT)`
:   Returns the extended source term, amplitudes, stoichiometric matrix and rates vector used by the extended TSR.
