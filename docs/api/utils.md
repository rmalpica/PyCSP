# `utils.py`

Utilities for running reactors, storing CSP results and plotting them. This page is new in the v1.5 documentation.

## Reactors

`integrate_batch_constP(gas, temperature, pressure, eqratio, FuComp, OxComp, tend, rtol=1e-9, atol=1e-15)`
:   Integrates a constant-pressure batch reactor from the given initial temperature, pressure and equivalence ratio (fuel and oxidizer compositions as Cantera strings) up to `tend`. It sets `gas.constP` and returns a Cantera `SolutionArray` with the extra columns `t` and `rhsT` (temperature source term).

`integrate_batch_constV(gas, temperature, pressure, eqratio, FuComp, OxComp, tend, rtol=1e-9, atol=1e-15)`
:   The same for a constant-volume reactor; `gas.constRho` is set to the initial mixture density.

`shared_reactor(reactor_class, gas)`
:   Creates a Cantera reactor of class `reactor_class` (e.g. `ct.IdealGasConstPressureReactor`) that shares the `gas` object instead of cloning it, so that `gas` follows the integrated state. This is needed with Cantera > 3.2, which clones the phase by default. *New in v1.5.*

## Data storage

`TaggedDatabase()`
:   A simple container for time series. Values are stored with `db.store(tag1=value1, tag2=value2, ...)` at each step; `db.to_arrays()` converts the stored lists to numpy arrays; each series is then available as `db.tag` or `db['tag']`.

## Plotting

`save_CSP_plots(db, gas, output_folder='CSP_Plots', threshold=0.05, xlim=None, xlabel='time [s]', species_threshold=1e-3)`
:   Generates and saves all the plots allowed by the content of a `TaggedDatabase`. The tags it recognizes are:

    | Tags | Plots |
    |---|---|
    | `API`, `Ifast`, `Islow`, `Pointers`, `TPI` | one image per variable or mode, in the folders `API/`, `Ifast/`, `Islow/`, `Pointers/`, `TPI/` |
    | `tsrApi`, `tsrTpi` | TSR participation indices |
    | `evals`, `M`, `tsr` | eigenvalues, $\lambda_{M+1}$ and TSR |
    | `M` | exhausted modes, number of active modes |
    | `fvec` | mode amplitudes |
    | `state` | temperature and major species |
    | `frozen` | modes map (slow / exhausted / frozen) |
    | `rates` | rates of progress |
    | `rhs`, `splitrhs` | source term, and its $\mathcal{S}\,\mathbf{r}$ reconstruction |
    | `species_type` | species map (fast / slow) |

    Every plot requires the tags `time` and `state` (state vector $[\,Y_1,\dots,Y_{N_s},T\,]$). Only the indices whose absolute value exceeds `threshold` times the maximum are plotted, and only the species whose mass fraction exceeds `species_threshold`.

`plot_index_over_time(time, data, temp, threshold, names, output_path, ylabel, title=None, xlim=None, xlabel='time [s]', frozen=None)`
:   Helper that plots the significant columns of an index time series together with the temperature.

## Eigenvalues

`select_eval(evals, indexes)`
:   Given a time series of eigenvalues [N$_t \times$ N$_v$] and an index per time step (e.g. $M$), returns the selected eigenvalue at each time step, e.g. $\lambda_{M+1}$.

`reorder_evals(ev, times)`
:   Reorders a time series of eigenvalues so that each column varies continuously in time.

## Mechanisms

`find_duplicate_reactions(R)`
:   Returns the groups of indices of duplicate reactions in the list of Cantera reactions `R`.
