# `Simplify.py`

## class `CSPsimplify`

This class generates simplified (skeletal) kinetic mechanisms based on CSP [[3]](../references.md#MalpicaGalassi2018). The class takes as input an instance of the [`CanteraCSP`](functions.md) class and a dataset. The instance should be initialized with the detailed mechanism. The dataset must be a 2D array, each row containing species mass fractions (in the order of the detailed mechanism), temperature and pressure: `dataset = [Y_1, Y_2, ..., Y_N, T, P]`.

The algorithm is described in the [mechanism simplification](../theory/simplification.md) theory page.

```python
import PyCSP.Functions as csp
import PyCSP.Simplify as simp
dtl_mech = csp.CanteraCSP('gri30.yaml')
simplifier = simp.CSPsimplify(dtl_mech, dataset)
```

### Instance attributes

`csprtol`
:   Type (*float*).
    Set/get the CSP relative tolerance for the exhausted modes calculation. Defaults to `1.0e-2`.

`cspatol`
:   Type (*float*).
    Set/get the CSP absolute tolerance for the exhausted modes calculation. Defaults to `1.0e-8`.

`scaled`
:   Type (*boolean*).
    Set/get the *scaled* option for the importance indices. When active, indices are normalized with respect to their maximum value. Defaults to `True`.

`targetset`
:   Type (*set of strings*).
    Set/get the set of target species. Example: `{'CH4', 'H2O', 'CO2'}`.

`TSRtargetset`
:   Type (*boolean*).
    Set/get the *TSRtargetset* option for the target set. When active, the user-defined `targetset`, which may also be empty, is augmented with the local TSR-relevant species: each state will have its own set of targets, based on the TSR-API indices. Defaults to `False`.

`TSRtol`
:   Type (*float*).
    Set/get the TSR-API tolerance for including species in the TSR target set. A species is included if it is involved in a reaction belonging to the smallest subset of reactions whose cumulative participation to the TSR exceeds `TSRtol`$\,\cdot 100\%$. The higher the value of `TSRtol`, the more species will be included. Defaults to `0.5`.

`problemtype`
:   Type (*string*).
    Set/get the problem type, among `'constP'` and `'constRho'`. Defaults to `'constP'`.

`norm_cut`
:   Type (*float*).
    Set/get the optional cut on the normalization of the importance and TSR indices (see `calc_CSPindices` in [`Functions.py`](functions.md)). Only a scalar (or `None`) is accepted, because the same cut is applied to the importance indices (one row per variable) and to the TSR participation indices (one row per mode). Defaults to `None`. *New in v1.5.*

`verbose`
:   Type (*boolean*).
    Print progress information. Defaults to `False`.

`nv`, `ns`, `nr`
:   Type (*int, read-only*).
    Number of variables, species and reactions of the detailed mechanism.

### Class methods

`dataset_info()`
:   Outputs info on `dataset`, including an estimate of the processing time. Note that this method changes the state of the gas object.

`process_dataset()`
:   Computes the fast/slow importance indices and the species fast/slow classification for each state in `dataset`.

`simplify_mechanism(threshold)`
:   Returns the lists of retained species and reactions (Cantera `Species` and `Reaction` objects), based on the value of `threshold`, which must be between 0 and 1. A simplified mechanism is created as:

    ```python
    species, reactions = simplifier.simplify_mechanism(0.1)
    skeletal = csp.CanteraCSP(thermo='ideal-gas', kinetics='gas', species=species, reactions=reactions)
    skeletal.write_yaml('skeletal.yaml')
    ```

### Module functions

`merge_mechanisms(gasdtl, species1, species2)`
:   Returns the species and reactions of the mechanism that includes the species of two skeletal mechanisms derived from the same detailed mechanism `gasdtl`.

`find_reactions_given_species(gas, species)`
:   Returns the reactions of `gas` that involve only the given species.
