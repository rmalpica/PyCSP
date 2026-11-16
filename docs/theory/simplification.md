# Mechanism simplification

The class [`CSPsimplify`](../api/simplify.md) generates skeletal kinetic mechanisms from a detailed one, following the CSP-based algorithm of [[3]](../references.md#MalpicaGalassi2018). The algorithm uses the CSP [importance indices](analysis.md#importance-indices) and the [species classification](analysis.md#csp-pointers-and-species-classification), computed on a dataset of thermochemical states that is representative of the application, e.g. the ignition trajectory of a homogeneous reactor. This page describes the algorithm as implemented.

## Input data

The dataset is a 2D array, each row being a thermochemical state $[\,Y_1, \dots, Y_{N_s}, T, P\,]$ computed with the detailed mechanism. `process_dataset()` analyses each state $n$ of the dataset, with the full Jacobian and the CSP tolerances `csprtol`, `cspatol`. It stores:

- the absolute values of the fast and slow importance indices, $|I^{j}_{k,\text{fast}}|$ and $|I^{j}_{k,\text{slow}}|$, of each process $k$ (forward and backward reactions) to each variable $j$ (species and temperature);
- the classification of each species as fast, slow or trace;
- if the TSR target set is active, the absolute values of the TSR amplitude participation indices $|P^{\omega_\tau}_k|$.

With `problemtype = 'constP'` the indices are computed at the pressure of each state. With `'constRho'` they are computed at the density of each state.

## Active species of one state

Given a threshold $\theta \in [0, 1]$, the set of active species $\mathcal{A}_n$ of state $n$ is built iteratively. It starts from the target set $\mathcal{T}$ of species chosen by the user (`targetset`).

1. **Relevant reactions.** For each active species $j$, the processes that are relevant to its evolution are selected with the importance indices of the subspace that governs that species:
    - if $j$ is a slow species, the processes $k$ with $|I^{j}_{k,\text{slow}}| \ge \theta\, \max_{k'} |I^{j}_{k',\text{slow}}|$;
    - if $j$ is a fast species, the processes $k$ with $|I^{j}_{k,\text{fast}}| \ge \theta\, \max_{k'} |I^{j}_{k',\text{fast}}|$.

    The processes relevant to temperature are always selected, with its slow importance indices. With `scaled = False`, the criterion is $|I| \ge \theta$ instead of being relative to the maximum.

2. **New species.** All the species involved (as reactants or products) in the selected reactions become active.
3. **Trace species.** The species classified as trace are removed from the active set.
4. **Iteration.** Steps 1&ndash;3 are repeated until the active set no longer changes.

Finally, the target species are added back to $\mathcal{A}_n$, so that target species removed as trace species (e.g. inert species) are retained.

## Skeletal mechanism

The species of the skeletal mechanism are the union of the active sets over the dataset, $\mathcal{A} = \bigcup_n \mathcal{A}_n$. Its reactions are all the reactions of the detailed mechanism whose reactants and products all belong to $\mathcal{A}$.

`simplify_mechanism(threshold)` returns these species and reactions. A small threshold retains more reactions per species, and hence produces larger and more accurate mechanisms. Scanning the threshold between 0 and 1 produces a family of skeletal mechanisms of decreasing size (see the [example](../examples/csp-simplify.md)).

## TSR target set

The choice of the target set affects the result. With `TSRtargetset = True`, each state receives its own additional target species, chosen by the TSR [[3]](../references.md#MalpicaGalassi2018). These are the species involved in the processes that contribute the most to the TSR, which in turn identifies the most energetic timescale of the system:

1. The processes are sorted by decreasing $|P^{\omega_\tau}_k|$.
2. The smallest leading group whose cumulative participation exceeds `TSRtol` is selected. The higher `TSRtol`, the more processes and species are included.
3. The reactants and products of the selected reactions are added to the target set of that state.

The user-defined `targetset` may then be empty, or contain only the species that must be retained in any case, e.g. N$_2$.

## Normalization cut

With `norm_cut` (new in v1.5), the importance and TSR indices whose sum of absolute contributions is smaller than the cut are set to zero instead of being normalized. For example, the indices of a variable whose rate of change is negligible no longer select any reaction. See `calc_CSPindices` in [`Functions.py`](../api/functions.md).

## Merging mechanisms

`merge_mechanisms(gasdtl, species1, species2)` returns the skeletal mechanism containing the union of the species of two skeletal mechanisms derived from the same detailed mechanism, e.g. mechanisms generated for different operating conditions. The reactions are recovered from the detailed mechanism with the same criterion as above.
