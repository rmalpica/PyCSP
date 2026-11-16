# CSP and TSR analysis

This page collects the definitions of the quantities computed by the CSP and TSR analysis tools of PyCSP, written as they are implemented. The algorithms built on them are described in the [mechanism simplification](simplification.md) and [CSP solver and G-Scheme](solvers.md) pages. The reader is referred to [[1]](../references.md#Valorani2020a)&ndash;[[7]](../references.md#MalpicaGalassi2022) for the theoretical background of the methods.

## Governing equations

PyCSP considers a homogeneous, adiabatic reacting mixture of $N_s$ species and $N_r$ reversible reactions. The state vector collects the species mass fractions and the temperature,

\[
\mathbf{y} = [\,Y_1, \dots, Y_{N_s}, T\,]^T, \qquad \frac{d\mathbf{y}}{dt} = \mathbf{g}(\mathbf{y}),
\]

with $N_v = N_s + 1$ variables. Forward and backward directions of each reaction are treated as separate processes, so that there are $2N_r$ rates of progress $r^k$ [kmol/m$^3$/s] and the net molar production rate of species $i$ is $\dot{\omega}_i = \sum_{k=1}^{2N_r} \nu_{ik} r^k$. Here $\nu_{ik}$ is the net stoichiometric coefficient of species $i$ in process $k$. It is positive for products, and its sign is reversed for the backward direction.

### Constant pressure (`problemtype = 'const_p'`)

\[
g_i = \frac{W_i}{\rho}\,\dot{\omega}_i \quad (i = 1,\dots,N_s), \qquad
g_T = -\frac{1}{c_p} \sum_{i=1}^{N_s} h_i\, g_i ,
\]

where $W_i$ is the molecular weight of species $i$, $\rho$ the mixture density, $c_p$ the mixture specific heat at constant pressure [J/kg/K] and $h_i = H_i/W_i$ the specific enthalpy of species $i$ [J/kg] (with $H_i$ the molar enthalpy).

### Constant volume (`problemtype = 'const_v'`)

\[
g_i = \frac{W_i}{\rho}\,\dot{\omega}_i , \qquad
g_T = -\frac{1}{c_v} \sum_{i=1}^{N_s} u_i\, g_i
     = -\frac{1}{c_v} \sum_{i=1}^{N_s} h_i\, g_i + \frac{R\,T}{c_v} \sum_{i=1}^{N_s} \frac{g_i}{W_i},
\]

where $c_v$ is the mixture specific heat at constant volume, $u_i = h_i - RT/W_i$ the specific internal energy of species $i$ and $R$ the universal gas constant.

### Generalized stoichiometric matrix

The source term is written as the sum of the contributions of the $2N_r$ processes, $\mathbf{g} = \mathcal{S}\,\mathbf{r} = \sum_k \mathcal{S}_k\, r^k$, where $\mathcal{S}_k$ is the $k$-th column of the generalized stoichiometric matrix $\mathcal{S}$ [$N_v \times 2N_r$]:

\[
\mathcal{S}_{i,k} = \frac{W_i}{\rho}\,\nu_{ik} \qquad i = 1,\dots,N_s ,
\]

\[
\text{const. pressure:}\quad \mathcal{S}_{T,k} = -\frac{1}{\rho\, c_p} \sum_{i=1}^{N_s} \nu_{ik}\, H_i ,
\]

\[
\text{const. volume:}\quad \mathcal{S}_{T,k} = -\frac{1}{\rho\, c_v} \sum_{i=1}^{N_s} \nu_{ik}\, (H_i - R\,T) .
\]

## Jacobian

The Jacobian $\mathbf{J} = \partial \mathbf{g}/\partial \mathbf{y}$ is evaluated by forward finite differences. Each mass fraction $Y_j$ is perturbed without renormalizing the others, at constant temperature. Temperature is perturbed at constant composition. In both cases the thermodynamic constraint of the problem is held fixed:

\[
J_{ij} = \left.\frac{\partial g_i}{\partial Y_j}\right|_{T,\,P\ \text{or}\ \rho}, \qquad
J_{i,T} = \left.\frac{\partial g_i}{\partial T}\right|_{Y,\,P\ \text{or}\ \rho},
\]

with steps $\delta Y_j = \max(\sqrt{\epsilon}\,|Y_j|,\,10^{-8})$ and $\delta T = \max(\sqrt{\epsilon}\,T,\,10^{-3}\,\mathrm{K})$, $\epsilon$ being the machine precision.


Three Jacobian types are available (attribute `jacobiantype` of `CanteraCSP`):

`full`
:   The $N_v \times N_v$ Jacobian of $(Y_1,\dots,Y_{N_s},T)$.

`kinetic`
:   The $N_s \times N_s$ species block $\mathbf{J}_{YY}$, temperature being frozen.

`constrained`
:   The $N_s \times N_s$ Jacobian of the species equations on the manifold of constant enthalpy (constant pressure) or constant internal energy (constant volume), where temperature is a function of the composition. Differentiating $\sum_i Y_i e_i(T) = \text{const}$ gives $\partial T/\partial Y_j = -e_j/c$, hence

    \[
    \mathbf{J}^{c} = \mathbf{J}_{YY} - \mathbf{J}_{YT}\, \frac{\mathbf{e}^T}{c},
    \qquad e_j = h_j,\ c = c_p \ \ \text{(const. pressure)}, \qquad e_j = u_j,\ c = c_v \ \ \text{(const. volume)}.
    \]

    The correction term is returned by `jacThermal()`.

The contribution of process $k$ to the Jacobian is $\mathbf{J}_k = \partial(\mathcal{S}_k r^k)/\partial \mathbf{y}$, so that $\mathbf{J} = \sum_{k=1}^{2N_r} \mathbf{J}_k$ (method `jac_contribution()`, available for the `full` and `kinetic` types).

## CSP kernel

The CSP basis is the eigenvector basis of the Jacobian (zeroth-order CSP basis):

- **Eigenvalues.** The eigenvalues $\lambda_i$ are sorted by decreasing modulus. The last $N_{el}$ eigenvalues, with $N_{el}$ the number of chemical elements, correspond to the elemental conservation laws and are set exactly to zero.
- **Basis vectors.** The right eigenvectors $\mathbf{a}_i$ are the rows of `Revec` ($\mathbf{A}$). The dual basis vectors $\mathbf{b}^i$ are the rows of `Levec` ($\mathbf{B} = \mathbf{A}^{-T}$ in matrix form, such that $\mathbf{b}^i\cdot\mathbf{a}_j = \delta^i_j$).
- **Complex-conjugate pairs.** A complex-conjugate pair $\lambda = \alpha \pm i\beta$ with eigenvectors $\mathbf{v}, \bar{\mathbf{v}}$ is represented by the real vectors $\mathbf{a}_i = \mathrm{Re}\,\mathbf{v}$, $\mathbf{a}_{i+1} = \mathrm{Im}\,\mathbf{v}$.
- **Mode amplitudes.** $f^i = \mathbf{b}^i \cdot \mathbf{g}$, so that $\mathbf{g} = \sum_i \mathbf{a}_i f^i$. The sign of each pair $(\mathbf{a}_i, \mathbf{b}^i)$ is chosen such that $f^i \ge 0$.
- **Timescales.** $\tau_i = 1/|\lambda_i|$.

!!! note
    The sign of a mode whose amplitude is (numerically) zero, e.g. an exhausted mode, is arbitrary. Quantities that depend on the sign of $\mathbf{a}_i$ or $\mathbf{b}^i$ separately, such as the amplitude participation indices of that mode, may flip sign from one state to the next.

### Exhausted modes

The number $M$ of exhausted (fast) modes is the largest $m$ such that, for all $m' \le m$, the cumulative contribution of the first $m'$ modes over the timescale of the next mode is below the error tolerance:

\[
\left| \sum_{r=1}^{m'} \mathbf{a}_r\, f^r\, \frac{e^{\lambda_r \tau_{r+1}} - 1}{\lambda_r} \right|_j < \mathrm{rtol}\, |y_j| + \mathrm{atol} \qquad \forall\, j = 1,\dots,N_v ,
\]

where $\lambda_r$ denotes the real part of the eigenvalue. In other words, a mode is fast if it has relaxed close enough to its local quasi-equilibrium that its remaining drift, integrated over the first active timescale, no longer matters at the prescribed accuracy. $M$ never separates the two members of a complex-conjugate pair, and at least one non-conserved mode is left active. The tolerances are the attributes `rtol` and `atol` of `CanteraCSP` (or keyword arguments of the `calc_*` methods).

### Slow modes

The Head index $H$ separates the active modes $[M, H)$ from the slow ones $[H, N_v - N_{el})$, following the G-Scheme [[6]](../references.md#gscheme2009). Scanning the non-conserved modes backwards from the slowest one, a mode $j$ is slow if the cumulative second order contribution of the slower modes (slow/dormant modes are quasi-linear in the active timeframe) stays below the tolerance over the timescale $\Delta t$:

\[
\left| \sum_{r \ge j} \tfrac{1}{2}\,\Delta t^2\, |\lambda_r|\, \mathbf{a}_r\, f^r \right|_j < \mathrm{rtol}_H\, |y_j| + \mathrm{atol}_H .
\]

In `calc_subspaces()`, $\Delta t = \tau_{M+1}$; in the G-Scheme integrator, $\Delta t$ is the integration time step.

## Tangential Stretching Rate

The TSR [[4]](../references.md#Valorani2017) is a weighted average of the eigenvalues:

\[
\omega_\tau = \sum_{i} W^i\, \mathrm{sgn}(\mathrm{Re}\,\lambda_i)\, |\lambda_i|, \qquad
W^i = \frac{(f^i/\|\mathbf{g}\|)^2}{\sum_j (f^j/\|\mathbf{g}\|)^2} .
\]

- **Excluded modes.** The weights of the $M$ exhausted modes and of the conserved modes (zero eigenvalue) are set to zero, so that the sum runs over the active and slow modes only.
- **Complex pairs.** For a complex pair, both amplitudes are replaced by $\sqrt{(f^i)^2 + (f^{i+1})^2}$.
- **Extended TSR.** For reaction-diffusion-convection systems [[4]](../references.md#Valorani2017), $\mathbf{g}$ is replaced by $\mathbf{g} + \mathbf{L}$, where $\mathbf{L}$ is the sum of the transport terms (diffusion and/or convection, supplied by the user in the same units as $\mathbf{g}$). The amplitudes become $h^i = \mathbf{b}^i\cdot(\mathbf{g}+\mathbf{L})$, and $M$ is recomputed with these amplitudes ($M_{ext}$).

## CSP indices

All the indices are normalized, unless `index_norm = 'none'`, by the sum of the absolute values of the contributions, so that their absolute values sum to one. The optional `norm_cut` (new in v1.5) sets to zero, instead of normalizing, the indices whose sum of absolute contributions is smaller than the cut.

### Amplitude participation index (API)

The participation of process $k$ to the amplitude of mode $i$ [[1]](../references.md#Valorani2020a):

\[
P^i_k = \frac{\mathbf{b}^i\cdot\mathcal{S}_k\, r^k}{\sum_{k'=1}^{2N_r} |\mathbf{b}^i\cdot\mathcal{S}_{k'}\, r^{k'}|} .
\]

For an exhausted mode, the processes with large $|P^i_k|$ are the ones in equilibrium.

### Timescale participation index (TPI)

The participation of process $k$ to the eigenvalue (hence the timescale) of mode $i$ [[1]](../references.md#Valorani2020a). From $\lambda_i = \mathbf{b}^i\cdot\mathbf{J}\,\mathbf{a}_i = \sum_k c^k_i$, with $c^k_i = \mathbf{b}^i\cdot\mathbf{J}_k\,\mathbf{a}_i$:

\[
J^i_k = \frac{c^k_i}{\sum_{k'} |c^{k'}_i|} .
\]

For a complex-conjugate pair the index decomposes $\mathrm{Re}\,\lambda_i$, using $c^k_i = \tfrac12\left(\mathbf{b}^i\cdot\mathbf{J}_k\,\mathbf{a}_i + \mathbf{b}^{i+1}\cdot\mathbf{J}_k\,\mathbf{a}_{i+1}\right)$.

### Importance indices

The fast and slow importance indices of process $k$ to variable $j$ [[1]](../references.md#Valorani2020a) measure the contribution of process $k$ to the fast and slow projections of $\mathbf{g}$:

\[
I^{j}_{k,\text{fast}} = \frac{\left(\mathbf{Q}_{\text{fast}}\,\mathcal{S}_k\, r^k\right)_j}{\sum_{k'} \left|\left(\mathbf{Q}_{\text{fast}}\,\mathcal{S}_{k'}\, r^{k'}\right)_j\right|}, \qquad
\mathbf{Q}_{\text{fast}} = \sum_{r=1}^{M} \mathbf{a}_r \mathbf{b}^r ,
\]

and similarly for $I^{j}_{k,\text{slow}}$, with $\mathbf{Q}_{\text{slow}} = \sum_{r > M} \mathbf{a}_r \mathbf{b}^r$.

### TSR participation indices

The participation of mode $i$ to the TSR, as implemented, is

\[
P^{\omega_\tau}_i = \frac{W^i\,|\lambda_i|}{\sum_j W^j\,|\lambda_j|} ,
\]

and the participation of process $k$ to the TSR combines it with the API (TSR-API) or the TPI (TSR-TPI) of each mode [[4]](../references.md#Valorani2017):

\[
P^{\omega_\tau}_k = \sum_i |P^{\omega_\tau}_i|\, P^i_k \quad \text{(TSR-API)}, \qquad
\sum_i |P^{\omega_\tau}_i|\, J^i_k \quad \text{(TSR-TPI)},
\]

normalized by the sum of absolute values.

### CSP pointers and species classification

The CSP pointer of mode $m$ to variable $j$ is the diagonal element $(\mathbf{a}_m \mathbf{b}^m)_{jj}$, following Lam and Goussis. PyCSP classifies the species with the following steps:

1. Sum the pointers of the $M$ exhausted modes, $\hat{Q}_{jj} = \sum_{m=1}^{M} (\mathbf{a}_m \mathbf{b}^m)_{jj}$.
2. Label as **fast** the $M$ variables with the largest $\hat{Q}_{jj}$. Temperature is always labelled **slow**.
3. Label all other species **slow**.
4. Optionally (`classify_traces = True`), label as **trace** the species with $Y_j < 10^{-20}$ and non-positive source term.

## SIM-constrained Jacobian

`jacSIM()` returns the Jacobian of the slow dynamics constrained to the slow invariant manifold (SIM) defined by the $M$ exhausted modes, expressed in terms of a set of major variables. The theoretical details can be found in [[5]](../references.md#valorani2022spectral).
