# CSP solver and G-Scheme

PyCSP includes two explicit integrators for the stiff chemical system $d\mathbf{y}/dt = \mathbf{g}(\mathbf{y})$. They use the CSP decomposition to remove the fast timescales from the integration:

- the [CSP solver](../api/solver.md) integrates all the non-exhausted modes. Works in physical coordinates;
- the [G-Scheme](../api/gscheme.md) [[6]](../references.md#gscheme2009) integrates only the active modes. Works in intrinsic coordinates.

In both cases the time step is set by the fastest retained mode rather than by the fastest mode of the system. This page describes the algorithms as implemented. The notation (basis vectors $\mathbf{a}_r$, $\mathbf{b}^r$, amplitudes $f^r$, eigenvalues $\lambda_r$, timescales $\tau_r$) is that of the [CSP and TSR analysis](analysis.md) page, with modes numbered $r = 1, \dots, N_v$ in order of decreasing $|\lambda_r|$; only the real part of $\lambda_r$ enters the corrections below.

## Subspaces

Each solver step partitions the modes into:

| Subspace | Modes | Treatment |
|---|---|---|
| fast (exhausted) | $r = 1,\dots,M$ | algebraic constraint $f^r \approx 0$, enforced by a correction |
| active | $r = M+1,\dots,H$ | integrated in time |
| slow (G-Scheme only) | $r = H+1,\dots,N_v-N_{el}$ | correction over the time step |
| conserved | $r = N_v-N_{el}+1,\dots,N_v$ | not evolving |

$M$ (called Tail, $T$, in the G-Scheme) and $H$ (Head) are computed with the criteria of the [analysis](analysis.md#exhausted-modes) page. The CSP solver has no slow subspace: all the non-exhausted modes are integrated.

## Time step

The time step lies between the timescale of the fastest retained mode and that of the fastest exhausted mode:

\[
\Delta t = \tau_M + \gamma\,(\tau_{M+1} - \tau_M) \quad (M > 0), \qquad \Delta t = \gamma\,\tau_1 \quad (M = 0),
\]

with $\gamma$ = `factor` (default 0.2). The step may grow by at most 50% with respect to the previous one.

## Radical (tail) correction

The exhausted modes have decayed, but their residual amplitude would drive the state away from the slow manifold. The correction removes the displacement that these modes would produce if they relaxed exponentially over the timescale of the first non-exhausted mode, $\tau_{M+1} = 1/|\mathrm{Re}\,\lambda_{M+1}|$:

\[
\mathbf{y} \leftarrow \mathbf{y} - \sum_{r=1}^{M} \mathbf{a}_r\, f^r\, \frac{1 - e^{\lambda_r \tau_{M+1}}}{\lambda_r} .
\]

For a fast dissipative mode, $\lambda_r \tau_{M+1} \ll -1$, so the correction reduces to the classical radical correction $\mathbf{y} \leftarrow \mathbf{y} - \sum_{r} \mathbf{a}_r f^r/\lambda_r$.

## CSP solver

One step of `CSPsolver.integrate()`, from $\mathbf{y}_n$:

1. **Basis.** Compute the CSP basis at $\mathbf{y}_n$.
2. **Radical correction.** Apply the radical correction with the current $M$, giving $\mathbf{y}^\ast$.
3. **Exhausted modes.** Update $M$ at $\mathbf{y}^\ast$, with the basis of step 1 and the tolerances `csprtol`, `cspatol`, and compute $\Delta t$.
4. **Integration.** Integrate the projected system with a fourth-order Runge&ndash;Kutta step:

    \[
    \frac{d\mathbf{y}}{dt} = \mathbf{Q}_s\, \mathbf{g}(\mathbf{y}), \qquad \mathbf{Q}_s = \mathbf{I} - \sum_{r=1}^{M} \mathbf{a}_r \mathbf{b}^r .
    \]

    The projector $\mathbf{Q}_s$ removes the fast components of the source term.

5. **Second correction.** Apply the radical correction again, with the amplitudes evaluated at the new state, giving $\mathbf{y}_{n+1}$; $t_{n+1} = t_n + \Delta t$.

## G-Scheme

One step of `GScheme.integrate()`, from $\mathbf{y}_n$:

1. **Basis.** Compute the CSP basis at $\mathbf{y}_n$, or reuse the previous one (see [Basis reuse](#basis-reuse)).
2. **Tail correction.** Apply the tail correction (the radical correction above) with the current $T$.
3. **Tail and time step.** Update $T$ with `csprtolT`, `cspatolT`, and compute $\Delta t$.
4. **Head.** Update $H$ with `csprtolH`, `cspatolH`, over the horizon $\Delta t$.
5. **Active integration.** Integrate the active dynamics only, in the coordinates $\mathbf{\xi}$ of the active subspace:

    \[
    \mathbf{y}(t) = \mathbf{y}_n + \sum_{r=T+1}^{H} \mathbf{a}_r\, \xi^r(t), \qquad
    \frac{d\xi^r}{dt} = \mathbf{b}^r \cdot \mathbf{g}(\mathbf{y}), \qquad \mathbf{\xi}(t_n) = 0,
    \]

    with a fourth-order Runge&ndash;Kutta step. The dimension of this system is $H - T$, usually much smaller than $N_v$.

6. **Head correction.** Account for the slow modes over the step, with their amplitudes evaluated at the new state. Each slow mode contributes $\mathbf{a}_r f^r (e^{\lambda_r \Delta t} - 1)/\lambda_r$, expanded for small $\lambda_r \Delta t$:

    \[
    \mathbf{y} \leftarrow \mathbf{y} + \sum_{r=H+1}^{N_v - N_{el}} \mathbf{a}_r\, f^r\, \Delta t \left(1 + \tfrac{1}{2}\,\lambda_r \Delta t\right) .
    \]

7. **Tail correction.** Apply the tail correction again, giving $\mathbf{y}_{n+1}$; $t_{n+1} = t_n + \Delta t$.

### Basis reuse

The basis computation (Jacobian and eigen-decomposition) is the most expensive part of a step. With `reuse = True`, the basis of the previous step is kept, and only the amplitudes are recomputed, as long as the Jacobian changes little. The change is measured on the inexpensive approximation of the Jacobian diagonal, $\mathbf{d}$ (attribute `jacobian_diagonal`):

\[
\nu = \left\| \left( \frac{d_j^{\,new} - d_j^{\,old}}{d_j^{\,new}} \right)_{j=1,\dots,N_s} \right\|_2 ,
\]

where entries with $|d_j^{\,new}| \le 10^{-20}$ are skipped. The basis is recomputed during the first three steps, when $\nu$ exceeds `reuseThr`, or after 250 consecutive reuses. The attributes `normJac` ($\nu$), `basisStatus`, `skipBasis` and `updateBasis` track the reuse.
