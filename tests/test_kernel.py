# -*- coding: utf-8 -*-
"""
CSP kernel along a constant-pressure H2/air ignition trajectory.

@author: Riccardo Malpica Galassi, Sapienza University, Roma, Italy
Run with pytest, or as a script (python test_kernel.py) to also plot the
eigenvalue evolution.
"""
import numpy as np

from conftest import h2_trajectory, make_gas


def compute(tend=1.0e-2):
    traj = h2_trajectory("const_p", tend)
    out = []
    for st in traj:
        gas = make_gas("const_p", st)
        lam, R, L, f = gas.get_kernel()
        out.append((st[0], lam, R, L, f, gas.n_elements, gas.rhs))
    return out


def test_kernel_properties():
    for T, lam, R, L, f, nel, rhs in compute()[::10]:
        nv = len(lam)
        # biorthonormal basis: B A = I (rows of R are the right vectors)
        assert np.allclose(L @ R.T, np.eye(nv), atol=1e-8)
        # eigenvalues sorted by decreasing magnitude, conserved modes zeroed
        assert np.all(np.diff(np.abs(lam)) <= 1e-12 * np.abs(lam).max())
        assert np.all(lam[nv - nel:] == 0)
        # amplitudes are non-negative and reconstruct the rhs: g = A f
        assert np.all(f >= 0)
        assert np.allclose(R.T @ f, rhs, rtol=1e-6, atol=1e-8 * np.abs(rhs).max())


def test_ignition_explosive_mode():
    """During induction the leading non-conserved eigenvalue has a positive real
    part (explosive mode); at equilibrium all non-conserved eigenvalues are negative."""
    res = compute()
    lam_ind = res[len(res) // 10][1]
    lam_eq = res[-1][1]
    assert np.any(lam_ind.real > 0)
    nel = res[-1][5]
    lam_eq = lam_eq[: len(lam_eq) - nel]
    # ignore round-off-sized eigenvalues (e.g. the quasi-conserved enthalpy mode)
    significant = np.abs(lam_eq) > 1e-8 * np.abs(lam_eq).max()
    assert np.all(lam_eq[significant].real < 0)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    res = compute()
    T = np.array([r[0] for r in res])
    evals = np.array([r[1] for r in res])
    logevals = np.clip(np.log10(1.0 + np.abs(evals.real)), 0, 100) * np.sign(evals.real)
    for idx in range(evals.shape[1]):
        plt.plot(T, logevals[:, idx], "k.", markersize=2)
    plt.xlabel("Temperature (K)"); plt.ylabel("evals"); plt.show()
