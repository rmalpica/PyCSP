# -*- coding: utf-8 -*-
"""
Constant-pressure reactor: the chemical source term must equal S.r
(generalized stoichiometric matrix times rates-of-progress vector).

@author: Riccardo Malpica Galassi, Sapienza University, Roma, Italy
Run with pytest, or as a script (python test_rhs.py) to also plot the solution.
"""
import numpy as np
import pytest

from conftest import h2_trajectory, make_gas


def compute(problemtype="const_p", tend=1.5e-3):
    traj = h2_trajectory(problemtype, tend)
    RHS, splitRHS = [], []
    for st in traj:
        gas = make_gas(problemtype, st)
        RHS.append(gas.source)
        splitRHS.append(gas.generalized_Stoich_matrix @ gas.R_vector)
    return traj, np.array(RHS), np.array(splitRHS)


@pytest.mark.parametrize("problemtype", ["const_p"])
def test_rhs_equals_Sr(problemtype):
    traj, RHS, splitRHS = compute(problemtype)
    # source term and S.r agree to round-off (scaled by the largest term)
    scale = np.abs(RHS).max(axis=0) + 1e-300
    assert np.max(np.abs(RHS - splitRHS) / scale) < 1e-8


def test_rhs_matches_cantera_reactor():
    """dY/dt of PyCSP equals the Cantera net production rates (W_k wdot_k / rho)."""
    traj, RHS, _ = compute("const_p", 5e-4)
    for st, rhs in zip(traj[::20], RHS[::20]):
        gas = make_gas("const_p", st)
        ref = gas.net_production_rates * gas.molecular_weights / gas.density
        assert np.allclose(rhs[:-1], ref, rtol=1e-12, atol=1e-300)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    traj, RHS, splitRHS = compute("const_p")
    T = [s[0] for s in traj]
    plt.plot(T, RHS[:, -1], "k", label="rhs")
    plt.plot(T, splitRHS[:, -1], "r--", label="S.r")
    plt.xlabel("Temperature (K)"); plt.ylabel("rhs[T]"); plt.legend(); plt.show()
