# -*- coding: utf-8 -*-
"""
Constant-volume reactor: the chemical source term must equal S.r, and the
temperature equation must match the constant-volume energy balance.

@author: Riccardo Malpica Galassi, Sapienza University, Roma, Italy
Run with pytest, or as a script (python test_rhs_constV.py) to also plot.
"""
import numpy as np
import cantera as ct

from test_rhs import compute
from conftest import make_gas


def test_rhs_equals_Sr_constV():
    traj, RHS, splitRHS = compute("const_v", 1e-3)
    scale = np.abs(RHS).max(axis=0) + 1e-300
    assert np.max(np.abs(RHS - splitRHS) / scale) < 1e-8


def test_temperature_equation_constV():
    """dT/dt = -sum_k u_k wdot_k / (rho cv) at constant volume."""
    traj, RHS, _ = compute("const_v", 1e-3)
    for st, rhs in zip(traj[::20], RHS[::20]):
        gas = make_gas("const_v", st)
        U = gas.standard_int_energies_RT * ct.gas_constant * gas.T
        ref = -np.dot(U, gas.net_production_rates) / (gas.density * gas.cv_mass)
        assert rhs[-1] == np.float64(rhs[-1]) and abs(rhs[-1] - ref) <= 1e-10 * max(abs(ref), 1.0)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    traj, RHS, splitRHS = compute("const_v", 1e-3)
    T = [s[0] for s in traj]
    plt.plot(T, RHS[:, -1], "k", label="rhs")
    plt.plot(T, splitRHS[:, -1], "r--", label="S.r")
    plt.xlabel("Temperature (K)"); plt.ylabel("rhs[T]"); plt.legend(); plt.show()
