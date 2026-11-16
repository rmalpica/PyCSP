"""Shared pytest fixtures for the PyCSP test-suite."""
import os

import matplotlib

matplotlib.use("Agg")

import numpy as np
import cantera as ct
import pytest

import PyCSP.Functions as csp
try:
    from PyCSP.utils import shared_reactor
except ImportError:   # allows running the suite against pre-1.5 code
    def shared_reactor(reactor_class, gas):
        try:
            return reactor_class(gas, clone=False)
        except TypeError:
            return reactor_class(gas)

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
H2_MECH = os.path.join(TESTS_DIR, "hydrogen.yaml")


def h2_trajectory(problemtype, tend=2.0e-3):
    """Integrate a stoichiometric H2/air batch reactor (1000 K, 1 atm) and
    return a list of (T, P, rho, Y) tuples sampled at every solver step."""
    gas = ct.Solution(H2_MECH)
    gas.TP = 1000.0, ct.one_atm
    gas.set_equivalence_ratio(1.0, "H2", "O2:1, N2:3.76")
    if problemtype == "const_p":
        r = shared_reactor(ct.IdealGasConstPressureReactor, gas)
    else:
        r = shared_reactor(ct.IdealGasReactor, gas)
    net = ct.ReactorNet([r])
    out = []
    while net.time < tend:
        net.step()
        out.append((gas.T, gas.P, gas.density, gas.Y.copy()))
    return out


def pick_states(traj):
    """Induction, ignition and post-ignition states of a trajectory."""
    T = np.array([s[0] for s in traj])
    iign = int(np.argmax(T > 0.5 * (T[0] + T[-1])))
    idx = [2, iign // 2, iign - 2, iign, iign + 5, len(T) - 1]
    return [traj[i] for i in idx]


def make_gas(problemtype, state, jacobiantype="full"):
    """CanteraCSP object set to `state` with the given problem type."""
    T, P, rho, Y = state
    gas = csp.CanteraCSP(H2_MECH)
    if problemtype == "const_p":
        gas.TPY = T, P, Y
        gas.constP = P
    else:
        gas.TDY = T, rho, Y
        gas.constRho = rho
    gas.jacobiantype = jacobiantype
    return gas


@pytest.fixture(scope="session")
def h2_states():
    return {pt: pick_states(h2_trajectory(pt)) for pt in ("const_p", "const_v")}
