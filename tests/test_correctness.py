"""Correctness tests for the PyCSP kernel.

Each test checks a property that follows from the governing equations,
independently of the implementation under test:
  * Jacobian columns are derivatives at constant P (const_p) or rho (const_v);
  * the per-reaction Jacobian contributions sum to the full Jacobian;
  * elemental conservation annihilates the Jacobian rows;
  * the timescale participation indices decompose the eigenvalues;
  * the Cantera state is left untouched by Jacobian evaluations.
"""
import numpy as np
import pytest

import PyCSP.Functions as csp
from conftest import make_gas

SRO = np.sqrt(np.finfo(float).eps)
PTYPES = ("const_p", "const_v")
JTYPES = ("full", "kinetic", "constrained")


def _rhs(gas, ptype):
    return gas.rhs_const_p() if ptype == "const_p" else gas.rhs_const_v()


def _set(gas, ptype, T, y, P, rho):
    """Set unnormalized mass fractions y and temperature T, holding P or rho."""
    gas.set_unnormalized_mass_fractions(y)
    if ptype == "const_p":
        gas.TP = T, P
    else:
        gas.TD = T, rho


def reference_jacobian(gas, ptype):
    """Forward-difference Jacobian with the thermodynamic constraint (P or rho)
    imposed explicitly at every perturbed state. Same step sizes as PyCSP."""
    T, P, rho, y = gas.T, gas.P, gas.density, gas.Y.copy()
    ns = gas.n_species
    g0 = _rhs(gas, ptype)
    J = np.zeros((ns + 1, ns + 1))
    for i in range(ns):
        dy = np.zeros(ns)
        dy[i] = max(SRO * abs(y[i]), 1e-8)
        _set(gas, ptype, T, y + dy, P, rho)
        J[:, i] = (_rhs(gas, ptype) - g0) / dy[i]
    dT = max(SRO * abs(T), 1e-3)
    _set(gas, ptype, T + dT, y, P, rho)
    J[:, -1] = (_rhs(gas, ptype) - g0) / dT
    _set(gas, ptype, T, y, P, rho)
    return J


def _col_close(J, Jref, rtol):
    """Column-wise comparison scaled by each column's magnitude."""
    scale = np.maximum(np.abs(Jref).max(axis=0), 1e-300)
    return np.max(np.abs(J - Jref).max(axis=0) / scale) < rtol


# --------------------------------------------------------------------------
# Jacobian
# --------------------------------------------------------------------------
@pytest.mark.parametrize("ptype", PTYPES)
def test_jacobian_matches_constrained_reference(h2_states, ptype):
    for st in h2_states[ptype]:
        gas = make_gas(ptype, st)
        Jref = reference_jacobian(gas, ptype)
        J = gas.jacobian
        # round-off in the restored state gives <~1e-7; the rho-fixed (pre-1.5) Jacobian fails by O(1)
        assert _col_close(J, Jref, 1e-6), "jacobian not taken at constant %s" % (
            "P" if ptype == "const_p" else "rho")


def test_jacobian_const_p_chain_rule(h2_states):
    """Independent check: dg/dY_i|_{P,T} = dg/dY_i|_{rho,T} + dg/drho|_{Y,T} * drho/dY_i|_{P,T},
    with drho/dY_i|_{P,T} = -rho * Wmix / W_i for an ideal gas."""
    for st in h2_states["const_p"]:
        gas = make_gas("const_p", st)
        T, P, rho, y = gas.T, gas.P, gas.density, gas.Y.copy()
        W, Wmix, ns = gas.molecular_weights, gas.mean_molecular_weight, gas.n_species
        g0 = gas.rhs_const_p()
        # derivative wrt density at fixed Y,T
        drho = 1e-6 * rho
        gas.TD = T, rho + drho
        dgdrho = (gas.rhs_const_p() - g0) / drho
        gas.TD = T, rho
        Jrho = np.zeros((ns + 1, ns))
        for i in range(ns):
            dy = np.zeros(ns)
            dy[i] = max(SRO * abs(y[i]), 1e-8)
            gas.set_unnormalized_mass_fractions(y + dy)   # holds rho
            Jrho[:, i] = (gas.rhs_const_p() - g0) / dy[i]
            gas.TDY = T, rho, y
        gas.TPY = T, P, y
        Jchain = Jrho + np.outer(dgdrho, -rho * Wmix / W)
        J = gas.jacobian[:, :ns]
        assert _col_close(J, Jchain, 1e-3)   # FD noise ~1e-5; the rho-fixed Jacobian fails by O(1)


@pytest.mark.parametrize("ptype", PTYPES)
def test_jacobian_elemental_conservation(h2_states, ptype):
    """sum_i a_ei/W_i * dY_i/dt = 0 identically, hence c_e . J = 0."""
    for st in h2_states[ptype]:
        gas = make_gas(ptype, st)
        J = gas.jacobian
        ns = gas.n_species
        for e in range(gas.n_elements):
            c = np.array([gas.n_atoms(k, e) for k in range(ns)]) / gas.molecular_weights
            row = c @ J[:ns]
            scale = np.abs(c[:, None] * J[:ns]).sum(axis=0) + 1e-300
            assert np.max(np.abs(row) / scale) < 1e-5


@pytest.mark.parametrize("ptype", PTYPES)
def test_jac_contribution_sums_to_jacobian(h2_states, ptype):
    for st in h2_states[ptype]:
        gas = make_gas(ptype, st)
        J = gas.jacobian
        JacK = gas.jac_contribution()
        # columns where large reaction contributions cancel (e.g. inert N2 near
        # equilibrium) carry a finite-difference round-off ~ sqrt(eps)*sum_k|J_k|
        err = np.abs(JacK.sum(axis=0) - J).max(axis=0)
        scale = np.maximum(np.abs(J).max(axis=0), 1e-6 * np.abs(JacK).sum(axis=0).max(axis=0))
        assert np.max(err / scale) < 1e-1


@pytest.mark.parametrize("ptype", PTYPES)
def test_jacobian_diagonal_consistent_with_problem_type(h2_states, ptype):
    """jacobian_diagonal uses one simultaneous perturbation of all variables,
    taken with the right constraint and the right rhs."""
    for st in h2_states[ptype]:
        gas = make_gas(ptype, st)
        T, P, rho, y = gas.T, gas.P, gas.density, gas.Y.copy()
        g0 = _rhs(gas, ptype)
        dy = np.array([max(SRO * abs(v), 1e-8) for v in y])
        dT = max(SRO * abs(T), 1e-3)
        _set(gas, ptype, T + dT, y + dy, P, rho)
        dg = _rhs(gas, ptype) - g0
        _set(gas, ptype, T, y, P, rho)
        ref = np.append(dg[:-1] / dy, dg[-1] / dT)
        d = gas.jacobian_diagonal
        assert np.allclose(d, ref, rtol=1e-8, atol=1e-12 * np.abs(ref).max())


@pytest.mark.parametrize("ptype", PTYPES)
def test_constrained_jacobian(h2_states, ptype):
    """Species Jacobian on the constant-enthalpy (const_p) or constant-internal-
    energy (const_v) manifold: dg_Y/dY_j + dg_Y/dT * dT/dY_j, with
    dT/dY_j = -h_j/cp (const_p) or -u_j/cv (const_v), evaluated by finite
    differences of the rhs moving Y_j and T together."""
    import cantera as ct
    # first state skipped: radicals ~1e-20 << FD step floor, second-order terms dominate
    for st in h2_states[ptype][1:]:
        gas = make_gas(ptype, st, "constrained")
        T, P, rho, y = gas.T, gas.P, gas.density, gas.Y.copy()
        ns = gas.n_species
        if ptype == "const_p":
            e = gas.standard_enthalpies_RT * ct.gas_constant * T / gas.molecular_weights
            c = gas.cp_mass
        else:
            e = gas.standard_int_energies_RT * ct.gas_constant * T / gas.molecular_weights
            c = gas.cv_mass
        g0 = _rhs(gas, ptype)[:ns]
        Jref = np.zeros((ns, ns))
        for j in range(ns):
            dy = np.zeros(ns)
            dy[j] = max(SRO * abs(y[j]), 1e-8)
            _set(gas, ptype, T - dy[j] * e[j] / c, y + dy, P, rho)
            Jref[:, j] = (_rhs(gas, ptype)[:ns] - g0) / dy[j]
            _set(gas, ptype, T, y, P, rho)
        J = gas.jacKinetic() - gas.jacThermal()
        assert _col_close(J, Jref, 1e-4)


# --------------------------------------------------------------------------
# State restoration
# --------------------------------------------------------------------------
@pytest.mark.parametrize("ptype", PTYPES)
@pytest.mark.parametrize("jtype", JTYPES)
def test_state_restored(h2_states, ptype, jtype):
    st = h2_states[ptype][3]
    gas = make_gas(ptype, st, jtype)
    T0, P0, rho0, Y0 = gas.T, gas.P, gas.density, gas.Y.copy()
    gas.jacobian
    gas.jacobian_diagonal
    gas.jac_contribution()
    if jtype != "constrained":
        gas.calc_CSPindices(API=True, Impo=True, TPI=True)
    gas.calc_subspaces()
    assert gas.T == pytest.approx(T0, rel=1e-14)
    assert gas.P == pytest.approx(P0, rel=1e-12)
    assert gas.density == pytest.approx(rho0, rel=1e-12)
    assert np.sum(gas.Y) == pytest.approx(1.0, abs=1e-14)
    assert np.allclose(gas.Y, Y0, rtol=1e-12, atol=1e-300)


# --------------------------------------------------------------------------
# Subspaces / TSR
# --------------------------------------------------------------------------
@pytest.mark.parametrize("ptype", PTYPES)
@pytest.mark.parametrize("jtype", JTYPES)
def test_calc_subspaces_all_jacobian_types(h2_states, ptype, jtype):
    for st in h2_states[ptype]:
        gas = make_gas(ptype, st, jtype)
        M, H = gas.calc_subspaces()
        assert 0 <= M <= H <= gas.nv


def test_findTSR_ignores_zero_eigenvalue_modes():
    """A mode with zero eigenvalue carries no stretching: its weight must vanish."""
    evals = np.array([-100.0, -10.0, 0.0, 0.0])
    Revec = np.eye(4)
    f = np.array([0.0, 1.0, 1.0, 0.0])
    rhs = Revec.T @ f
    TSR, w = csp.findTSR(1, rhs, evals, Revec, f, 0)
    assert w[2] == 0.0
    assert TSR == pytest.approx(-10.0)


# --------------------------------------------------------------------------
# Timescale participation indices
# --------------------------------------------------------------------------
def _random_split(J, nparts, rng):
    parts = [rng.normal(size=J.shape) for _ in range(nparts - 1)]
    parts.append(J - sum(parts))
    return np.array(parts)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_TPI_decomposes_eigenvalues_including_complex_pairs(seed):
    """Unnormalized TPI rows sum to Re(lambda_n) (P21 Eq. 3.45)."""
    rng = np.random.default_rng(seed)
    # matrix with one complex pair and real modes
    D = np.zeros((4, 4))
    D[0:2, 0:2] = [[-50.0, 20.0], [-20.0, -50.0]]
    D[2, 2], D[3, 3] = -5.0, -0.5
    V = rng.normal(size=(4, 4))
    J = V @ D @ np.linalg.inv(V)
    evals, A, B = csp.eigsys(J)
    assert np.any(evals.imag != 0)
    nr = 3
    JacK = _random_split(J, 2 * nr, rng)
    TPI = csp.CSP_timescale_participation_indices(nr, JacK, evals, A, B, "none")
    assert np.allclose(TPI.sum(axis=1), evals.real, rtol=1e-8, atol=1e-8)


@pytest.mark.parametrize("ptype", PTYPES)
def test_TPI_decomposes_eigenvalues_mechanism(h2_states, ptype):
    for st in h2_states[ptype]:
        gas = make_gas(ptype, st)
        gas.index_norm = "none"
        evals = gas.get_kernel()[0]
        TPI = gas.calc_CSPindices(TPI=True)[1]
        nonzero = evals != 0
        scale = np.abs(evals).max()
        assert np.allclose(TPI.sum(axis=1)[nonzero], evals.real[nonzero], rtol=1e-4, atol=1e-6 * scale)


def test_TPI_constrained_not_supported(h2_states):
    gas = make_gas("const_p", h2_states["const_p"][3], "constrained")
    with pytest.raises(NotImplementedError):
        gas.calc_CSPindices(TPI=True)
    with pytest.raises(NotImplementedError):
        gas.calc_TSRindices(type="timescale")
