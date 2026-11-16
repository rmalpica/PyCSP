"""Golden-master regression tests.

Reference data: tests/golden/golden_h2.npz, produced by tests/golden/make_golden.py.
The stored input states are re-analysed and compared against the stored outputs.
Comparisons are invariant to the arbitrary parts of the CSP basis:
  * per-mode quantities of conserved modes (zero eigenvalue, degenerate
    eigenspace) are not compared;
  * amplitude participation indices are compared up to the sign of each mode
    (the sign of exhausted modes, whose amplitude is ~0, is arbitrary).
Tolerances absorb finite-difference Jacobian noise and BLAS/Cantera-version
differences; integer outputs (M, H, species classification) must match exactly.
"""
import os

import numpy as np
import pytest

import PyCSP.Functions as csp
from conftest import H2_MECH

HERE = os.path.dirname(os.path.abspath(__file__))
GOLDEN = os.path.join(HERE, "golden", "golden_h2.npz")
PTYPES = ("const_p", "const_v")
JTYPES = ("full", "kinetic", "constrained")

RTOL_EIG = 1e-5     # eigenvalues, TSR
RTOL_IDX = 1e-4     # indices and projectors, relative to the array's max


@pytest.fixture(scope="module")
def golden():
    with np.load(GOLDEN) as d:
        return dict(d)


def _close(a, b, rtol):
    scale = max(np.abs(b).max(), 1e-300)
    return np.abs(a - b).max() <= rtol * scale


def _close_up_to_row_sign(a, b, rtol, rows):
    scale = max(np.abs(b).max(), 1e-300)
    for i in rows:
        d = min(np.abs(a[i] - b[i]).max(), np.abs(a[i] + b[i]).max())
        if d > rtol * scale:
            return False
    return True


@pytest.mark.parametrize("ptype", PTYPES)
@pytest.mark.parametrize("jtype", JTYPES)
def test_kernel_golden(golden, ptype, jtype):
    from golden.make_golden import kernel_outputs, set_state
    g = golden
    gas = csp.CanteraCSP(H2_MECH)
    for k in range(len(g[f"{ptype}/T"])):
        set_state(gas, ptype, g[f"{ptype}/T"][k], g[f"{ptype}/P"][k], g[f"{ptype}/rho"][k],
                  g[f"{ptype}/Y"][k], jtype)
        out = kernel_outputs(gas, jtype)
        ref = {name: g[f"{ptype}/{jtype}/{k}/{name}"] for name in out}
        tag = f"{ptype}/{jtype}/state {k} (T={g[f'{ptype}/T'][k]:.0f} K)"

        assert int(out["M"]) == int(ref["M"]), tag + " M"
        assert int(out["H"]) == int(ref["H"]), tag + " H"
        assert np.array_equal(out["species_type"], ref["species_type"]), tag + " species_type"

        lam, lam_ref = out["evals"], ref["evals"]
        active = lam_ref != 0
        assert np.array_equal(lam == 0, ~active), tag + " conserved modes"
        assert np.allclose(lam[active], lam_ref[active], rtol=RTOL_EIG,
                           atol=RTOL_EIG * np.abs(lam_ref).max()), tag + " evals"
        assert out["TSR"] == pytest.approx(float(ref["TSR"]), rel=RTOL_EIG,
                                           abs=RTOL_EIG * np.abs(lam_ref).max()), tag + " TSR"

        assert _close(out["Qfast"], ref["Qfast"], RTOL_IDX), tag + " fast projector"
        assert _close(out["Ifast"], ref["Ifast"], RTOL_IDX), tag + " Ifast"
        assert _close(out["Islow"], ref["Islow"], RTOL_IDX), tag + " Islow"
        rows = np.flatnonzero(active)
        assert _close_up_to_row_sign(out["API"], ref["API"], RTOL_IDX, rows), tag + " API"
        if "TPI" in ref:
            assert _close(out["TPI"][rows], ref["TPI"][rows], RTOL_IDX), tag + " TPI"
            assert _close(out["TSRTPI"], ref["TSRTPI"], 1e-3), tag + " TSR-TPI"
        assert _close(out["TSRAPI"], ref["TSRAPI"], 1e-3), tag + " TSR-API"


@pytest.mark.parametrize("ptype", PTYPES)
@pytest.mark.parametrize("kind", ["CSPsolver", "GScheme"])
def test_solvers_golden(golden, ptype, kind):
    from golden.make_golden import solver_run
    out = solver_run(kind, ptype)
    ref = {k: golden[f"{kind}/{ptype}/{k}"] for k in out}
    # the adaptive step depends on the discrete M/H: allow a few steps of difference
    assert abs(int(out["nsteps"]) - int(ref["nsteps"])) <= max(3, int(0.02 * ref["nsteps"]))
    assert out["y"][-1] == pytest.approx(float(ref["y"][-1]), rel=1e-4)
    assert np.allclose(out["y"][:-1], ref["y"][:-1], rtol=1e-3, atol=1e-6)


def test_solvers_reach_equilibrium(golden):
    """Physics check independent of the golden data: after ignition both CSP
    integrators approach the adiabatic equilibrium temperature."""
    import cantera as ct
    for ptype, mode in (("const_p", "HP"), ("const_v", "UV")):
        g = ct.Solution(H2_MECH)
        g.TP = 1000.0, ct.one_atm
        g.set_equivalence_ratio(1.0, "H2", "O2:1, N2:3.76")
        g.equilibrate(mode)
        for kind in ("CSPsolver", "GScheme"):
            Tend = float(golden[f"{kind}/{ptype}/y"][-1])
            assert Tend == pytest.approx(g.T, rel=5e-3), (kind, ptype)


def test_simplify_golden(golden):
    from golden.make_golden import simplify_run
    out = simplify_run(golden["simplify_dataset"])   # stored dataset: no ODE integration
    for k, v in out.items():
        assert np.array_equal(v, golden[k]), k
