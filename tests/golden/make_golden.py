"""Generate the golden-master reference data for tests/test_golden.py.

Run ONLY on purpose, after a deliberate and reviewed change of results:

    python tests/golden/make_golden.py

The file stores the input states (so the tests do not depend on the ODE
integrator) together with the reference outputs of PyCSP 1.5.0.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))   # tests/ (for conftest helpers)

import cantera as ct  # noqa: E402
import PyCSP  # noqa: E402
import PyCSP.Functions as csp  # noqa: E402
import PyCSP.Solver as cspS  # noqa: E402
import PyCSP.GScheme as gsc  # noqa: E402
import PyCSP.Simplify as simp  # noqa: E402
from conftest import H2_MECH, h2_trajectory  # noqa: E402

GOLDEN = os.path.join(HERE, "golden_h2.npz")
PTYPES = ("const_p", "const_v")
JTYPES = ("full", "kinetic", "constrained")


def select_states(traj, n=10):
    """n states spread over induction, ignition and relaxation (by temperature)."""
    T = np.array([s[0] for s in traj])
    targets = np.linspace(T[0], T[-1], n)
    idx = sorted({int(np.argmin(np.abs(T - t))) for t in targets} | {1, len(T) - 1})
    return [traj[i] for i in idx]


def set_state(gas, ptype, T, P, rho, Y, jtype):
    if ptype == "const_p":
        gas.TPY = T, P, Y
        gas.constP = P
    else:
        gas.TDY = T, rho, Y
        gas.constRho = rho
    gas.jacobiantype = jtype


def kernel_outputs(gas, jtype):
    """Reference quantities that are invariant to eigenvector sign/scaling."""
    out = {}
    evals, A, B, f = gas.get_kernel()
    M, H = gas.calc_subspaces()
    out["evals"] = evals
    out["M"] = np.array(gas.calc_exhausted_modes())
    out["H"] = np.array(H)
    out["TSR"] = np.array(gas.calc_TSR())
    out["Qfast"] = A[:M].T @ B[:M]          # fast-subspace projector
    gas.index_norm = "none"                  # raw contributions (no normalization noise)
    api, tpi, ifast, islow, stype = gas.calc_CSPindices(
        API=True, Impo=True, species_type=True, TPI=(jtype != "constrained"))
    out["API"] = api
    out["Ifast"], out["Islow"] = ifast, islow
    if tpi is not None:
        out["TPI"] = tpi
    gas.index_norm = "abs"
    out["species_type"] = np.array([str(s) for s in stype])
    out["TSRAPI"] = gas.calc_TSRindices(type="amplitude")
    if jtype != "constrained":
        out["TSRTPI"] = gas.calc_TSRindices(type="timescale")
    return out


def solver_run(kind, ptype):
    gas = csp.CanteraCSP(H2_MECH)
    gas.TP = 1000.0, ct.one_atm
    gas.set_equivalence_ratio(1.0, "H2", "O2:1, N2:3.76")
    if ptype == "const_p":
        gas.constP = ct.one_atm
    else:
        gas.constRho = gas.density
    y0 = np.hstack((gas.Y, gas.T))
    if kind == "CSPsolver":
        s = cspS.CSPsolver(gas)
        s.set_integrator(cspRtol=1e-3, cspAtol=1e-9, factor=0.2, jacobiantype="full")
    else:
        s = gsc.GScheme(gas)
        s.set_integrator(cspRtolTail=1e-3, cspAtolTail=1e-9, cspRtolHead=1e-4,
                         cspAtolHead=1e-10, factor=0.25, jacobiantype="full")
    s.set_initial_value(y0, 0.0)
    nsteps = 0
    while s.t < 1.0e-3:
        s.integrate()
        nsteps += 1
    return {"y": np.array(s.y), "t": np.array(s.t), "nsteps": np.array(nsteps)}


def simplify_dataset():
    """Dataset rows [Y_1..Y_Ns, T, P] from an H2/air constant-pressure ignition."""
    traj = h2_trajectory("const_p", 1.0e-2)[::5]
    return np.array([np.hstack((Y, T, P)) for T, P, rho, Y in traj])


def simplify_run(dataset):
    gas = csp.CanteraCSP(H2_MECH)
    s = simp.CSPsimplify(gas, dataset)
    s.targetset = {"H2", "O2", "H2O"}
    s.process_dataset()
    out = {}
    for thr in (0.05, 0.2, 0.5):
        species, reactions = s.simplify_mechanism(thr)
        out["simplify_%g_species" % thr] = np.array(sorted(sp.name for sp in species))
        out["simplify_%g_nreac" % thr] = np.array(len(reactions))
    return out


def main():
    data = {"pycsp_version": np.array(PyCSP.__version__),
            "cantera_version": np.array(ct.__version__)}
    for ptype in PTYPES:
        states = select_states(h2_trajectory(ptype, 1.0e-2))
        data[f"{ptype}/T"] = np.array([s[0] for s in states])
        data[f"{ptype}/P"] = np.array([s[1] for s in states])
        data[f"{ptype}/rho"] = np.array([s[2] for s in states])
        data[f"{ptype}/Y"] = np.array([s[3] for s in states])
        for jtype in JTYPES:
            gas = csp.CanteraCSP(H2_MECH)
            for k, (T, P, rho, Y) in enumerate(states):
                set_state(gas, ptype, T, P, rho, Y, jtype)
                for name, val in kernel_outputs(gas, jtype).items():
                    data[f"{ptype}/{jtype}/{k}/{name}"] = val
        for kind in ("CSPsolver", "GScheme"):
            for name, val in solver_run(kind, ptype).items():
                data[f"{kind}/{ptype}/{name}"] = val
    data["simplify_dataset"] = simplify_dataset()
    data.update(simplify_run(data["simplify_dataset"]))
    np.savez_compressed(GOLDEN, **data)
    print("written", GOLDEN, "with", len(data), "entries")


if __name__ == "__main__":
    main()
