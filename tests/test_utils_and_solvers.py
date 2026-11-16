"""Regression tests for utils, Simplify and the CSP integrators."""
import numpy as np
import cantera as ct
import pytest

import PyCSP.Functions as csp
import PyCSP.utils as utils
import PyCSP.Solver as cspS
import PyCSP.GScheme as gsc
import PyCSP.Simplify as simp
from conftest import H2_MECH, h2_trajectory, make_gas


def test_integrate_batch_constV_uses_mixture_density():
    gas = csp.CanteraCSP(H2_MECH)
    gas.TPX = 1000.0, ct.one_atm, "N2:1"          # previous composition, different density
    utils.integrate_batch_constV(gas, 1000.0, ct.one_atm, 1.0, "H2", "O2:1, N2:3.76", 1e-6)
    ref = ct.Solution(H2_MECH)
    ref.TP = 1000.0, ct.one_atm
    ref.set_equivalence_ratio(1.0, "H2", "O2:1, N2:3.76")
    assert gas.constRho == pytest.approx(ref.density, rel=1e-12)


def test_integrate_batch_constP_follows_state():
    gas = csp.CanteraCSP(H2_MECH)
    states = utils.integrate_batch_constP(gas, 1000.0, ct.one_atm, 1.0, "H2", "O2:1, N2:3.76", 1e-3)
    assert states.T[-1] > 2000.0                  # ignited
    assert gas.T == pytest.approx(states.T[-1])   # gas follows the reactor


def _small_db(gas, traj):
    db = utils.TaggedDatabase()
    for T, P, rho, Y in traj:
        gas.TPY = T, P, Y
        gas.constP = P
        stateYT = np.append(Y, T)
        lam, R, L, f = gas.get_kernel()
        api, tpi, ifast, islow, stype = gas.calc_CSPindices(API=True, Impo=True, species_type=True)
        db.store(time=T, state=stateYT, evals=lam, fvec=f, M=gas.calc_exhausted_modes(),
                 frozen=csp.detectFrozenModes(gas.nv - gas.n_elements, stateYT, R, L,
                                              gas.generalized_Stoich_matrix, gas.R_vector, 1e-4, 1e-12),
                 tsr=gas.calc_TSR(), API=api, Ifast=ifast, Islow=islow, species_type=stype,
                 Pointers=csp.CSP_pointers(R, L), tsrApi=gas.calc_TSRindices(type="amplitude"),
                 rates=gas.R_vector, rhs=gas.source,
                 splitrhs=gas.generalized_Stoich_matrix @ gas.R_vector)
    db.to_arrays()
    return db


def test_save_CSP_plots_short_series(tmp_path):
    """Few points (markevery would be 0) and the rates/rhs sections."""
    gas = csp.CanteraCSP(H2_MECH)
    traj = h2_trajectory("const_p", 1e-3)[::40][:10]
    db = _small_db(gas, traj)
    utils.save_CSP_plots(db, gas, output_folder=str(tmp_path), threshold=0.2)
    for name in ("eigenvalues.png", "state_evolution.png", "modes_map.png",
                 "rate_of_progress.png", "rhs.png"):
        assert (tmp_path / name).exists(), name


def test_get_tsrtarget_threshold_one():
    gas = csp.CanteraCSP(H2_MECH)
    traj = h2_trajectory("const_p", 1e-3)[::50]
    dataset = np.array([np.hstack((Y, T, P)) for T, P, rho, Y in traj])
    s = simp.CSPsimplify(gas, dataset)
    api = np.zeros(2 * gas.n_reactions)
    api[[0, 3, 5]] = [0.5, 0.3, 0.2]
    # 0.99: reactions 0, 3 and 5 are needed to exceed the threshold
    names = lambda k: set(gas.reaction(k).reactants) | set(gas.reaction(k).products)
    assert s.get_tsrtarget(api, 0.99) == names(0) | names(3) | names(5)
    # 1.0 (pre-1.5 raised IndexError): cumulative sum never exceeds 1, all reactions kept
    every = set().union(*(names(k) for k in range(gas.n_reactions)))
    assert s.get_tsrtarget(api, 1.0) == every


@pytest.mark.parametrize("cls,kw", [(cspS.CSPsolver, "cspRtolTail"), (gsc.GScheme, "cspRtol")])
def test_set_integrator_unknown_keyword_raises(cls, kw):
    gas = make_gas("const_p", h2_trajectory("const_p", 1e-4)[-1])
    with pytest.raises(ValueError):
        cls(gas).set_integrator(**{kw: 1e-3})


@pytest.mark.parametrize("cls", [cspS.CSPsolver, gsc.GScheme])
def test_profiling_before_integration(cls, capsys):
    gas = make_gas("const_p", h2_trajectory("const_p", 1e-4)[-1])
    s = cls(gas)
    # counters exist only after the first profiled call: emulate an idle solver
    for name in ("integrate", "CSPbasis", "calcM", "calcH", "RK4", "Qs"):
        setattr(s, name + "_time", 0.0)
        setattr(s, name + "_n", 0)
    s.profiling()
    assert "Total" in capsys.readouterr().out


def test_simplify_norm_cut_scalar_only():
    gas = csp.CanteraCSP(H2_MECH)
    traj = h2_trajectory("const_p", 1e-3)[::100]
    dataset = np.array([np.hstack((Y, T, P)) for T, P, rho, Y in traj])
    s = simp.CSPsimplify(gas, dataset)
    s.norm_cut = 1e-3
    with pytest.raises(ValueError):
        s.norm_cut = np.ones(gas.n_species + 1)


def test_kernel_follows_shared_reactor():
    """Pattern used by the examples: a reactor sharing the CanteraCSP object and a
    SolutionArray built on it. Appending the state to the SolutionArray sets the
    state of the object, so that the cached kernel is recomputed at every step."""
    gas = csp.CanteraCSP(H2_MECH)
    gas.TP = 1000.0, ct.one_atm
    gas.set_equivalence_ratio(1.0, "H2", "O2:1, N2:3.76")
    gas.constP = ct.one_atm
    r = utils.shared_reactor(ct.IdealGasConstPressureReactor, gas)
    net = ct.ReactorNet([r])
    states = ct.SolutionArray(gas, extra=["t"])
    nsteps = 0
    while net.time < 3e-4:
        net.step()
        nsteps += 1
        states.append(gas.state, t=net.time)
        lam = gas.get_kernel()[0]
    assert gas.nUpdates == nsteps
    ref = csp.CanteraCSP(H2_MECH)
    ref.TPY = gas.TPY
    ref.constP = ct.one_atm
    assert np.allclose(lam, ref.get_kernel()[0])
