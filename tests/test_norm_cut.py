"""Optional norm_cut on CSP index normalization (student contribution, merged in 1.5.0)."""
import numpy as np
import pytest

from conftest import make_gas

JTYPES = ("full", "kinetic", "constrained")


def _indices(gas, **kw):
    tpi = gas.jacobiantype != "constrained"
    return gas.calc_CSPindices(API=True, Impo=True, TPI=tpi, **kw)


@pytest.mark.parametrize("jtype", JTYPES)
def test_norm_cut_default_and_zero_are_noop(h2_states, jtype):
    gas = make_gas("const_p", h2_states["const_p"][3], jtype)
    ref = _indices(gas)
    for cut in (None, 0.0):
        out = _indices(gas, norm_cut=cut)
        for a, b in zip(ref, out):
            if a is not None:
                assert np.array_equal(a, b)


@pytest.mark.parametrize("jtype", JTYPES)
def test_norm_cut_large_zeroes_rows(h2_states, jtype):
    gas = make_gas("const_p", h2_states["const_p"][3], jtype)
    out = _indices(gas, norm_cut=1e300)
    for a in out:
        if a is not None:
            assert not np.any(a)
    assert not np.any(gas.calc_TSRindices(norm_cut=1e300))


def test_norm_cut_per_row(h2_states):
    """Rows whose sum of |contributions| is below their own cut are zeroed,
    the others are normalized as usual."""
    gas = make_gas("const_p", h2_states["const_p"][3])
    gas.index_norm = "none"
    raw = gas.calc_CSPindices(API=True)[0]
    gas.index_norm = "abs"
    norms = np.abs(raw).sum(axis=1)
    cut = np.where(np.arange(len(norms)) % 2 == 0, 2 * norms + 1, 0.5 * norms)
    api = gas.calc_CSPindices(API=True, norm_cut=cut)[0]
    for i in range(len(norms)):
        if i % 2 == 0 or norms[i] == 0:
            assert not np.any(api[i])
        else:
            assert np.abs(api[i]).sum() == pytest.approx(1.0)


def test_norm_cut_wrong_length(h2_states):
    gas = make_gas("const_p", h2_states["const_p"][3])
    with pytest.raises(ValueError):
        gas.calc_CSPindices(API=True, norm_cut=np.ones(3))
