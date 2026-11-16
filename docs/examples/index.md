# Examples

Several examples are available in the `examples` folder of the repository to exploit the functionalities related to:

- exhausted modes (M) and the fast/active/slow subspaces;
- tangential stretching rate (TSR);
- CSP and TSR indices (importance indices, amplitude and timescale participation indices, TSR amplitude/timescale participation indices);
- CSP solver and G-Scheme;
- CSP simplify.

| Folder | Script | Content |
|---|---|---|
| `cspAnalysis` | `exhaustedModes.py` | H$_2$/air ignition: eigenvalues, M at three tolerance pairs, amplitudes |
| | `subspaces.py` | CH$_4$/air ignition: fast/active/slow subspaces (`calc_subspaces`) |
| | `indices.py` | Importance, participation and TSR indices at a single state |
| | `indices-integral.py` | Time-integrated importance indices (`index_norm = 'none'`) |
| | `pointers.py` | CSP pointers |
| | `JacobianSIM.py` | Eigenvalues of the SIM-constrained Jacobian (`jacSIM`) |
| | `full_CSP_analysis.py` | Complete analysis stored in a `TaggedDatabase` and plotted with `utils.save_CSP_plots` |
| `dataImportExport` | `export_import.py` | Writing a trajectory to file, reading it back and analysing it |
| `cspSolver` | `cspsolver_const_p.py`, `cspsolver_const_v.py` | CSP solver at constant pressure and constant volume |
| | `gscheme_const_p.py`, `gscheme_const_p_reuse.py` | G-Scheme, without and with basis reuse |
| `cspSimplify` | `simp.py` | Skeletal mechanisms of GRI-3.0 for CH$_4$/air ignition |
| | `simp_TSR.py` | The same, with the TSR target set, over several conditions |
| `tsrAnalysis` | `TSR.py` | Chemical TSR during H$_2$/air ignition |
| | `TSR_ext.py` | Extended TSR of a flamelet dataset |

The reader is referred to [[1]](../references.md#Valorani2020a)&ndash;[[4]](../references.md#Valorani2017), [[6]](../references.md#gscheme2009) for the theoretical details of the methods employed.

!!! tip "Analysing a reactor that shares the `CanteraCSP` object"
    In the examples, the `CanteraCSP` object is passed to a Cantera reactor, and the CSP quantities are queried on the same object while the reactor advances. With Cantera > 3.2, reactors clone the phase by default: create the reactor with `utils.shared_reactor(ct.IdealGasConstPressureReactor, gas)`, or set the state explicitly on `gas` before each CSP query (e.g. `gas.TPY = r.phase.TPY`; `r.thermo` before Cantera 3.2). Setting the state also guarantees that the cached CSP kernel is updated (see `CanteraCSP` in [`Functions.py`](../api/functions.md)).
