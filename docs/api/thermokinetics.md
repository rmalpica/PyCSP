# `ThermoKinetics.py`

## class `CanteraThermoKinetics`

This class inherits attributes and methods from Cantera's `Solution` class and extends its capabilities by adding thermodynamics and kinetics-related class methods. Hence, the constructor for this class is equivalent to the constructor of Cantera's `Solution` class. The proper way to instantiate `CanteraThermoKinetics` objects is by using species and reactions defined in an input file, e.g. `hydrogen.yaml`:

```python
import PyCSP.ThermoKinetics as csp
gas = csp.CanteraThermoKinetics('hydrogen.yaml')
```

Different formulations exist for the majority of this class methods depending on whether the reactor is considered having a constant pressure or a constant volume (density). Hence, it is mandatory to choose between constant pressure and constant density after instantiating an object of the `CanteraThermoKinetics` class and before accessing its attributes. The choice is made by defining either the pressure value or the density value through the attributes `constP` or `constRho` (see below). A lack of this choice will raise an exception when accessing the class attributes having multiple formulations.

The equations implemented by this class are described in the [Theory](../theory/analysis.md) page.

### Instance attributes

`constP`
:   Type (*float*).
    Set/get the thermodynamic pressure [Pa] and activates the constant pressure version of all the class methods (`problemtype == 'const_p'`).

`constRho`
:   Type (*float*).
    Set/get the thermodynamic density [kg/m$^3$] and activates the constant volume version of all the class methods (`problemtype == 'const_v'`).

`problemtype`
:   Type (*string, read-only*).
    Get the active definition of `problemtype`, either `'const_p'` or `'const_v'`.

`nv`
:   Type (*int*).
    Set/get the number of variables. Defaults to `n_species` + 1. The only alternative value is `n_species`, e.g., to exclude temperature from the list of variables and get appropriately sized objects from the class methods. It is set automatically by the `jacobiantype` attribute of [`CanteraCSP`](functions.md).

`source`
:   Type ([N$_s$+1] *array of float, read-only*).
    Get the chemical source term $[\,g_1, \dots, g_{N_s}, g_T\,]$ for the species mass fractions and temperature, based on either the constant pressure or constant volume formulation.

`jacobian`
:   Type ([N$_s$+1 $\times$ N$_s$+1] *array of float, read-only*).
    Get the numerical Jacobian of the chemical source term, based on either the constant pressure or constant volume formulation.

`generalized_Stoich_matrix`
:   Type ([N$_v$ $\times$ 2N$_r$] *array of float, read-only*).
    Get the generalized stoichiometric matrix, based on either the constant pressure or constant volume formulation. Depending on the value of `nv`, the last row (temperature coefficients) is returned or not.

`R_vector`
:   Type ([2N$_r$] *array of float, read-only*).
    Get the rates of progress vector (forward rates followed by backward rates) [kmol/m$^3$/s].

`jacobian_diagonal`
:   Type ([N$_s$+1] *array of float, read-only*).
    Get an inexpensive approximation of the Jacobian diagonal, obtained with a single simultaneous perturbation of all the variables. Used by the basis-reuse heuristic of the [G-Scheme](gscheme.md).

### Class methods

`set_stateYT(state)`
:   Set the thermochemical state of the mixture. This is an alternative to separately setting Cantera's attributes `Y` and either `TP` or `TD`. Pressure and density are automatically set based on the values of `constP` or `constRho`.

    Mandatory positional arguments:

    - `state` (*array of float*). Array containing the mass fractions $Y_i$ of all the species and the temperature: $[\,Y_1, Y_2, \dots, Y_{N_s}, T\,]$. The order of the $Y_i$ values must comply with the species ordering set at the class instantiation. This can be retrieved with the method `species()`. The mass fractions are normalized by Cantera.

`stateYT()`
:   Returns the thermochemical state of the mixture $[\,Y_1, Y_2, \dots, Y_{N_s}, T\,]$.

`rhs_const_p()`
:   Returns the constant-pressure chemical source term of the mixture $[\,g_1, \dots, g_{N_s}, g_T\,]_P$:

    \[
    g_i = \frac{W_i}{\rho} \sum_{k=1}^{2 N_r} \nu_{ik}\, r^k \qquad i=1,\dots,N_s, \qquad
    g_T = - \frac{1}{c_p} \sum_{i=1}^{N_s} h_i\, g_i
    \]

    where $W_i$ is the molecular weight of species $i$, $\rho$ is the mixture density, $\nu_{ik}$ is the molar stoichiometric coefficient of species $i$ in reaction $k$, $r^k$ is the molar rate of progress of reaction $k$, $c_p$ is the mixture constant-pressure specific heat and $h_i$ is the specific (per unit mass) enthalpy of species $i$.

`rhs_const_v()`
:   Returns the constant-volume chemical source term of the mixture $[\,g_1, \dots, g_{N_s}, g_T\,]_v$:

    \[
    g_i = \frac{W_i}{\rho} \sum_{k=1}^{2 N_r} \nu_{ik}\, r^k \qquad i=1,\dots,N_s, \qquad
    g_T = - \frac{1}{c_v} \sum_{i=1}^{N_s} h_i\, g_i + \frac{R\, T}{c_v} \sum_{i=1}^{N_s} \frac{g_i}{W_i}
    \]

`generalized_Stoich_matrix_const_p()`
:   Returns the constant-pressure generalized stoichiometric matrix $\mathcal{S}$, defined as:

    \[
    \mathcal{S}_{i,k} = \frac{W_i}{\rho}\, \nu_{ik} \quad i=1,\dots,N_s, \qquad
    \mathcal{S}_{N_s+1,k} = - \frac{1}{\rho\, c_p} \sum_{i=1}^{N_s} W_i\, \nu_{ik}\, h_i \qquad k =1,\dots,2N_r
    \]

`generalized_Stoich_matrix_const_v()`
:   Returns the constant-volume generalized stoichiometric matrix $\mathcal{S}$, defined as:

    \[
    \mathcal{S}_{i,k} = \frac{W_i}{\rho}\, \nu_{ik} \quad i=1,\dots,N_s, \qquad
    \mathcal{S}_{N_s+1,k} = - \frac{1}{\rho\, c_v} \sum_{i=1}^{N_s} W_i\, \nu_{ik}\, h_i + \frac{R\, T}{\rho\, c_v} \sum_{i=1}^{N_s} \nu_{ik} \qquad k =1,\dots,2N_r
    \]

`Rates_vector()`
:   Returns the molar rates of progress $r^k$ vector of the mixture (forward rates followed by backward rates).

`jacobian_const_p()`
:   Returns the constant-pressure numerically-evaluated Jacobian of the chemical source term of the mixture:

    \[
    \mathbf{J} =
    \begin{bmatrix}
    \left. \frac{\partial g_1}{\partial Y_1}\right|_{P,T} & \dots & \left. \frac{\partial g_{1}}{\partial Y_{N_s}}\right|_{P,T} & \left. \frac{\partial g_1}{\partial T}\right|_{P,Y} \\
    \vdots & \ddots & & \vdots \\
    \left. \frac{\partial g_{T}}{\partial Y_1}\right|_{P,T} & \dots & \dots & \left. \frac{\partial g_T}{\partial T}\right|_{P,Y}
    \end{bmatrix}
    \]

    Mass fractions are perturbed one at a time, without renormalization. See [Theory](../theory/analysis.md#jacobian) for the finite-difference steps.

`jacobian_const_v()`
:   Returns the constant-volume numerically-evaluated Jacobian of the chemical source term of the mixture, with derivatives taken at constant density instead of constant pressure.

`jac_contribution()`
:   Returns a 2N$_r$-long array of [N$_v$ $\times$ N$_v$] Jacobian matrices, each one representing the contribution of reaction $k$ to the Jacobian:

    \[
    \mathbf{J}_k = \frac{\partial\, (\mathcal{S}_{k}\, r^k)}{\partial \mathbf{y}}, \qquad \sum_{k=1}^{2N_r} \mathbf{J}_k = \mathbf{J} .
    \]

    The derivatives are taken with the same constraint (pressure or density) as `jacobian`. With `nv = n_species`, the species block is returned.

`jacobian_diagonal_const_p()`, `jacobian_diagonal_const_v()`
:   Return the approximate Jacobian diagonal (see the attribute `jacobian_diagonal`) for the two formulations.

`jacKinetic()`
:   Returns the $N_s \times N_s$ Jacobian, excluding the temperature row/column.

`jacThermal()`
:   Returns the $N_s \times N_s$ correction $\mathbf{J}_{YT}\,\mathbf{e}^T/c$ such that `jacKinetic() - jacThermal()` is the Jacobian constrained to constant enthalpy (constant pressure, $e_j = h_j$, $c = c_p$) or constant internal energy (constant volume, $e_j = u_j$, $c = c_v$). See [Theory](../theory/analysis.md#jacobian).

`jacSIM(Jac, major_ind, Levec)`
:   Returns the Jacobian constrained to the SIM. Theoretical details can be found in [[5]](../references.md#valorani2022spectral).

    Mandatory positional arguments:

    - `Jac` ([N$_v \times$ N$_v$] *array of float*). A Jacobian matrix.
    - `major_ind` ([N$_{major}$] *array of integers*). A list of integers corresponding to the progressive index of the variables deemed as major.
    - `Levec` ([N$_v \times$ N$_v$] *array of float*). A left eigenvectors matrix.

`reaction_names()`
:   Returns the names of the chemical reactions (forward + backward), in the form `'(Rf-k) equation'` and `'(Rb-k) equation'`.
